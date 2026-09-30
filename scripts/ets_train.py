r"""
ets_train.py - the cross-validation harness every configuration runs through.

The training script hands this module a feature transform and a model factory
and gets back identical bookkeeping for every configuration. That is the
point: if two configurations were evaluated by separate code, any difference
in their scores could be a difference in evaluation rather than in the model,
and the comparison would prove nothing.

THE PROTOCOL
------------
1. Participants are split into folds. Never windows - see ets_data.participant_folds.
   The default is leave-one-subject-out (LOSO): every participant is a fold
   of one. Saeb et al. (2017) showed that anything short of a subject-wise
   split overstates performance on new people, and with a few dozen
   participants LOSO is affordable. --folds K gives K-fold for quick runs.
2. Within each training fold a few participants are held out again, as an INNER
   validation set used for early stopping and for choosing the decision
   threshold. The test fold is not touched by either.
3. The chosen threshold is applied unchanged to the test fold.
4. Test windows are the non-overlapping ones only, even when training used
   overlapping windows, so every configuration is scored on the same windows.
5. Every held-out participant is also scored on their own, so two runs can be
   compared participant by participant (ets_eval.paired_comparison).

Steps 2 and 4 are what make the resulting number an estimate of performance on
a participant the model has never seen.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Keras is noisy on import and TensorFlow prints CPU feature banners; neither
# says anything the user needs while a training run is starting.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ets_data
import ets_eval


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def describe_device() -> str:
    """Report whether a GPU was actually found.

    Worth printing every run: TensorFlow 2.16 has NO GPU support on native
    Windows (Google stopped shipping Windows GPU builds after 2.10, and
    `tensorflow-intel` is CPU-only), so a machine with a working GPU can
    silently train on the CPU. The models here are small enough that CPU is
    fine, but nobody should have to guess which one ran.
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU x{len(gpus)} ({gpus[0].name})"
    return ("CPU (no GPU visible - expected on native Windows, where "
            "TensorFlow >= 2.11 is CPU-only; use WSL2 for GPU)")


def aligned_mask(windows: ets_data.WindowSet) -> np.ndarray:
    """Windows that lie on a NON-OVERLAPPING grid.

    When training uses a half-window hop, consecutive windows share half their
    samples. Scoring on those would count the same seconds twice and inflate
    the result, and would make runs trained at different hops score on
    different windows. So evaluation always uses this subset.
    """
    if abs(windows.hop_seconds - windows.window_seconds) < 1e-9:
        return np.ones(len(windows), dtype=bool)

    mask = np.zeros(len(windows), dtype=bool)
    for participant in np.unique(windows.participants):
        rows = np.flatnonzero(windows.participants == participant)
        order = rows[np.argsort(windows.starts[rows])]
        origin = windows.starts[order[0]]
        offsets = windows.starts[order] - origin
        steps = np.round(offsets / windows.window_seconds, 6)
        mask[order[np.isclose(steps, np.round(steps))]] = True
    return mask


@dataclass
class FoldResult:
    fold: int
    held_out: List[str]
    threshold: float
    metrics: Dict[str, float]
    smoothed_metrics: Dict[str, float]
    probabilities: np.ndarray
    indices: np.ndarray
    history: Dict[str, List[float]] = field(default_factory=dict)
    participant_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    inner_participants: List[str] = field(default_factory=list)
    epochs: int = 0
    best_epoch: int = 0
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
# cross-validation options, shared by every training script
# --------------------------------------------------------------------------- #
def add_cv_arguments(parser) -> None:
    parser.add_argument("--folds", type=int, default=None,
                        help="K-fold participant cross-validation instead of "
                             "the default leave-one-subject-out. Faster; use "
                             "for quick checks, LOSO for reported numbers")
    parser.add_argument("--inner-participants", type=int, default=2,
                        help="participants held out of each training fold for "
                             "early stopping and the threshold (default 2)")
    parser.add_argument("--inner-selection", choices=("auto", "first", "rotate"),
                        default="auto",
                        help="which training participants become the inner "
                             "validation set. 'first' is the original "
                             "alphabetical choice; 'rotate' spreads that job "
                             "around. auto = rotate under LOSO, first otherwise")


def is_loso(args) -> bool:
    return getattr(args, "folds", None) in (None, 0)


def resolve_inner_selection(args) -> str:
    mode = getattr(args, "inner_selection", "auto")
    if mode == "auto":
        return "rotate" if is_loso(args) else "first"
    return mode


def cv_tag(args) -> str:
    return "loso" if is_loso(args) else f"{args.folds}fold"


def choose_inner_participants(train_participants: Sequence[str], count: int,
                              fold_index: int, seed: int, mode: str
                              ) -> np.ndarray:
    """The inner validation participants for one fold, never the test ones.

    'first' takes the first `count` in sorted order - what every 4-fold result
    before LOSO used, kept so those stay reproducible. Under LOSO it would hand
    the threshold decision to the same two people in almost every fold, so
    'rotate' orders participants by a seeded hash and starts at a different
    place in that order each fold. Deterministic either way.
    """
    unique = np.unique(np.asarray(train_participants))
    count = min(count, max(len(unique) - 1, 0))
    if mode == "first":
        return unique[:count]
    if mode != "rotate":
        raise ValueError(f"unknown inner selection {mode!r}")
    keyed = sorted(unique, key=lambda p: hashlib.md5(
        f"{seed}:{p}".encode()).hexdigest())
    start = (fold_index * count) % len(keyed)
    return np.asarray([keyed[(start + i) % len(keyed)] for i in range(count)])


def fold_records(results: Sequence["FoldResult"]) -> List[Dict]:
    return [{"fold": r.fold, "held_out": r.held_out, "threshold": r.threshold,
             "inner_participants": r.inner_participants, "epochs": r.epochs,
             "best_epoch": r.best_epoch,
             "seconds": round(r.seconds, 1), "metrics": r.metrics,
             "smoothed_metrics": r.smoothed_metrics} for r in results]


def participant_records(results: Sequence["FoldResult"]
                        ) -> Dict[str, Dict[str, float]]:
    """Every held-out participant's metrics, from whichever fold held them."""
    merged: Dict[str, Dict[str, float]] = {}
    for result in results:
        merged.update(result.participant_metrics)
    return dict(sorted(merged.items()))


def class_weights(y: np.ndarray) -> Dict[int, float]:
    """Weight the rarer class up, so a model cannot do well by ignoring it.

    Per-session eating fraction ranges from 17% to 74%, so which class is rare
    depends on the fold. Computing the weights per fold rather than fixing them
    keeps that from mattering.
    """
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    total = positives + negatives
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    return {0: total / (2.0 * negatives), 1: total / (2.0 * positives)}


def cross_validate(
        windows: ets_data.WindowSet,
        transform: Callable[[np.ndarray], np.ndarray],
        build_model: Callable[[Tuple[int, ...]], "object"],
        *,
        n_folds: int = 4,
        loso: bool = False,
        inner_selection: str = "first",
        seed: int = 9,
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 10,
        # Two. It was briefly raised to three to steady a wild threshold (0.67
        # on one fold), but a held-out participant is ~7% of the training data
        # here, and both the FFT baseline (pooled F1 0.777 -> 0.765) and the
        # model (0.853 -> 0.828) lost ground when it went up. Threshold stability is now
        # handled where it belongs - in ets_eval.choose_threshold, which takes
        # the centre of the plateau rather than the argmax - so it no longer has
        # to be bought with training data.
        inner_validation_participants: int = 2,
        smoothing_kernel: int = 3,
        auxiliary: bool = False,
        n_models: int = 1,
        monitor: str = "loss",
        threshold_objective: str = "f1",
        max_false_alarms: float = 15.0,
        verbose: bool = True) -> Tuple[List[FoldResult], np.ndarray]:
    """Run participant-level cross-validation and return per-fold results.

    `transform` converts raw windows into model input. `build_model` returns a
    compiled Keras model. When `auxiliary` is set the model is expected to have
    two outputs (food probability, chew count) and is fed both targets.
    """
    import tensorflow as tf

    features = transform(windows.X)
    evaluate_on = aligned_mask(windows)
    if loso:
        n_folds = len(np.unique(windows.participants))
    folds = ets_data.participant_folds(windows.participants, n_folds, seed)

    if verbose:
        print(f"\ndevice: {describe_device()}")
        print(f"input shape: {features.shape[1:]}")
        print(f"evaluating on {int(evaluate_on.sum()):,} non-overlapping "
              f"windows of {len(windows):,} total")
        scheme = "leave-one-subject-out" if loso else f"{n_folds}-fold"
        print(f"validation: {scheme}, {len(folds)} folds, inner validation = "
              f"{inner_validation_participants} participants ({inner_selection})")
    started = time.time()

    # The auxiliary target is chews per window, scaled to a comparable range so
    # its loss does not dominate the classification loss purely by magnitude.
    chew_scale = float(np.percentile(windows.chews, 99)) or 1.0
    chew_target = (windows.chews / chew_scale).astype(np.float32)

    # What early stopping watches on the inner validation participants.
    #   loss: total validation loss. With auxiliary training that includes the
    #         chew-count error, and even alone BCE can RISE while the ranking of
    #         eating above non-eating keeps improving (the model grows more
    #         confident on the few windows it gets wrong). On the 61-session
    #         data, three of five folds kept their epoch-1 weights this way.
    #   auc:  area under the ROC curve of the eating output - threshold-free,
    #         and exactly what the threshold chosen next depends on.
    if monitor == "auc":
        monitor_key = "val_food_auc" if auxiliary else "val_auc"
        monitor_mode = "max"
    elif monitor == "loss":
        monitor_key, monitor_mode = "val_loss", "min"
    else:
        raise ValueError(f"unknown monitor {monitor!r}")

    results: List[FoldResult] = []
    pooled_probability = np.full(len(windows), np.nan, dtype=np.float64)

    for index, held_out in enumerate(folds):
        train_rows, test_rows = ets_data.fold_indices(windows, held_out)
        test_rows = test_rows[evaluate_on[test_rows]]

        # Inner split: hold out whole participants again, for early stopping and
        # the threshold. Taking a random slice of windows instead would leak -
        # the neighbouring window of the same meal is nearly the same window.
        fold_started = time.time()
        train_participants = np.unique(windows.participants[train_rows])
        inner_held = choose_inner_participants(
            train_participants, inner_validation_participants, index, seed,
            inner_selection)
        inner_mask = np.isin(windows.participants[train_rows], inner_held)
        fit_rows = train_rows[~inner_mask]
        validation_rows = train_rows[inner_mask]

        # An ensemble of the SAME architecture at different seeds. With ~5-8
        # hours of data a single network's decision boundary depends noticeably
        # on initialisation - the fold-to-fold spread is +-0.05 F1 - and
        # averaging several runs cancels part of that variance. It adds no
        # capacity and no new assumption, only stability, which is why it is
        # defensible on a dataset this size where a bigger model would not be.
        models = []
        for member in range(n_models):
            set_seed(seed + index * 100 + member)
            models.append(build_model(features.shape[1:]))
        model = models[0]

        y_fit = windows.y[fit_rows].astype(np.float32)
        y_validation = windows.y[validation_rows].astype(np.float32)

        if auxiliary:
            targets_fit = {"food": y_fit, "chews": chew_target[fit_rows]}
            targets_validation = {"food": y_validation,
                                  "chews": chew_target[validation_rows]}
            # Keras applies class_weight per output only for single-output
            # models, so the imbalance is handled by sample weights instead.
            weights = class_weights(windows.y[fit_rows])
            sample_weight = {
                "food": np.where(y_fit == 1, weights[1], weights[0]).astype(np.float32),
                "chews": np.ones_like(y_fit),
            }
            fit_kwargs = dict(sample_weight=sample_weight)
        else:
            targets_fit, targets_validation = y_fit, y_validation
            fit_kwargs = dict(class_weight=class_weights(windows.y[fit_rows]))

        def make_callbacks():
            """Fresh callbacks per ensemble member. Callbacks carry state -
            EarlyStopping keeps the best weights and the wait counter - so
            reusing one instance across members would let the first member's
            history stop the second before it had trained."""
            return [
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_key, mode=monitor_mode, patience=patience,
                    restore_best_weights=True, verbose=0),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_key, mode=monitor_mode, factor=0.5,
                    patience=max(3, patience // 3), min_lr=1e-5, verbose=0),
            ]

        history = None
        for member, member_model in enumerate(models):
            member_history = member_model.fit(
                features[fit_rows], targets_fit,
                validation_data=(features[validation_rows], targets_validation),
                epochs=epochs, batch_size=batch_size,
                callbacks=make_callbacks(), verbose=0, **fit_kwargs)
            if history is None:
                history = member_history

        def predict(rows: np.ndarray) -> np.ndarray:
            """Food-intake probability for these rows.

            Calls the model directly instead of model.predict(). predict()
            builds a compiled tf.function keyed on the input signature, and
            each fold feeds it a differently sized array, so TensorFlow retraces
            and prints a retracing warning several times per run. Calling the
            model eagerly in fixed-size batches avoids both the warning and the
            repeated compilation.
            """
            if len(rows) == 0:
                return np.zeros(0, dtype=np.float32)
            total = np.zeros(len(rows), dtype=np.float64)
            for member_model in models:
                chunks = []
                for start in range(0, len(rows), 256):
                    batch = features[rows[start:start + 256]]
                    output = member_model(batch, training=False)
                    if isinstance(output, dict):
                        output = output["food"]
                    elif isinstance(output, (list, tuple)):
                        output = output[0]
                    chunks.append(np.asarray(output).ravel())
                total += np.concatenate(chunks)
            return total / len(models)

        # Threshold from the inner validation participants only.
        #
        # ALL their windows, including the overlapping ones. Restricting this to
        # the non-overlapping grid (to mirror test conditions exactly) halves
        # the data the threshold is estimated from, and a threshold is a single
        # scalar: overlapping windows do not bias it, they only weight some
        # stretches of a meal slightly more. Measured: restricting it cost the
        # overlapping-hop model more than the FFT baseline, which trains on a
        # hop equal to its window and so was not affected at all.
        validation_probability = predict(validation_rows)
        threshold = ets_eval.select_threshold(
            windows.y[validation_rows], validation_probability,
            threshold_objective, windows.window_seconds, max_false_alarms)

        # Smoothing is different: the median filter walks along consecutive
        # windows, so its threshold MUST be estimated at the same spacing the
        # test fold is smoothed at. Selected on the non-overlapping grid.
        validation_eval_rows = validation_rows[evaluate_on[validation_rows]]
        smoothed_eval_probability = predict(validation_eval_rows)

        # Smoothing needs its OWN threshold. A median filter pulls each value
        # toward its neighbours, which shifts the probability distribution;
        # reusing the unsmoothed threshold measures that mismatch rather than
        # the effect of smoothing, and made smoothing look harmful (F1 0.777 ->
        # 0.730 on the first real run) when it had not been given a fair test.
        smoothed_validation = ets_eval.smooth_predictions(
            smoothed_eval_probability, windows.starts[validation_eval_rows],
            windows.participants[validation_eval_rows],
            windows.window_seconds, windows.window_seconds,
            kernel=smoothing_kernel)
        smoothed_threshold = ets_eval.select_threshold(
            windows.y[validation_eval_rows], smoothed_validation,
            threshold_objective, windows.window_seconds, max_false_alarms)

        test_probability = predict(test_rows)
        pooled_probability[test_rows] = test_probability

        fold_metrics = ets_eval.metrics(windows.y[test_rows],
                                        test_probability, threshold)
        smoothed = ets_eval.smooth_predictions(
            test_probability, windows.starts[test_rows],
            windows.participants[test_rows], windows.window_seconds,
            windows.window_seconds, kernel=smoothing_kernel)
        smoothed_metrics = ets_eval.metrics(windows.y[test_rows], smoothed,
                                            smoothed_threshold)

        epochs_run = len(history.history["loss"])
        monitored = history.history.get(monitor_key, [])
        best_epoch = (int(np.argmax(monitored) if monitor_mode == "max"
                          else np.argmin(monitored)) + 1) if monitored else epochs_run
        results.append(FoldResult(
            fold=index + 1, held_out=[str(p) for p in held_out],
            threshold=threshold,
            metrics=fold_metrics, smoothed_metrics=smoothed_metrics,
            probabilities=test_probability, indices=test_rows,
            history={k: [float(v) for v in vals]
                     for k, vals in history.history.items()},
            participant_metrics=ets_eval.per_participant(
                windows.y[test_rows], test_probability,
                windows.participants[test_rows], threshold),
            inner_participants=[str(p) for p in inner_held],
            epochs=epochs_run, best_epoch=best_epoch,
            seconds=time.time() - fold_started))

        if verbose:
            members = f" x{len(models)}" if len(models) > 1 else ""
            smoothed_note = (f"(smoothed {smoothed_metrics['f1']:.3f})  "
                             if smoothing_kernel > 1 else "")
            elapsed = time.time() - started
            remaining = elapsed / (index + 1) * (len(folds) - index - 1)
            print(f"  fold {index + 1:>2}/{len(folds)}{members}  held out "
                  f"{', '.join(map(str, held_out)):<22} "
                  f"n={len(test_rows):>5,}  F1 {fold_metrics['f1']:.3f} "
                  f"{smoothed_note}acc {fold_metrics['accuracy']:.3f}  "
                  f"thr {threshold:.2f}  best ep {best_epoch}/{epochs_run}  "
                  f"{time.time() - fold_started:.0f}s  "
                  f"(~{remaining / 60:.0f} min left)", flush=True)

    return results, pooled_probability

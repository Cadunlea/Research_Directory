r"""
ets_train.py - the cross-validation harness both models run through.

Both training scripts hand this module a feature transform and a model factory
and get back identical bookkeeping. That is the point: if the two models were
evaluated by separate code, any difference in their scores could be a
difference in evaluation rather than in the model, and the comparison would
prove nothing.

THE PROTOCOL
------------
1. Participants are split into folds. Never windows - see ets_data.participant_folds.
2. Within each training fold a few participants are held out again, as an INNER
   validation set used for early stopping and for choosing the decision
   threshold. The test fold is not touched by either.
3. The chosen threshold is applied unchanged to the test fold.
4. Test windows are the non-overlapping ones only, even when training used
   overlapping windows, so every model is scored on the same windows the
   previous model was scored on.

Steps 2 and 4 are what make the resulting number an estimate of performance on
a participant the model has never seen.
"""

from __future__ import annotations

import os
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
    the result; it would also stop the numbers being comparable with the
    previous model, which used non-overlapping windows. So evaluation always
    uses this subset.
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
        seed: int = 9,
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 10,
        inner_validation_participants: int = 2,
        smoothing_kernel: int = 3,
        auxiliary: bool = False,
        verbose: bool = True) -> Tuple[List[FoldResult], np.ndarray]:
    """Run participant-level cross-validation and return per-fold results.

    `transform` converts raw windows into model input. `build_model` returns a
    compiled Keras model. When `auxiliary` is set the model is expected to have
    two outputs (food probability, chew count) and is fed both targets.
    """
    import tensorflow as tf

    features = transform(windows.X)
    evaluate_on = aligned_mask(windows)
    folds = ets_data.participant_folds(windows.participants, n_folds, seed)

    if verbose:
        print(f"\ndevice: {describe_device()}")
        print(f"input shape: {features.shape[1:]}")
        print(f"evaluating on {int(evaluate_on.sum()):,} non-overlapping "
              f"windows of {len(windows):,} total")

    # The auxiliary target is chews per window, scaled to a comparable range so
    # its loss does not dominate the classification loss purely by magnitude.
    chew_scale = float(np.percentile(windows.chews, 99)) or 1.0
    chew_target = (windows.chews / chew_scale).astype(np.float32)

    results: List[FoldResult] = []
    pooled_probability = np.full(len(windows), np.nan, dtype=np.float64)

    for index, held_out in enumerate(folds):
        train_rows, test_rows = ets_data.fold_indices(windows, held_out)
        test_rows = test_rows[evaluate_on[test_rows]]

        # Inner split: hold out whole participants again, for early stopping and
        # the threshold. Taking a random slice of windows instead would leak -
        # the neighbouring window of the same meal is nearly the same window.
        train_participants = np.unique(windows.participants[train_rows])
        inner_held = train_participants[:inner_validation_participants]
        inner_mask = np.isin(windows.participants[train_rows], inner_held)
        fit_rows = train_rows[~inner_mask]
        validation_rows = train_rows[inner_mask]

        set_seed(seed + index)
        model = build_model(features.shape[1:])

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

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(3, patience // 3),
                min_lr=1e-5, verbose=0),
        ]

        history = model.fit(
            features[fit_rows], targets_fit,
            validation_data=(features[validation_rows], targets_validation),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            verbose=0, **fit_kwargs)

        def predict(rows: np.ndarray) -> np.ndarray:
            output = model.predict(features[rows], batch_size=256, verbose=0)
            if isinstance(output, (list, tuple)):
                output = output[0]
            elif isinstance(output, dict):
                output = output["food"]
            return np.asarray(output).ravel()

        # Threshold from the inner validation participants only.
        validation_probability = predict(validation_rows)
        threshold = ets_eval.choose_threshold(
            windows.y[validation_rows], validation_probability, "f1")

        test_probability = predict(test_rows)
        pooled_probability[test_rows] = test_probability

        fold_metrics = ets_eval.metrics(windows.y[test_rows],
                                        test_probability, threshold)
        smoothed = ets_eval.smooth_predictions(
            test_probability, windows.starts[test_rows],
            windows.participants[test_rows], windows.window_seconds,
            windows.window_seconds, kernel=smoothing_kernel)
        smoothed_metrics = ets_eval.metrics(windows.y[test_rows], smoothed,
                                            threshold)

        results.append(FoldResult(
            fold=index + 1, held_out=list(held_out), threshold=threshold,
            metrics=fold_metrics, smoothed_metrics=smoothed_metrics,
            probabilities=test_probability, indices=test_rows,
            history={k: [float(v) for v in vals]
                     for k, vals in history.history.items()}))

        if verbose:
            epochs_run = len(history.history["loss"])
            print(f"  fold {index + 1}  held out {', '.join(held_out):<46} "
                  f"n={len(test_rows):>5,}  F1 {fold_metrics['f1']:.3f} "
                  f"(smoothed {smoothed_metrics['f1']:.3f})  "
                  f"acc {fold_metrics['accuracy']:.3f}  thr {threshold:.2f}  "
                  f"{epochs_run} epochs")

    return results, pooled_probability

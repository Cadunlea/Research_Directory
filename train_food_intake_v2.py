r"""
train_food_intake_v2.py - MODEL B, the one built to win.

Every change from Model A is listed here with the reason it is expected to
help, because every one of them has to survive the question "why did you do
that?".

1. TIME-DOMAIN INPUT INSTEAD OF FFT MAGNITUDE
   Model A throws away phase: an FFT magnitude spectrum says which rhythms are
   present in 8 seconds but not how they are arranged in time. Chewing is a
   rhythmic ENVELOPE and a bite is a transient, and the magnitude spectrum
   smears both. Feeding the filtered time series lets the convolutions learn
   their own frequency selectivity AND keep the temporal structure. This is
   also what the 4/22 lab-meeting deck proposed.

2. CROSS-CHANNEL ATTENTION  (squeeze-and-excitation)
   The optical channel carries chewing; the accelerometers carry head and body
   motion that is sometimes signal and often artefact. Which channel deserves
   attention depends on the moment and the participant, so the network is given
   an explicit, cheap mechanism to reweight them per window rather than having
   to encode that in the convolution filters.

3. ATTENTION POOLING INSTEAD OF GLOBAL AVERAGE
   Global average pooling dilutes a short event across the whole window: with
   8 seconds and one bite, averaging buries it. Attention pooling learns which
   time steps matter and weights them, which is the natural fit for sparse,
   brief events inside a long window.

4. AUXILIARY CHEW-COUNT HEAD
   A second head regresses the number of CHGT chew marks in the window. The
   binary label carries one bit per window, and with ~5 hours of annotated data
   that is very little supervision. Forcing the shared trunk to also explain
   HOW MANY chews occurred makes it represent the chewing rhythm rather than
   memorising participant-specific amplitude quirks. The head is discarded at
   inference; it exists only to shape the features. (Multi-task learning.)

5. OVERLAPPING WINDOWS AT TRAINING TIME ONLY
   A half-window hop roughly doubles the training windows at no cost in data
   collection. Evaluation still uses the non-overlapping grid, so the numbers
   stay comparable with Model A and with the previous model - see
   ets_train.aligned_mask.

6. TEMPORAL SMOOTHING OF THE OUTPUT - TRIED, AND IT DOES NOT WORK HERE
   The idea was that eating is contiguous, so an isolated positive window is
   probably an error a median filter would remove. On the real data it LOWERED
   both precision and recall (F1 0.853 -> 0.770 at 8 s). Losing both is the
   signature of destroyed information rather than a mis-set threshold: a median
   over three 8 s windows spans 24 s and assumes contiguity the labels do not
   have. Eating here is `bout OR bite`, and real meals are punctuated - bite,
   chew, swallow, pause, talk, bite - so at 8 s resolution the label sequence
   genuinely flips on and off, and the filter erases short but real bouts. This
   is the same fact the lab's own pause metrics (PADU) exist to measure.
   Kept, off by default in spirit (--smoothing 1 disables it), and still
   reported so the negative result is visible rather than buried.

7. CONTEXT WINDOWS - the principled version of what smoothing attempted
   --context-seconds widens the SIGNAL the model sees either side of the
   labelled window while leaving the label alone. Rather than asserting
   afterwards that neighbouring windows should agree, it lets the model look at
   them and learn how much they matter.

   FIRST ATTEMPT FAILED, and the reason is instructive. Plain 8 s context cost
   F1 0.854 -> 0.775, precision 0.876 -> 0.683, while recall ROSE to 0.894. The
   model had 24 s of signal and no way to tell which 8 s it was being asked
   about, so it learned the easier question - eating SOMEWHERE in the 24 s -
   and fired on every window next to a meal. A marker channel (see
   mark_target_window) now identifies the labelled window. Whether that is
   enough is an open question, not a claim.

8. SEED ENSEMBLING
   --ensemble N trains N copies at different initialisations and averages them.
   Fold-to-fold spread is +-0.05 F1 on this dataset, a good part of which is
   initialisation noise rather than genuine difficulty; averaging cancels some
   of it. It adds no capacity and no new assumption, only stability - which is
   why it is defensible at this data size when a bigger model would not be.

DELIBERATELY NOT DONE
    No transformer, no pretrained backbone. With roughly 8 hours of annotated
    data a large model would overfit and be harder to defend, not easier. Every
    gain here comes from representing the data properly, not from parameter
    count. (Seed ensembling is not extra capacity: every member is the same
    small network.)

USAGE
-----
    python train_food_intake_v2.py --config path\to\jitai_config.ini
    python train_food_intake_v2.py --config ... --sweep-windows
    python train_food_intake_v2.py --config ... --no-auxiliary   (ablation)
    python train_food_intake_v2.py --config ... --window-seconds 4

    the two improvements, most promising first:
    python train_food_intake_v2.py --config ... --context-seconds 8
    python train_food_intake_v2.py --config ... --context-seconds 8 --ensemble 3
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ets_data
import ets_eval
import ets_train

OUTPUT_DIR = Path("model_v2_outputs")

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 80
PATIENCE = 12
DROPOUT = 0.3

# Window lengths for the resolution sweep. Powers of two in seconds, as the
# 128 Hz sampling rate and the 8 s packet size both are.
SWEEP_WINDOWS = (2.0, 4.0, 8.0, 16.0)


def mark_target_window(features: np.ndarray, windows) -> np.ndarray:
    """Append a channel that is 1 over the LABELLED window and 0 over context.

    Without it, context makes the model worse, badly: 8 s of context either
    side cost F1 0.854 -> 0.775, with precision collapsing 0.876 -> 0.683 while
    recall ROSE to 0.894. That pattern is the giveaway. Given 24 s of signal and
    no indication of which 8 s it is being asked about, and attention pooling
    free to attend anywhere in them, the model answers an easier question -
    "is there eating SOMEWHERE in this 24 s?" - and so fires on every window
    adjacent to a meal, which is precisely where the false positives appeared.

    One extra channel tells it where the question is. The context is still
    available to disambiguate; it is no longer mistakable for the subject.
    """
    if not getattr(windows, "context_seconds", 0.0):
        return features
    total = features.shape[1]
    context = int(round(windows.context_seconds * ets_data.SENSOR_FS))
    marker = np.zeros((1, total, 1), dtype=np.float32)
    marker[0, context:total - context, 0] = 1.0
    return np.concatenate(
        [features, np.repeat(marker, features.shape[0], axis=0)], axis=2)


def time_domain_features(X: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score with the mean removed and -1 excluded.

    high_pass=True removes the per-window DC level: the optical sensor's
    baseline varies enormously between participants and with how the device
    sits on the face, and none of that variation is information about chewing.
    Removing it stops the network learning 'this participant reads 3500' as a
    shortcut that will not transfer.
    """
    return ets_data.normalise(X, high_pass=True)


def build_model(input_shape, auxiliary: bool = True, attention: bool = True):
    """Two convolution blocks -> channel attention -> attention pooling -> heads.

    `attention=False` is the ablation: both attention mechanisms are replaced by
    plain global average pooling, leaving the same convolution trunk. It
    isolates what the attention actually contributes, rather than leaving it
    bundled with everything else Model B changed at once.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=input_shape, name="signal")

    # Strided convolutions rather than pooling: at 128 Hz an 8 s window is 1024
    # samples, and the interesting rhythm is 1-2 Hz, so aggressive downsampling
    # early costs nothing and keeps the model small.
    x = layers.Conv1D(32, 9, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 9, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    features = layers.ReLU()(x)

    if not attention:
        # The ablation: average over time, the way Model A does.
        pooled = layers.GlobalAveragePooling1D(name="average_pool")(features)
        shared = layers.Dense(64, activation="relu")(pooled)
        shared = layers.Dropout(DROPOUT)(shared)
        return _heads(inputs, shared, auxiliary, "ets_cnn_v2_no_attention")

    # --- cross-channel attention -------------------------------------------
    # Squeeze each learned channel to one number, learn a gate from those, and
    # rescale. This is the cheapest form of the deck's 'cross-channel
    # attention': a handful of parameters, no sequence-length dependence.
    squeezed = layers.GlobalAveragePooling1D()(features)
    gate = layers.Dense(max(4, features.shape[-1] // 4), activation="relu")(squeezed)
    gate = layers.Dense(features.shape[-1], activation="sigmoid")(gate)
    gate = layers.Reshape((1, features.shape[-1]))(gate)
    recalibrated = layers.Multiply(name="channel_attention")([features, gate])

    # --- attention pooling --------------------------------------------------
    # One score per time step, softmax over time, weighted sum. Lets a two
    # second bite dominate a sixteen second window instead of being averaged
    # into the background.
    scores = layers.Conv1D(1, 1)(recalibrated)
    scores = layers.Softmax(axis=1, name="temporal_attention")(scores)
    pooled = layers.Multiply()([recalibrated, scores])
    pooled = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                           output_shape=(recalibrated.shape[-1],),
                           name="attention_pool")(pooled)

    shared = layers.Dense(64, activation="relu")(pooled)
    shared = layers.Dropout(DROPOUT)(shared)
    return _heads(inputs, shared, auxiliary, "ets_cnn_v2")


def _heads(inputs, shared, auxiliary: bool, name: str):
    """The output heads and compilation, shared by both trunk variants so an
    ablation cannot accidentally differ in how it is trained."""
    import tensorflow as tf
    from tensorflow.keras import layers

    food = layers.Dense(1, activation="sigmoid", name="food")(shared)

    if not auxiliary:
        model = tf.keras.Model(inputs, food, name=name)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # Auxiliary head: chews per window, scaled in ets_train. softplus keeps it
    # non-negative, which a count is.
    chews = layers.Dense(32, activation="relu")(shared)
    chews = layers.Dense(1, activation="softplus", name="chews")(chews)

    # Outputs as a DICT, not a list. Keras 3 matches targets, losses, metrics
    # and sample weights to outputs by name only when the model's outputs are a
    # dict; with a list it matches by position and a dict of targets fails with
    # a bare KeyError: 0.
    model = tf.keras.Model(inputs, {"food": food, "chews": chews},
                           name=f"{name}_multitask")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={"food": "binary_crossentropy", "chews": "mse"},
        # The auxiliary loss is a regulariser, not the objective. Weighted well
        # below the classification loss so it shapes the features without
        # competing with the task being measured.
        loss_weights={"food": 1.0, "chews": 0.2},
        metrics={"food": ["accuracy"]})
    return model


def run(args, window_seconds: float, auxiliary: bool) -> Dict:
    hop = window_seconds / 2.0 if args.overlap else window_seconds
    windows = ets_data.get_windows(args, window_seconds, hop_seconds=hop,
                                   cache_dir=args.cache_dir,
                                   rebuild=args.rebuild_cache,
                                   context_seconds=args.context_seconds)

    if args.fft_input:
        # The representation ablation: Model B's architecture fed Model A's
        # features. Isolates what the time-domain input contributes, with the
        # attention, the auxiliary head and the augmentation all held fixed.
        def transform(X):
            return ets_data.fft_features(X)
    else:
        def transform(X):
            return mark_target_window(time_domain_features(X), windows)

    results, _ = ets_train.cross_validate(
        windows, transform,
        lambda shape: build_model(shape, auxiliary=auxiliary,
                                  attention=not args.no_attention),
        n_folds=args.folds, seed=args.seed, epochs=args.epochs,
        batch_size=BATCH_SIZE, patience=PATIENCE, auxiliary=auxiliary,
        smoothing_kernel=args.smoothing, n_models=args.ensemble,
        loso=args.loso,
        inner_selection=ets_train.resolve_inner_selection(args), verbose=True)

    extras = ""
    if args.fft_input:
        extras += " + FFT input (not time domain)"
    if args.no_attention:
        extras += " (no attention)"
    if args.context_seconds:
        extras += f" + {args.context_seconds:g}s context"
    if args.ensemble > 1:
        extras += f" + {args.ensemble}-model ensemble"
    label = (f"MODEL B  time-domain CNN + attention"
             f"{' + auxiliary chew head' if auxiliary else ' (no auxiliary head)'}"
             f"{extras}  ({window_seconds:g}s windows, "
             f"{'LOSO' if args.loso else f'{args.folds}-fold'})")
    summary = ets_eval.report(label, [r.metrics for r in results])
    smoothed = ets_eval.report(f"{label}  + temporal smoothing",
                               [r.smoothed_metrics for r in results])

    return {
        "window_seconds": window_seconds,
        "hop_seconds": hop,
        "context_seconds": args.context_seconds,
        "ensemble": args.ensemble,
        "auxiliary": auxiliary,
        "attention": not args.no_attention,
        "fft_input": args.fft_input,
        "windows": windows.summary(),
        "cv": ets_train.cv_tag(args),
        "inner_selection": ets_train.resolve_inner_selection(args),
        "folds": ets_train.fold_records(results),
        "participants": ets_train.participant_records(results),
        "summary": summary,
        "summary_smoothed": smoothed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--sweep-windows", action="store_true",
                        help=f"train at each of {SWEEP_WINDOWS} and compare")
    parser.add_argument("--no-auxiliary", action="store_true",
                        help="ablation: drop the chew-count head")
    parser.add_argument("--no-attention", action="store_true",
                        help="ablation: replace cross-channel attention and "
                             "attention pooling with global average pooling")
    parser.add_argument("--fft-input", action="store_true",
                        help="ablation: feed Model A's FFT magnitude features "
                             "to Model B's architecture, isolating what the "
                             "time-domain representation contributes")
    parser.add_argument("--no-overlap", dest="overlap", action="store_false",
                        help="ablation: train on non-overlapping windows only")
    parser.add_argument("--context-seconds", type=float, default=0.0,
                        help="extra signal shown to the model either side of "
                             "the labelled window (the label is unchanged). "
                             "Try 8: it is the principled replacement for "
                             "post-hoc median smoothing")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="train N models per fold at different seeds and "
                             "average them (try 3); N times slower")
    parser.add_argument("--smoothing", type=int, default=3,
                        help="median filter width in windows (1 disables). "
                             "Measured HARMFUL at 8 s on the real data: eating "
                             "is punctuated by pauses at that resolution, so "
                             "the filter erases short genuine bouts. Prefer "
                             "--context-seconds")
    ets_train.add_cv_arguments(parser)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-dir", default="ets_cache")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auxiliary = not args.no_auxiliary

    print("MODEL B - time-domain CNN with cross-channel and temporal attention\n")

    runs: List[Dict] = []
    if args.sweep_windows:
        for window_seconds in SWEEP_WINDOWS:
            print(f"\n{'#' * 74}\n#  window = {window_seconds:g} s\n{'#' * 74}")
            runs.append(run(args, window_seconds, auxiliary))

        print(f"\n{'=' * 74}\n  WINDOW LENGTH SWEEP\n{'=' * 74}")
        print(f"\n  {'window':>8}{'windows':>10}{'F1':>9}{'F1 smooth':>11}"
              f"{'accuracy':>10}{'bal acc':>9}")
        for entry in runs:
            summary = entry["summary"]
            print(f"  {entry['window_seconds']:>7g}s"
                  f"{summary['n']:>10,}"
                  f"{summary['pooled_f1']:>9.3f}"
                  f"{entry['summary_smoothed']['pooled_f1']:>11.3f}"
                  f"{summary['pooled_accuracy']:>10.3f}"
                  f"{summary['pooled_balanced_accuracy']:>9.3f}")
        best = max(runs, key=lambda e: e["summary_smoothed"]["pooled_f1"])
        print(f"\n  best F1 at {best['window_seconds']:g}s windows: "
              f"{best['summary_smoothed']['pooled_f1']:.3f}")
    else:
        runs.append(run(args, args.window_seconds, auxiliary))

    payload = {"model": "v2_time_domain_attention", "runs": runs,
               "annotation_shift_seconds": 0.0}
    # The name carries the CV scheme and every ablation flag, so runs meant to
    # be compared with compare_runs.py never overwrite one another.
    flags = [ets_train.cv_tag(args)]
    if args.no_auxiliary:
        flags.append("noaux")
    if args.no_attention:
        flags.append("noattn")
    if args.fft_input:
        flags.append("fft")
    if not args.overlap:
        flags.append("nooverlap")
    if args.context_seconds:
        flags.append(f"ctx{args.context_seconds:g}")
    if args.ensemble > 1:
        flags.append(f"ens{args.ensemble}")
    if args.seed != 9:
        flags.append(f"seed{args.seed}")
    suffix = "_".join(flags)
    name = (f"model_v2_sweep_{suffix}.json" if args.sweep_windows
            else f"model_v2_metrics_w{args.window_seconds:g}_{suffix}.json")
    path = output_dir / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

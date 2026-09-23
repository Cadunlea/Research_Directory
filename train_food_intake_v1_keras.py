r"""
train_food_intake_v1_keras.py - MODEL A, the honest baseline.

The previous model's architecture, unchanged, ported from PyTorch to
TensorFlow/Keras and fed from the database instead of annotation CSVs.

WHY THIS EXISTS AT ALL
----------------------
Model B changes the input representation, the architecture, the augmentation
and the evaluation protocol all at once. If only Model B were built, and it
scored better, there would be no way to say WHY - and the first question at a
lab meeting is exactly that. This script isolates the change of data source:
same network, same FFT features, same 8 s windows as before, but with

    labels from study_data.numeric_data instead of Desktop CSVs
    NO -8 s annotation shift (measured as unnecessary - see the module
        docstring of check_annotation_drift.py)
    missing samples handled as missing rather than fed in as the literal -1
    participant-level cross-validation instead of one 2-participant split
    the decision threshold chosen on training data instead of fixed at 0.40

Any difference between this and the old 80.3% / F1 0.788 is attributable to
the data and the protocol. Any further difference in Model B is attributable
to the model.

ARCHITECTURE (unchanged from the PyTorch FrequencyCNN)
    Conv1D  16, kernel 7  -> BatchNorm -> ReLU -> MaxPool 2
    Conv1D  32, kernel 7  -> BatchNorm -> ReLU -> MaxPool 2
    Conv1D  64, kernel 5  -> BatchNorm -> ReLU -> GlobalAveragePooling
    Dense   32 -> ReLU -> Dropout 0.3 -> Dense 1 (sigmoid)

USAGE
-----
    python train_food_intake_v1_keras.py --config path\to\jitai_config.ini
    python train_food_intake_v1_keras.py --config ... --rebuild-cache
    python train_food_intake_v1_keras.py --config ... --window-seconds 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ets_data
import ets_eval
import ets_train

OUTPUT_DIR = Path("model_v1_outputs")

# The old script's values, kept so the port is a port.
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 10
DROPOUT = 0.3


def build_model(input_shape):
    """The previous FrequencyCNN, layer for layer.

    Input is (frequency_bins, 4): Conv1D slides along the frequency axis with
    the four sensors as channels, which is what the PyTorch version did with
    its (4, 513) channels-first tensor.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=input_shape, name="fft")
    x = layers.Conv1D(16, 7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(32, 7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="food")(x)

    model = tf.keras.Model(inputs, outputs, name="frequency_cnn_v1")
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    ets_train.add_cv_arguments(parser)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-dir", default="ets_cache")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("MODEL A - previous architecture, new data source\n")
    windows = ets_data.get_windows(args, args.window_seconds,
                                   hop_seconds=args.window_seconds,
                                   cache_dir=args.cache_dir,
                                   rebuild=args.rebuild_cache)

    results, pooled = ets_train.cross_validate(
        windows, ets_data.fft_features, build_model,
        n_folds=args.folds, seed=args.seed, epochs=args.epochs,
        batch_size=BATCH_SIZE, patience=PATIENCE, loso=args.loso,
        inner_selection=ets_train.resolve_inner_selection(args), verbose=True)

    scheme = ("leave-one-subject-out" if args.loso
              else f"{args.folds}-fold participant CV")
    summary = ets_eval.report(
        f"MODEL A  FFT + CNN  ({args.window_seconds:g}s windows, {scheme})",
        [r.metrics for r in results])

    smoothed = ets_eval.report(
        "MODEL A  with temporal median smoothing",
        [r.smoothed_metrics for r in results])

    payload = {
        "model": "v1_fft_cnn_keras",
        "window_seconds": args.window_seconds,
        "cv": ets_train.cv_tag(args),
        "inner_selection": ets_train.resolve_inner_selection(args),
        "folds": ets_train.fold_records(results),
        "participants": ets_train.participant_records(results),
        "summary": summary,
        "summary_smoothed": smoothed,
        "windows": windows.summary(),
        "annotation_shift_seconds": 0.0,
    }
    path = (output_dir / f"model_v1_metrics_w{args.window_seconds:g}"
                          f"_{ets_train.cv_tag(args)}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

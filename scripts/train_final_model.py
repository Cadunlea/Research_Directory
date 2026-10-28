r"""
train_final_model.py - train the one model that ships, and save it for inference.

Cross-validation estimates how well the method works on new people; it leaves
43 models behind, none of which is THE model. This trains a single model on
every annotated participant, using the procedure that was validated:

  * the default architecture, 8 s windows at 128 Hz, half-overlapping
    training windows, Adam at 0.0003, early stopping on validation ROC AUC;
  * 5 participants held out of training as the validation set, used for
    early stopping and for choosing the decision threshold (the leave-one-out
    comparison found 5 better than 2, and 8 no better than 5);
  * the threshold rule of your choice: best F1 (default) or a false-alarm
    budget per hour of non-eating.

It saves, in --model-dir (default models\food_intake_final):

    model.weights.h5   the trained weights
    model.json         everything needed to rebuild and use it: architecture
                       and input settings, the threshold and how it was
                       chosen, the validation participants and their scores,
                       the training participants, versions and the date

detect_eating.py loads that folder. Weights plus settings, rather than a whole
saved Keras model, because the model contains custom layers that a saved
model file cannot always reload on another machine or Keras version.

USAGE
-----
    python train_final_model.py --config "%CFG%" --threads 8
    python train_final_model.py --config "%CFG%" --threads 8 --threshold-objective false-alarms --max-false-alarms 15 --model-dir models\food_intake_fa15

Needs the database only if the window cache is missing. About as long as one
leave-one-out fold (a few minutes).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ets_data
import ets_eval
import ets_train
import train_food_intake as tfi


def train(config) -> Path:
    import tensorflow as tf

    if config.sequence or config.context_seconds or config.soft_labels:
        raise SystemExit("the final model uses the validated default setup: no "
                         "--sequence, --context-seconds or --soft-labels")
    config = tfi.normalise_config(config)
    hop = config.window_seconds / 2.0 if config.overlap else config.window_seconds
    windows = ets_data.get_windows(config, config.window_seconds, hop_seconds=hop,
                                   cache_dir=config.cache_dir,
                                   rebuild=config.rebuild_cache)

    participants = np.unique(windows.participants)
    validation = ets_train.choose_inner_participants(
        participants, config.inner_participants, 0, config.seed, "rotate")
    is_validation = np.isin(windows.participants, validation)
    fit_rows = np.flatnonzero(~is_validation)
    validation_rows = np.flatnonzero(is_validation)
    print(f"\ntraining on {len(participants) - len(validation)} participants "
          f"({len(fit_rows):,} windows); validation: {', '.join(validation)} "
          f"({len(validation_rows):,} windows)")

    features = tfi.make_transform(config)(windows.X)
    ets_train.set_seed(config.seed)
    builder = tfi.ARCHITECTURES[config.architecture]
    model = builder(features.shape[1:], config)

    hard_fit = windows.y[fit_rows]
    weights = ets_train.class_weights(hard_fit)
    food_weight = np.where(hard_fit == 1, weights[1], weights[0]).astype(np.float32)
    y_fit = hard_fit.astype(np.float32)
    y_validation = windows.y[validation_rows].astype(np.float32)
    auxiliary = config.chew_weight > 0
    chew_scale = None
    if auxiliary:
        chew_scale = float(np.percentile(windows.chews, 99)) or 1.0
        chews = (windows.chews / chew_scale).astype(np.float32)
        targets = {"food": y_fit, "chews": chews[fit_rows]}
        validation_targets = {"food": y_validation, "chews": chews[validation_rows]}
        sample_weight = {"food": food_weight, "chews": np.ones_like(y_fit)}
        monitor = "val_food_auc" if config.monitor == "auc" else "val_loss"
    else:
        targets, validation_targets, sample_weight = y_fit, y_validation, food_weight
        monitor = "val_auc" if config.monitor == "auc" else "val_loss"
    mode = "max" if config.monitor == "auc" else "min"

    started = time.time()
    history = model.fit(
        features[fit_rows], targets, sample_weight=sample_weight,
        validation_data=(features[validation_rows], validation_targets),
        epochs=config.epochs, batch_size=tfi.BATCH_SIZE, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor=monitor, mode=mode, patience=config.patience,
                       restore_best_weights=True, verbose=0),
                   tf.keras.callbacks.ReduceLROnPlateau(
                       monitor=monitor, mode=mode, factor=0.5,
                       patience=max(3, config.patience // 3), min_lr=1e-5,
                       verbose=0)])
    monitored = history.history.get(monitor, [])
    best_epoch = int(np.argmax(monitored) if mode == "max" else np.argmin(monitored)) + 1

    output = model(features[validation_rows], training=False)
    probability = np.asarray(output["food"] if isinstance(output, dict) else output).ravel()
    threshold = ets_eval.select_threshold(
        windows.y[validation_rows], probability, config.threshold_objective,
        config.window_seconds, config.max_false_alarms)

    # Validation scores on the non-overlapping grid, as everywhere else.
    grid = ets_train.aligned_mask(windows)[validation_rows]
    scores = ets_eval.metrics(windows.y[validation_rows][grid], probability[grid],
                              threshold)
    scores["false_alarms_per_hour"] = ets_eval.false_alarms_per_hour(
        scores["fp"], scores["tn"], config.window_seconds)

    out = Path(config.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out / "model.weights.h5"))
    record = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "description": tfi.describe(config),
        "config": {key: getattr(config, key) for key in tfi.CONFIG_KEYS},
        "input_shape": list(features.shape[1:]),
        "threshold": float(threshold),
        "threshold_objective": config.threshold_objective,
        # The chew output is chews per window divided by this.
        "chew_scale": chew_scale,
        "max_false_alarms": config.max_false_alarms,
        "validation_participants": [str(p) for p in validation],
        "training_participants": [str(p) for p in participants
                                  if p not in set(validation)],
        "training_windows": int(len(fit_rows)),
        "best_epoch": best_epoch, "epochs_run": len(history.history["loss"]),
        "validation_scores": {k: float(v) for k, v in scores.items()},
        "training_seconds": round(time.time() - started, 1),
        "tensorflow_version": tf.__version__,
        "note": ("Validation scores are on the 5 participants that chose the "
                 "threshold, so they are optimistic. The unbiased estimate of "
                 "performance on new people is the leave-one-subject-out result."),
    }
    with open(out / "model.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)

    print(f"\nsaved {out / 'model.weights.h5'}\n      {out / 'model.json'}")
    print(f"best epoch {best_epoch}/{record['epochs_run']}; threshold "
          f"{threshold:.3f} ({config.threshold_objective})")
    print(f"validation participants: F1 {scores['f1']:.3f}, precision "
          f"{scores['precision']:.3f}, recall {scores['recall']:.3f}, "
          f"{scores['false_alarms_per_hour']:.1f} false alarms/h (optimistic; "
          f"they chose the threshold)")
    return out


def main() -> int:
    parser = tfi.build_parser()
    parser.description = __doc__.split("\n")[1]
    parser.set_defaults(monitor="auc", learning_rate=0.0003, inner_participants=5)
    parser.add_argument("--model-dir", default=str(Path("models") / "food_intake_final"))
    config = parser.parse_args()
    if config.threads:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(config.threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, config.threads // 2))
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

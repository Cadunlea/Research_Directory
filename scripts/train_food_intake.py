r"""
train_food_intake.py - food-intake detection from AIM-2 sensor signals.

One script for every experiment in the paper. The default configuration is
the model described in docs/literature_review (Figure 1, Table 3); every flag
changes ONE design dimension and holds the rest fixed, so each run is an
ablation or an experiment against that default. Every run writes its own
results file, named after its configuration, so nothing overwrites anything.

THE DEFAULT MODEL, AND WHY EACH PIECE IS THERE
----------------------------------------------
1. TIME-DOMAIN INPUT, NOT A SPECTRUM                      (--input fft)
   A magnitude spectrum says which rhythms are present in a window but not
   when they occur. Chewing is a rhythmic envelope and a bite is a transient;
   the time series keeps both. The largest single effect in the ablation.

2. CROSS-CHANNEL ATTENTION  (squeeze-and-excitation)      (--no-channel-attention)
   The optical channel carries chewing; the accelerometers carry head and body
   motion that is sometimes signal and often artefact. A cheap learned gate
   reweights the feature channels in each window (Hu et al. 2018).

3. ATTENTION POOLING OVER TIME                            (--pooling average)
   Global average pooling dilutes a short event across the whole window.
   Attention pooling weights the time steps that carry the evidence, as in
   attention-based multiple-instance learning (Ilse et al. 2018).

4. AUXILIARY CHEW-COUNT HEAD                              (--chew-weight)
   A second head regresses the number of CHGT chew marks per window, weighted
   0.2 against the eating loss and discarded at inference. One label bit per
   window is thin supervision; counting chews asks the shared trunk to
   represent the chewing rhythm (Caruana 1997).

5. OVERLAPPING WINDOWS AT TRAINING TIME ONLY              (--no-overlap)
   A half-window hop roughly doubles the training windows. Evaluation always
   uses the non-overlapping grid - see ets_train.aligned_mask.

6. LEAVE-ONE-SUBJECT-OUT VALIDATION                       (--folds K)
   Every participant is the test set once; the threshold is chosen on other,
   inner-validation participants (Saeb et al. 2017).

EXPERIMENTS BEYOND THE DEFAULT
------------------------------
--sample-rate      128 / 64 / 32 / 16 Hz, anti-aliased (ets_data.downsample).
                   The network is scaled so every rate gives the same 64 time
                   steps and the same receptive field IN SECONDS, so the
                   comparison is about the signal, not about a network that
                   suddenly sees four times as much time per filter.
--window-seconds   2 / 4 / 8 / 16 s. Attention pooling handles any length.
--context-seconds  extra signal either side of the labelled window, with a
                   marker channel saying which part is being asked about.
                   Without the marker it cost F1 0.080: the model answered "is
                   there eating anywhere nearby?" instead.
--transformer-layers  self-attention between the 64 time steps, between the
                   channel attention and the pooling. Many more parameters
                   than the rest of the model; run it on the full dataset.
--architecture fcn / resnet
                   the published time-series baselines (Wang et al. 2017) that
                   Ghosh & Sazonov (2022) compared on AIM-2, run exactly as
                   published - single head, global average pooling - on the
                   same folds, so the comparison with prior work is like for
                   like.
--repeats N        the whole cross-validation N times at different seeds, so
                   initialisation noise is not mistaken for an effect.

NEGATIVE RESULTS KEPT VISIBLE
-----------------------------
--smoothing 3      median smoothing over 3 windows. Measured harmful (F1
                   -0.079): meals at 8 s are punctuated by pauses, and the
                   filter erases short real bouts. Off by default.

USAGE
-----
    python train_food_intake.py --config path\to\jitai_config.ini
    python train_food_intake.py --config ... --rebuild-cache      (new annotations)
    python train_food_intake.py --config ... --sweep sample-rates
    python train_food_intake.py --config ... --input fft
    python compare_runs.py results\*.json --reference results\<default>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import ets_data
import ets_eval
import ets_train

OUTPUT_DIR = Path("results")

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 80
PATIENCE = 12
DROPOUT = 0.3
CHEW_WEIGHT = 0.2

# Levels for --sweep. Each level is one run with its own results file.
SWEEPS = {
    "windows": ("window_seconds", (2.0, 4.0, 8.0, 16.0)),
    "sample-rates": ("sample_rate", (128, 64, 32, 16)),
    "chew-weights": ("chew_weight", (0.0, 0.1, 0.2, 0.5, 1.0)),
    "transformer-layers": ("transformer_layers", (0, 1, 2)),
    "context": ("context_seconds", (0.0, 4.0, 8.0)),
}


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
def mark_target_window(features: np.ndarray, context_seconds: float,
                       sample_rate: float) -> np.ndarray:
    """Append a channel that is 1 over the LABELLED window and 0 over context.

    Without it, context makes the model worse, badly: 8 s of context either
    side cost F1 0.854 -> 0.775, with precision collapsing 0.876 -> 0.683 while
    recall ROSE to 0.894. Given 24 s of signal and no indication of which 8 s it
    is being asked about, the model answers an easier question - "is there
    eating SOMEWHERE in this 24 s?" - and fires on every window next to a meal.
    One extra channel tells it where the question is.
    """
    if not context_seconds:
        return features
    total = features.shape[1]
    context = int(round(context_seconds * sample_rate))
    marker = np.zeros((1, total, 1), dtype=np.float32)
    marker[0, context:total - context, 0] = 1.0
    return np.concatenate(
        [features, np.repeat(marker, features.shape[0], axis=0)], axis=2)


def make_transform(config):
    """Raw windows -> model input, for this configuration.

    Per window and channel: mean removed and z-scored over valid samples, with
    missing samples set to 0 (ets_data.normalise). Then anti-aliased
    downsampling if a lower rate was asked for, then either the time series
    itself or its magnitude spectrum.
    """
    def transform(X: np.ndarray) -> np.ndarray:
        features = ets_data.normalise(X, high_pass=True)
        features = ets_data.downsample(features, config.sample_rate)
        if config.input == "fft":
            return ets_data.magnitude_spectrum(features)
        return mark_target_window(features, config.context_seconds,
                                  config.sample_rate)
    return transform


# --------------------------------------------------------------------------- #
# the default architecture
# --------------------------------------------------------------------------- #
def _odd(value: float) -> int:
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def build_attention_cnn(input_shape, config):
    """Three conv blocks -> channel attention -> [transformer] -> pooling -> heads.

    At 128 Hz an 8 s window is 1024 samples; two stride-2 convolutions and two
    max-pools bring it to 64 time steps of 125 ms, each seeing about 0.9 s of
    signal (one chew cycle). At lower sampling rates the network drops one
    downsampling step per halving and shortens its kernels to match, so every
    rate ends at the same 64 steps and roughly the same receptive field in
    seconds. With --input fft the same trunk slides over frequency bins.
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    factor = 1 if config.input == "fft" else int(round(
        ets_data.SENSOR_FS / config.sample_rate))
    removed = int(round(math.log2(factor)))          # 0 at 128 Hz ... 3 at 16 Hz
    pool2 = removed < 1
    pool1 = removed < 2
    stride2 = 2 if removed < 3 else 1
    long_kernel = _odd(9 / factor)
    short_kernel = _odd(5 / factor)

    inputs = layers.Input(shape=input_shape, name="signal")
    x = layers.Conv1D(32, long_kernel, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if pool1:
        x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, long_kernel, strides=stride2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if pool2:
        x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, short_kernel, padding="same")(x)
    x = layers.BatchNormalization()(x)
    features = layers.ReLU()(x)
    channels = features.shape[-1]

    # --- cross-channel attention (squeeze-and-excitation) ------------------
    if config.channel_attention:
        squeezed = layers.GlobalAveragePooling1D()(features)
        gate = layers.Dense(max(4, channels // 4), activation="relu")(squeezed)
        gate = layers.Dense(channels, activation="sigmoid")(gate)
        gate = layers.Reshape((1, channels))(gate)
        features = layers.Multiply(name="channel_attention")([features, gate])

    # --- optional self-attention between time steps -------------------------
    if config.transformer_layers:
        features = PositionalEmbedding(features.shape[1], channels)(features)
        for _ in range(config.transformer_layers):
            features = transformer_block(features, config.transformer_heads)

    # --- pooling over time --------------------------------------------------
    if config.pooling == "attention":
        # One score per time step, softmax over time, weighted sum: a two
        # second bite can dominate the window instead of being averaged away.
        scores = layers.Conv1D(1, 1)(features)
        scores = layers.Softmax(axis=1, name="temporal_attention")(scores)
        pooled = layers.Multiply()([features, scores])
        pooled = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                               output_shape=(channels,),
                               name="attention_pool")(pooled)
    else:
        pooled = layers.GlobalAveragePooling1D(name="average_pool")(features)

    shared = layers.Dense(64, activation="relu")(pooled)
    shared = layers.Dropout(DROPOUT)(shared)
    return _heads(inputs, shared, config.chew_weight, "food_intake_cnn")


def transformer_block(x, heads: int, dropout: float = 0.1):
    """Post-norm Transformer encoder block over the convolutional time steps."""
    from tensorflow.keras import layers
    width = x.shape[-1]
    attended = layers.MultiHeadAttention(
        num_heads=heads, key_dim=max(1, width // heads), dropout=dropout)(x, x)
    x = layers.LayerNormalization()(layers.Add()([x, attended]))
    forward = layers.Dense(2 * width, activation="relu")(x)
    forward = layers.Dropout(dropout)(forward)
    forward = layers.Dense(width)(forward)
    return layers.LayerNormalization()(layers.Add()([x, forward]))


def _positional_embedding_class():
    import tensorflow as tf

    class _PositionalEmbedding(tf.keras.layers.Layer):
        """A learned vector per time step, added to the features, so
        self-attention knows the ORDER of the steps as well as their content."""

        def __init__(self, steps: int, width: int, **kwargs):
            super().__init__(**kwargs)
            self.steps, self.width = steps, width

        def build(self, input_shape):
            self.positions = self.add_weight(
                name="positions", shape=(self.steps, self.width),
                initializer="random_normal", trainable=True)

        def call(self, inputs):
            return inputs + self.positions

    return _PositionalEmbedding


def PositionalEmbedding(steps: int, width: int):                   # noqa: N802
    return _positional_embedding_class()(steps, width)


# --------------------------------------------------------------------------- #
# published baselines (Wang, Yan & Oates 2017), as compared on AIM-2 by
# Ghosh & Sazonov (2022). Run as published: one head, global average pooling.
# --------------------------------------------------------------------------- #
def build_fcn(input_shape, config):
    from tensorflow.keras import layers
    inputs = layers.Input(shape=input_shape, name="signal")
    x = inputs
    for filters, kernel in ((128, 8), (256, 5), (128, 3)):
        x = layers.Conv1D(filters, kernel, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    pooled = layers.GlobalAveragePooling1D()(x)
    return _heads(inputs, pooled, 0.0, "fcn_wang2017")


def build_resnet(input_shape, config):
    from tensorflow.keras import layers
    inputs = layers.Input(shape=input_shape, name="signal")
    x = inputs
    for filters in (64, 128, 128):
        shortcut = x
        y = x
        for kernel, last in ((8, False), (5, False), (3, True)):
            y = layers.Conv1D(filters, kernel, padding="same")(y)
            y = layers.BatchNormalization()(y)
            if not last:
                y = layers.ReLU()(y)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        x = layers.ReLU()(layers.Add()([shortcut, y]))
    pooled = layers.GlobalAveragePooling1D()(x)
    return _heads(inputs, pooled, 0.0, "resnet_wang2017")


ARCHITECTURES = {"attention-cnn": build_attention_cnn, "fcn": build_fcn,
                 "resnet": build_resnet}


def _heads(inputs, shared, chew_weight: float, name: str):
    """Output heads and compilation, shared by every architecture so a
    comparison cannot accidentally differ in how the models are trained."""
    import tensorflow as tf
    from tensorflow.keras import layers

    food = layers.Dense(1, activation="sigmoid", name="food")(shared)

    if chew_weight <= 0:
        model = tf.keras.Model(inputs, food, name=name)
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss="binary_crossentropy",
                      metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
        return model

    # Chews per window, scaled in ets_train. softplus keeps it non-negative.
    chews = layers.Dense(32, activation="relu")(shared)
    chews = layers.Dense(1, activation="softplus", name="chews")(chews)

    # Outputs as a DICT: Keras 3 matches targets, losses and sample weights to
    # outputs by name only for dict outputs; a list fails with KeyError: 0.
    model = tf.keras.Model(inputs, {"food": food, "chews": chews},
                           name=f"{name}_multitask")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={"food": "binary_crossentropy", "chews": "mse"},
        loss_weights={"food": 1.0, "chews": float(chew_weight)},
        metrics={"food": ["accuracy", tf.keras.metrics.AUC(name="auc")]})
    return model


# --------------------------------------------------------------------------- #
# one configuration
# --------------------------------------------------------------------------- #
def normalise_config(config):
    """Make the configuration say what will actually run."""
    if config.architecture != "attention-cnn":
        config.chew_weight = 0.0
        config.channel_attention = False
        config.pooling = "average"
        config.transformer_layers = 0
    if config.input == "fft":
        config.context_seconds = 0.0
        config.transformer_layers = 0
    return config


def run_tag(config) -> str:
    """A file name that says exactly what ran. Unique per configuration."""
    parts = [config.tag] if config.tag else []
    if config.architecture == "attention-cnn":
        parts += [config.input,
                  "se" if config.channel_attention else "nose",
                  "attnpool" if config.pooling == "attention" else "avgpool"]
        if config.transformer_layers:
            parts.append(f"tf{config.transformer_layers}")
        parts.append(f"chew{config.chew_weight:g}")
    else:
        parts += [config.architecture, config.input]
    parts += [f"w{config.window_seconds:g}", f"{config.sample_rate:g}hz"]
    if config.context_seconds:
        parts.append(f"ctx{config.context_seconds:g}")
    if not config.overlap:
        parts.append("nooverlap")
    if config.smoothing > 1:
        parts.append(f"smooth{config.smoothing}")
    if config.ensemble > 1:
        parts.append(f"ens{config.ensemble}")
    parts.append(ets_train.cv_tag(config))
    parts.append(f"seed{config.seed}")
    if config.repeats > 1:
        parts.append(f"x{config.repeats}")
    if config.monitor != "loss":
        parts.append(f"stop{config.monitor}")
    if config.inner_participants != 2:
        parts.append(f"inner{config.inner_participants}")
    return "_".join(parts)


def describe(config) -> str:
    if config.architecture != "attention-cnn":
        return (f"{config.architecture.upper()} baseline (Wang et al. 2017), "
                f"{config.input} input")
    pieces = [f"{config.input}-domain CNN",
              "channel attention" if config.channel_attention else "no channel attention",
              f"{config.pooling} pooling"]
    if config.transformer_layers:
        pieces.append(f"{config.transformer_layers}-layer transformer")
    pieces.append(f"chew head {config.chew_weight:g}" if config.chew_weight > 0
                  else "no chew head")
    return ", ".join(pieces)


def run(config) -> Dict:
    config = normalise_config(config)
    tag = run_tag(config)
    print(f"\n{'#' * 74}\n#  {tag}\n#  {describe(config)}\n"
          f"#  {config.window_seconds:g} s windows at {config.sample_rate:g} Hz"
          f"{f', {config.context_seconds:g} s context' if config.context_seconds else ''}"
          f"\n{'#' * 74}")

    hop = config.window_seconds / 2.0 if config.overlap else config.window_seconds
    windows = ets_data.get_windows(config, config.window_seconds, hop_seconds=hop,
                                   cache_dir=config.cache_dir,
                                   rebuild=config.rebuild_cache,
                                   context_seconds=config.context_seconds)
    # A cache is built once per window/hop/context; rebuilding it again for the
    # next level of a sweep would only repeat the same database reads.
    config.rebuild_cache = False

    builder = ARCHITECTURES[config.architecture]
    auxiliary = config.chew_weight > 0
    repeats: List[Dict] = []
    participant_counts: Dict[str, Dict[str, float]] = {}
    predictions = {}

    for repeat in range(config.repeats):
        seed = config.seed + repeat
        results, pooled = ets_train.cross_validate(
            windows, make_transform(config),
            lambda shape: builder(shape, config),
            n_folds=config.folds or 0, loso=ets_train.is_loso(config),
            inner_selection=ets_train.resolve_inner_selection(config),
            inner_validation_participants=config.inner_participants,
            seed=seed, epochs=config.epochs, batch_size=BATCH_SIZE,
            patience=config.patience, auxiliary=auxiliary,
            smoothing_kernel=config.smoothing, n_models=config.ensemble,
            monitor=config.monitor, verbose=True)

        participants = ets_train.participant_records(results)
        title = f"{tag}" + (f"  (seed {seed})" if config.repeats > 1 else "")
        summary = ets_eval.report(title, [r.metrics for r in results],
                                  window_seconds=config.window_seconds,
                                  participant_metrics=participants)
        entry = {"seed": seed, "summary": summary,
                 "folds": ets_train.fold_records(results),
                 "participants": participants}
        if config.smoothing > 1:
            entry["summary_smoothed"] = ets_eval.report(
                f"{title}  + median smoothing ({config.smoothing} windows)",
                [r.smoothed_metrics for r in results],
                window_seconds=config.window_seconds)
        repeats.append(entry)

        for participant, counts in participants.items():
            total = participant_counts.setdefault(
                participant, {k: 0 for k in ("tp", "fp", "fn", "tn", "n")})
            for key in total:
                total[key] += counts[key]

        threshold = np.full(len(windows), np.nan)
        for result in results:
            threshold[result.indices] = result.threshold
        predictions[f"probability_seed{seed}"] = pooled
        predictions[f"threshold_seed{seed}"] = threshold

    # Across repeats: counts are summed per participant, so paired comparisons
    # between two runs use every seed at once.
    combined = {p: {**c, **ets_eval.metrics_from_counts(c)}
                for p, c in participant_counts.items()}
    pooled_f1 = np.array([r["summary"]["pooled_f1"] for r in repeats])
    if config.repeats > 1:
        print(f"\n  across {config.repeats} seeds: pooled F1 "
              f"{pooled_f1.mean():.4f}  (range {pooled_f1.min():.4f} - "
              f"{pooled_f1.max():.4f})")

    record = {
        "tag": tag,
        "description": describe(config),
        "config": {key: getattr(config, key) for key in CONFIG_KEYS},
        "cv": ets_train.cv_tag(config),
        "inner_selection": ets_train.resolve_inner_selection(config),
        "windows": windows.summary(),
        "n_windows_evaluated": int(repeats[0]["summary"]["n"]),
        "summary": repeats[0]["summary"],
        "pooled_f1_across_seeds": {"mean": float(pooled_f1.mean()),
                                   "min": float(pooled_f1.min()),
                                   "max": float(pooled_f1.max())},
        "participants": combined,
        "repeats": repeats,
        "annotation_shift_seconds": 0.0,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{tag}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=_json_default)
    evaluated = ets_train.aligned_mask(windows)
    np.savez_compressed(
        output_dir / f"{tag}_predictions.npz",
        participants=windows.participants, studies=windows.studies,
        starts=windows.starts, y=windows.y, chews=windows.chews,
        evaluated=evaluated, window_seconds=config.window_seconds, **predictions)
    print(f"\nwrote {path}\n      {output_dir / (tag + '_predictions.npz')}")
    return record


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


CONFIG_KEYS = ("architecture", "input", "window_seconds", "sample_rate",
               "context_seconds", "overlap", "channel_attention", "pooling",
               "transformer_layers", "transformer_heads", "chew_weight",
               "smoothing", "ensemble", "folds", "inner_participants",
               "seed", "repeats", "epochs", "patience", "monitor")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Food-intake detection from AIM-2 signals: every "
                    "experiment in the paper, one flag at a time.")
    ets_data.add_database_arguments(parser)

    g = parser.add_argument_group("data and representation")
    g.add_argument("--input", choices=("time", "fft"), default="time",
                   help="time series (default) or FFT magnitude spectrum")
    g.add_argument("--window-seconds", type=float, default=8.0)
    g.add_argument("--sample-rate", type=float, default=128,
                   choices=ets_data.SAMPLE_RATES,
                   help="Hz; anti-aliased downsampling from 128")
    g.add_argument("--context-seconds", type=float, default=0.0,
                   help="extra signal each side of the labelled window, with "
                        "a marker channel; the label is unchanged")
    g.add_argument("--no-overlap", dest="overlap", action="store_false",
                   help="train on non-overlapping windows only")

    g = parser.add_argument_group("architecture")
    g.add_argument("--architecture", choices=tuple(ARCHITECTURES),
                   default="attention-cnn",
                   help="the model (default) or a published baseline")
    g.add_argument("--no-channel-attention", dest="channel_attention",
                   action="store_false",
                   help="remove the squeeze-and-excitation gate")
    g.add_argument("--pooling", choices=("attention", "average"),
                   default="attention")
    g.add_argument("--transformer-layers", type=int, default=0,
                   help="self-attention layers between time steps (try 1 or 2)")
    g.add_argument("--transformer-heads", type=int, default=4)

    g = parser.add_argument_group("supervision and training")
    g.add_argument("--chew-weight", type=float, default=CHEW_WEIGHT,
                   help="loss weight of the chew-count head; 0 removes it")
    g.add_argument("--ensemble", type=int, default=1,
                   help="models per fold at different seeds, averaged")
    g.add_argument("--repeats", type=int, default=1,
                   help="run the whole cross-validation at N seeds")
    g.add_argument("--seed", type=int, default=9)
    g.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    g.add_argument("--patience", type=int, default=PATIENCE)
    g.add_argument("--monitor", choices=("loss", "auc"), default="loss",
                   help="what early stopping watches on the validation "
                        "participants: total loss, or ROC AUC of the eating "
                        "output (threshold-free)")
    g.add_argument("--smoothing", type=int, default=1,
                   help="median filter width in windows; 1 = off (default). "
                        "Measured harmful at 8 s")
    g.add_argument("--threads", type=int, default=0,
                   help="CPU threads for TensorFlow (0 = all). Set e.g. 4 to "
                        "run four experiments side by side")

    g = parser.add_argument_group("validation")
    ets_train.add_cv_arguments(g)

    g = parser.add_argument_group("experiments and output")
    g.add_argument("--sweep", choices=tuple(SWEEPS),
                   help="run every level of one dimension: " + "; ".join(
                       f"{name} {levels}" for name, (_, levels) in SWEEPS.items()))
    g.add_argument("--tag", default="",
                   help="prefix for the results file name")
    g.add_argument("--rebuild-cache", action="store_true",
                   help="rebuild the windows from the database (do this once "
                        "after new annotations are uploaded)")
    g.add_argument("--cache-dir", default="ets_cache")
    g.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.threads:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(
            max(1, args.threads // 2))

    if not args.sweep:
        run(copy.copy(args))
        return 0

    attribute, levels = SWEEPS[args.sweep]
    records = []
    for level in levels:
        config = copy.copy(args)
        setattr(config, attribute, level)
        records.append(run(config))
        args.rebuild_cache = False

    print(f"\n{'=' * 74}\n  SWEEP: {args.sweep}\n{'=' * 74}")
    print(f"\n  {attribute:>20}{'windows':>9}{'F1':>8}{'bal acc':>9}"
          f"{'prec':>8}{'recall':>8}{'FA/h':>7}{'F1/part':>9}")
    for level, record in zip(levels, records):
        s = record["summary"]
        print(f"  {level:>20g}{s['n']:>9,}{s['pooled_f1']:>8.3f}"
              f"{s['pooled_balanced_accuracy']:>9.3f}{s['pooled_precision']:>8.3f}"
              f"{s['pooled_recall']:>8.3f}{s.get('false_alarms_per_hour', 0):>7.1f}"
              f"{s.get('participant_f1_mean', float('nan')):>9.3f}")
    print("\n  Test the differences with compare_runs.py - a gap smaller than "
          "the\n  per-participant spread is not a result until it is paired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

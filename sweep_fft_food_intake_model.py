r"""
sweep_fft_food_intake_model_v1_timefix_shift8_threshold04.py

Hyperparameter sweep for train_fft_food_intake_model_V1_TIMEFIX_SHIFT8_THRESHOLD04.py.

DESIGN, stated up front:
  - This script does NOT modify the baseline training script at all (beyond
    the one mean_absolute_error line already added there). Your baseline's
    train_model()/FrequencyCNN don't take hyperparameters, so rather than
    editing those (which is exactly the kind of change that caused the last
    regression), this script imports the baseline module for everything that
    already works -- build_dataset, split_participants, metrics_from_probabilities,
    threshold sweeping, FFT feature extraction -- and defines its OWN small,
    separate, parameterized training loop (ParamFrequencyCNN / train_model_with_params)
    used only here.
  - The dataset is built/loaded ONCE (from cache, or the DB) and reused
    across every config in the sweep.
  - The participant split is the same single fixed split your baseline uses
    (base.split_participants with base.RANDOM_SEED) -- reused as-is, not
    replaced with k-fold or anything else not already in your pipeline.
  - Model selection during the sweep is done on VALIDATION only, at every
    threshold in base.THRESHOLD_GRID (same grid your baseline already
    prints). The TEST set is never touched until run_final_test(), called
    once, manually, after you've picked a config from the sweep summary.
  - Each config is repeated over --seeds model-init seeds (weight init /
    training-time randomness only -- the data split does not change) so you
    get mean +/- std rather than one run's numbers.
  - Results are written incrementally to CSV so a crashed/interrupted sweep
    still leaves usable partial results.

Run example:
  python sweep_fft_food_intake_model_v1_timefix_shift8_threshold04.py --config jitai_config.ini --seeds 3

After the sweep: inspect fft_food_intake_outputs_v1_timefix_shift8_threshold04/sweep_summary.csv,
pick a config + threshold (by mean val f1 or mean val balanced_accuracy),
then call run_final_test() with that config for your one honest,
held-out test-set number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import train_fft_food_intake_model_V2_TIMEFIX_SHIFT8_THRESHOLD04 as base


# =============================================================================
# SWEEP GRID -- edit this to control what gets searched
# =============================================================================

SWEEP_GRID = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [16, 32, 64],
    "dropout": [0.2, 0.3, 0.5],
    "conv_channels": [(16, 32, 64), (32, 64, 128)],
    "pos_weight_scale": [0.5, 1.0, 1.5],
}
KERNEL_SIZES_BY_DEPTH = {3: (7, 7, 5)}  # matches base.FrequencyCNN's default depth

N_SEEDS_DEFAULT = 3
MAX_EPOCHS_SWEEP = 25   # shorter than the baseline's 40 to keep sweep time reasonable
PATIENCE_SWEEP = 5

SWEEP_OUTPUT_DIR = base.OUTPUT_DIR  # same output dir as the baseline script
RESULTS_CSV = SWEEP_OUTPUT_DIR / "sweep_results.csv"
SUMMARY_CSV = SWEEP_OUTPUT_DIR / "sweep_summary.csv"
BEST_CONFIG_JSON = SWEEP_OUTPUT_DIR / "sweep_best_config.json"


# =============================================================================
# PARAMETERIZED MODEL + TRAINING LOOP (separate from the baseline script)
# =============================================================================

class ParamFrequencyCNN(nn.Module):
    """Same shape/logic as base.FrequencyCNN, but with architecture and
    dropout exposed as arguments so the sweep can vary them. Kept ONLY in
    this sweep script -- the baseline's FrequencyCNN is untouched."""
    def __init__(
        self,
        input_channels: int = 4,
        conv_channels: Sequence[int] = (16, 32, 64),
        kernel_sizes: Sequence[int] = (7, 7, 5),
        dropout: float = 0.3,
        hidden_units: int = 32,
    ):
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have the same length")

        layers = []
        in_ch = input_channels
        for i, (out_ch, k) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if i < len(conv_channels) - 1:
                layers.append(nn.MaxPool1d(2))
            else:
                layers.append(nn.AdaptiveAvgPool1d(1))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def train_model_with_params(
    X_train, y_train, X_val, y_val, device: torch.device,
    lr: float, batch_size: int, max_epochs: int, patience: int,
    conv_channels: Sequence[int], kernel_sizes: Sequence[int], dropout: float,
    pos_weight_scale: float, model_seed: int, hidden_units: int = 32,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Mirrors base.train_model()'s logic exactly (same loss, optimizer,
    early-stopping-on-val-loss rule) but with hyperparameters as arguments."""
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    train_loader = base.make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = base.make_loader(X_val, y_val, batch_size, shuffle=False)

    model = ParamFrequencyCNN(
        input_channels=X_train.shape[1], conv_channels=conv_channels,
        kernel_sizes=kernel_sizes, dropout=dropout, hidden_units=hidden_units,
    ).to(device)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    pos_weight_value = (negative_count / positive_count) * pos_weight_scale if positive_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = math.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))

        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def get_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            all_probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().tolist())
    return np.asarray(all_probs)


# =============================================================================
# SWEEP
# =============================================================================

def iter_grid(grid: Dict[str, list]):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def run_sweep(
    X_train, y_train, X_val, y_val, device: torch.device,
    n_seeds: int = N_SEEDS_DEFAULT, max_epochs: int = MAX_EPOCHS_SWEEP, patience: int = PATIENCE_SWEEP,
) -> pd.DataFrame:
    rows: List[dict] = []
    configs = list(iter_grid(SWEEP_GRID))
    total_runs = len(configs) * n_seeds
    print(f"Sweep: {len(configs)} configs x {n_seeds} seeds = {total_runs} training runs")

    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_idx = 0

    for config in configs:
        conv_channels = config["conv_channels"]
        kernel_sizes = KERNEL_SIZES_BY_DEPTH[len(conv_channels)]

        for seed_offset in range(n_seeds):
            model_seed = base.RANDOM_SEED + seed_offset
            run_idx += 1
            t0 = time.time()

            model, history = train_model_with_params(
                X_train, y_train, X_val, y_val, device=device,
                lr=config["lr"], batch_size=config["batch_size"], max_epochs=max_epochs, patience=patience,
                conv_channels=conv_channels, kernel_sizes=kernel_sizes, dropout=config["dropout"],
                pos_weight_scale=config["pos_weight_scale"], model_seed=model_seed,
            )

            val_loader = base.make_loader(X_val, y_val, config["batch_size"], shuffle=False)
            val_probs = get_probs(model, val_loader, device)
            elapsed = time.time() - t0

            # Reuse the baseline's own threshold-sweep + metrics functions --
            # same THRESHOLD_GRID your script already prints.
            sweep_df = base.threshold_sweep_from_probabilities(y_val.astype(int), val_probs, base.THRESHOLD_GRID)
            for _, row in sweep_df.iterrows():
                rows.append({
                    "run_idx": run_idx, "model_seed": model_seed,
                    "lr": config["lr"], "batch_size": config["batch_size"], "dropout": config["dropout"],
                    "conv_channels": str(conv_channels), "pos_weight_scale": config["pos_weight_scale"],
                    "epochs_run": len(history["train_loss"]), "best_val_loss": float(min(history["val_loss"])),
                    "train_seconds": round(elapsed, 1),
                    **row.to_dict(),
                })

            best_row = sweep_df.sort_values("f1", ascending=False).iloc[0]
            print(
                f"[{run_idx}/{total_runs}] lr={config['lr']} bs={config['batch_size']} "
                f"dropout={config['dropout']} conv={conv_channels} pos_w_scale={config['pos_weight_scale']} "
                f"seed={model_seed} -> best val_loss={min(history['val_loss']):.4f}, "
                f"best-threshold f1={best_row['f1']:.4f} @ th={best_row['threshold']:.2f} ({elapsed:.1f}s)"
            )

            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    return pd.DataFrame(rows)


def summarize_sweep(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Mean +/- std across model_seed for each (config, threshold) -- this is
    the table to use for the paper, since it shows sensitivity to random
    init/training noise rather than one lucky run."""
    group_cols = ["lr", "batch_size", "dropout", "conv_channels", "pos_weight_scale", "threshold"]
    agg = (
        df.groupby(group_cols)
        .agg(
            mean_f1=("f1", "mean"), std_f1=("f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"), std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_precision=("precision", "mean"), std_precision=("precision", "std"),
            mean_recall=("recall", "mean"), std_recall=("recall", "std"),
            mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
            mean_error_rate=("error_rate", "mean"), std_error_rate=("error_rate", "std"),
            mean_mae=("mean_absolute_error", "mean"), std_mae=("mean_absolute_error", "std"),
            n_seeds=("model_seed", "nunique"),
        )
        .reset_index()
        .sort_values(f"mean_{metric}", ascending=False)
    )
    return agg


def run_final_test(
    best_config: dict, X_train, y_train, X_val, y_val, X_test, y_test,
    device: torch.device, n_seeds: int = N_SEEDS_DEFAULT,
) -> pd.DataFrame:
    """Run the chosen config on the TEST set, exactly once per seed, at the
    threshold picked from the sweep summary. Call this manually, after
    you've chosen a config -- never called automatically during the sweep."""
    conv_channels = eval(best_config["conv_channels"]) if isinstance(best_config["conv_channels"], str) else tuple(best_config["conv_channels"])
    kernel_sizes = KERNEL_SIZES_BY_DEPTH[len(conv_channels)]
    threshold = float(best_config["threshold"])

    rows = []
    for seed_offset in range(n_seeds):
        model_seed = base.RANDOM_SEED + seed_offset
        model, _ = train_model_with_params(
            X_train, y_train, X_val, y_val, device=device,
            lr=best_config["lr"], batch_size=int(best_config["batch_size"]),
            max_epochs=base.MAX_EPOCHS, patience=base.PATIENCE,
            conv_channels=conv_channels, kernel_sizes=kernel_sizes, dropout=best_config["dropout"],
            pos_weight_scale=best_config["pos_weight_scale"], model_seed=model_seed,
        )
        test_loader = base.make_loader(X_test, y_test, int(best_config["batch_size"]), shuffle=False)
        test_probs = get_probs(model, test_loader, device)
        m = base.metrics_from_probabilities(y_test.astype(int), test_probs, threshold)
        m["model_seed"] = model_seed
        rows.append(m)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="jitai_config.ini")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--annotation-root", default=str(base.ANNOTATION_ROOT))
    parser.add_argument("--label-rule", default=base.LABEL_RULE, choices=base.VALID_LABEL_RULES)
    parser.add_argument("--annotation-shift-seconds", type=float, default=base.ANNOTATION_TIME_SHIFT_SECONDS)
    parser.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS_SWEEP)
    parser.add_argument("--patience", type=int, default=PATIENCE_SWEEP)
    args = parser.parse_args()

    base.ANNOTATION_TIME_SHIFT_SECONDS = float(args.annotation_shift_seconds)
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if base.CACHE_PATH.exists() and not args.rebuild_cache:
        print(f"Loading existing dataset cache: {base.CACHE_PATH.resolve()}")
        X, y, participant_per_window, records, participant_diagnostics = base.load_dataset_cache(base.CACHE_PATH)
    else:
        sessions = base.discover_annotation_sessions(Path(args.annotation_root), base.ALLOWED_PARTICIPANTS)
        config = base.load_config(args.config)
        conn = base.connect_raw_db(config)
        try:
            X, y, participant_per_window, records, participant_diagnostics = base.build_dataset(
                conn, sessions, label_rule=args.label_rule
            )
        finally:
            conn.close()
        base.save_dataset_cache(X, y, participant_per_window, records, participant_diagnostics, base.CACHE_PATH)

    train_pids, val_pids, test_pids = base.split_participants(participant_per_window, base.RANDOM_SEED)
    print(f"Train: {train_pids}\nVal:   {val_pids}\nTest:  {test_pids}  (untouched until run_final_test)")

    train_idx = base.indices_for_participants(participant_per_window, train_pids)
    val_idx = base.indices_for_participants(participant_per_window, val_pids)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_df = run_sweep(X_train, y_train, X_val, y_val, device=device, n_seeds=args.seeds, max_epochs=args.epochs, patience=args.patience)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved full sweep results: {RESULTS_CSV.resolve()}")

    summary_df = summarize_sweep(results_df, metric="f1")
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary table (mean/std across seeds, sorted by val F1): {SUMMARY_CSV.resolve()}")

    if len(summary_df) > 0:
        best_row = summary_df.iloc[0].to_dict()
        with open(BEST_CONFIG_JSON, "w", encoding="utf-8") as f:
            json.dump(best_row, f, indent=2, default=str)
        print(f"\nBest config by mean val F1:\n{json.dumps(best_row, indent=2, default=str)}")
        print(
            "\nThis is a VALIDATION-set result. Call run_final_test(best_config, ...) with "
            "this config for your one honest test-set number -- don't report a test number "
            "that was used to pick the config."
        )


if __name__ == "__main__":
    main()

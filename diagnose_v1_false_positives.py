r"""
diagnose_v1_false_positives.py

Diagnostic script for the Version 1 FFT food-intake model.

Purpose:
  This does NOT create a new model.
  It uses the already-trained V1 timefix/threshold model and helps answer:

      Why is the model creating false positives?

It saves:
  1. all_test_predictions.csv
       Every test window with true label, predicted label, probability, and error type.

  2. false_positive_windows.csv
       Only windows where true label = 0 but model predicted food.

  3. participant_error_summary.csv
       Error counts grouped by participant, so you can see if one participant is causing most FPs.

  4. probability_histogram_by_true_label.png
       Shows whether food and non-food probabilities are separated or overlapping.

  5. false_positive_plots/*.png
       Raw sensor + annotation context plots for the highest-confidence false positives.

How to use:
  Put this script in the same folder as:
      train_fft_food_intake_model_V1_TIMEFIX_THRESHOLD.py

  First, make sure you already ran:
      python train_fft_food_intake_model_V1_TIMEFIX_THRESHOLD.py --rebuild-cache

  Then run:
      python diagnose_v1_false_positives.py

Optional:
      python diagnose_v1_false_positives.py --top-n 30
      python diagnose_v1_false_positives.py --threshold 0.40
      python diagnose_v1_false_positives.py --split test

Notes:
  - This script uses the same participant split and model architecture as V1.
  - It reconnects to the database only to generate raw signal plots for false positives.
  - It does not train or change the model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Import the V1 model/script as a module.
# This diagnostic script should be placed in the same folder as the V1 script.
try:
    import train_fft_food_intake_model_V1_TIMEFIX_THRESHOLD as base
except ImportError as exc:
    raise ImportError(
        "Could not import train_fft_food_intake_model_V1_TIMEFIX_THRESHOLD.py.\n"
        "Put this diagnostic script in the same folder as that V1 script, then run again."
    ) from exc


DIAG_OUTPUT_DIR = Path("fft_food_intake_outputs_v1_diagnostics")
FALSE_POSITIVE_PLOT_DIR = DIAG_OUTPUT_DIR / "false_positive_plots"

CHANNEL_NAMES = ["Acc X", "Acc Y", "Acc Z", "Optical"]


def parse_record_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value))


def load_trained_v1_model(device: torch.device):
    if not base.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained model: {base.MODEL_PATH}\n"
            "Run train_fft_food_intake_model_V1_TIMEFIX_THRESHOLD.py first."
        )

    checkpoint = torch.load(base.MODEL_PATH, map_location=device)
    model = base.FrequencyCNN(input_channels=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def get_selected_indices(participants: Sequence[str], split_name: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    train_pids, val_pids, test_pids = base.split_participants(participants, base.RANDOM_SEED)

    if split_name == "train":
        selected = train_pids
    elif split_name == "val":
        selected = val_pids
    elif split_name == "test":
        selected = test_pids
    elif split_name == "all":
        selected = sorted(set(participants))
    else:
        raise ValueError(f"Unknown split: {split_name}")

    idx = base.indices_for_participants(participants, selected)
    return idx, train_pids, val_pids, test_pids


def predict_probabilities(model, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    probs: List[float] = []
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(batch_probs.tolist())
    return np.asarray(probs, dtype=float)


def build_results_dataframe(
    records: List[dict],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    y_pred = (y_prob >= threshold).astype(int)

    for i, rec in enumerate(records):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        prob = float(y_prob[i])

        if true_label == 0 and pred_label == 0:
            error_type = "TN"
        elif true_label == 0 and pred_label == 1:
            error_type = "FP"
        elif true_label == 1 and pred_label == 0:
            error_type = "FN"
        else:
            error_type = "TP"

        rows.append(
            {
                "row_index_in_split": i,
                "participant_id": rec["participant_id"],
                "session_id": rec.get("session_id", "main"),
                "start_time": rec["start_time"],
                "end_time": rec["end_time"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "probability_food": prob,
                "threshold": threshold,
                "error_type": error_type,
            }
        )

    df = pd.DataFrame(rows)
    return add_nearest_food_distance(df)


def add_nearest_food_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance in seconds to nearest true food window within same participant/session."""
    out = df.copy()
    out["start_dt"] = pd.to_datetime(out["start_time"])
    out["end_dt"] = pd.to_datetime(out["end_time"])
    out["seconds_to_nearest_true_food"] = np.nan

    for (pid, sid), group in out.groupby(["participant_id", "session_id"]):
        food_times = group.loc[group["true_label"] == 1, "start_dt"].sort_values().to_numpy()
        if len(food_times) == 0:
            continue

        group_indices = group.index.tolist()
        for idx in group_indices:
            t = out.at[idx, "start_dt"]
            # Convert numpy datetime64 distances to seconds.
            diffs = np.abs((food_times - np.datetime64(t)).astype("timedelta64[ms]").astype(float) / 1000.0)
            out.at[idx, "seconds_to_nearest_true_food"] = float(np.min(diffs))

    out["near_food_within_8s"] = out["seconds_to_nearest_true_food"] <= 8
    out["near_food_within_16s"] = out["seconds_to_nearest_true_food"] <= 16
    out["near_food_within_32s"] = out["seconds_to_nearest_true_food"] <= 32
    out = out.drop(columns=["start_dt", "end_dt"])
    return out


def summarize_by_participant(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in df.groupby("participant_id"):
        tn = int((group["error_type"] == "TN").sum())
        fp = int((group["error_type"] == "FP").sum())
        fn = int((group["error_type"] == "FN").sum())
        tp = int((group["error_type"] == "TP").sum())
        total = len(group)
        positives = int((group["true_label"] == 1).sum())
        negatives = int((group["true_label"] == 0).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        rows.append(
            {
                "participant_id": pid,
                "total_windows": total,
                "true_food_windows": positives,
                "true_nonfood_windows": negatives,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "recall_food_detection": recall,
                "specificity_nonfood_rejection": specificity,
                "precision_when_predicting_food": precision,
                "false_positive_rate": fp_rate,
                "mean_probability_food": float(group["probability_food"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["FP", "false_positive_rate"], ascending=[False, False])


def plot_probability_histogram(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    nonfood = df.loc[df["true_label"] == 0, "probability_food"].to_numpy()
    food = df.loc[df["true_label"] == 1, "probability_food"].to_numpy()
    plt.hist(nonfood, bins=20, alpha=0.55, label="True non-food")
    plt.hist(food, bins=20, alpha=0.55, label="True food")
    plt.axvline(float(df["threshold"].iloc[0]), linestyle="--", label="Decision threshold")
    plt.xlabel("Model probability of food intake")
    plt.ylabel("Number of windows")
    plt.title("Prediction Probabilities by True Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_sensor_array_any_length(packets: Sequence[base.PacketRecord], window_start: datetime, window_end: datetime) -> np.ndarray:
    total_samples = int(round((window_end - window_start).total_seconds() * base.SENSOR_SAMPLE_RATE_HZ))
    sensor_window = np.full((total_samples, 4), -1.0, dtype=np.float32)

    for packet in packets:
        packet_arr = base.load_packet_array(packet)
        packet_start_dt, _ = base.get_packet_start_end(packet)
        packet_start_idx = int(round((packet_start_dt - window_start).total_seconds() * base.SENSOR_SAMPLE_RATE_HZ))
        packet_end_idx = packet_start_idx + packet_arr.shape[0]

        dst_start = max(packet_start_idx, 0)
        dst_end = min(packet_end_idx, sensor_window.shape[0])
        if dst_start >= dst_end:
            continue

        src_start = dst_start - packet_start_idx
        src_end = src_start + (dst_end - dst_start)
        sensor_window[dst_start:dst_end, :] = packet_arr[src_start:src_end, :]

    return sensor_window


def get_session_map(annotation_root: Path) -> Dict[str, base.AnnotationSession]:
    sessions = base.discover_annotation_sessions(annotation_root, base.ALLOWED_PARTICIPANTS)
    return {s.participant_id: s for s in sessions}


def make_false_positive_plot(
    conn,
    session_map: Dict[str, base.AnnotationSession],
    row: pd.Series,
    output_path: Path,
    context_seconds: int = 32,
) -> None:
    pid = str(row["participant_id"])
    if pid not in session_map:
        print(f"  Skipping plot for {pid}; no annotation session found.")
        return

    center_start = parse_record_time(row["start_time"])
    center_end = parse_record_time(row["end_time"])
    context_start = center_start - timedelta(seconds=context_seconds)
    context_end = center_end + timedelta(seconds=context_seconds)

    packets = base.get_sensor_packets(conn, pid)
    sensor_context = build_sensor_array_any_length(packets, context_start, context_end)

    session = session_map[pid]
    # Use sensor start date as base date, matching the training script.
    sensor_start, _ = base.get_sensor_span(packets)
    annotation_data = base.load_annotation_csv(session.csv_path, base_date=sensor_start.date())

    sensor_t = np.arange(sensor_context.shape[0]) / base.SENSOR_SAMPLE_RATE_HZ

    ann_times = annotation_data["times"]
    mask = (ann_times >= context_start) & (ann_times <= context_end)
    ann_sec = np.asarray([(t - context_start).total_seconds() for t in ann_times[mask]])

    missing_fraction = float(np.mean(sensor_context == -1.0)) if sensor_context.size else np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"False positive diagnostic: {pid}\n"
        f"Window {center_start} to {center_end} | "
        f"P(food)={float(row['probability_food']):.3f} | "
        f"nearest true food={float(row['seconds_to_nearest_true_food']):.1f}s | "
        f"context missing={missing_fraction:.3f}"
    )

    for ch, ax in enumerate(axes[:4]):
        ax.plot(sensor_t, sensor_context[:, ch], linewidth=0.8)
        ax.axvspan((center_start - context_start).total_seconds(), (center_end - context_start).total_seconds(), alpha=0.20)
        ax.set_ylabel(CHANNEL_NAMES[ch])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax = axes[4]
    if mask.sum() > 0:
        food = annotation_data["food"][mask].to_numpy(dtype=float)
        chew_count = annotation_data["chew_count"][mask].to_numpy(dtype=float)
        chew_bout = annotation_data["chew_bout"][mask].to_numpy(dtype=float)
        bite = annotation_data["bite"][mask].to_numpy(dtype=float)
        ax.step(ann_sec, food, where="post", label="Food fallback", linewidth=1.2)
        ax.step(ann_sec, chew_count, where="post", label="Chew Count", linewidth=0.8)
        ax.step(ann_sec, chew_bout, where="post", label="Chew Bout", linewidth=0.8)
        ax.step(ann_sec, bite, where="post", label="Bite", linewidth=0.8)
    ax.axvspan((center_start - context_start).total_seconds(), (center_end - context_start).total_seconds(), alpha=0.20)
    ax.set_ylabel("Annotation")
    ax.set_xlabel(f"Seconds from {context_start.strftime('%H:%M:%S')}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.2, 1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="jitai_config.ini", help="Path to jitai_config.ini")
    parser.add_argument("--annotation-root", default=str(base.ANNOTATION_ROOT), help="Root folder containing AIM participant annotation folders")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to analyze")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold. If omitted, use threshold saved in V1 model checkpoint.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of highest-confidence false positives to plot")
    parser.add_argument("--context-seconds", type=int, default=32, help="Seconds before/after the false-positive window to show in plots")
    args = parser.parse_args()

    DIAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FALSE_POSITIVE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not base.CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset cache: {base.CACHE_PATH}\n"
            "Run train_fft_food_intake_model_V1_TIMEFIX_THRESHOLD.py --rebuild-cache first."
        )

    X, y, participant_per_window, records = base.load_dataset_cache(base.CACHE_PATH)
    selected_idx, train_pids, val_pids, test_pids = get_selected_indices(participant_per_window, args.split)

    X_selected = X[selected_idx]
    y_selected = y[selected_idx].astype(int)
    records_selected = [records[i] for i in selected_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_v1_model(device)

    threshold = args.threshold
    if threshold is None:
        threshold = float(checkpoint.get("selected_threshold", 0.5))

    print("\nDiagnostic configuration")
    print(f"  Split analyzed: {args.split}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Train participants: {train_pids}")
    print(f"  Val participants:   {val_pids}")
    print(f"  Test participants:  {test_pids}")
    print(f"  Windows analyzed: {len(y_selected)}")
    print(f"  True positives in split: {int(y_selected.sum())}")
    print(f"  True negatives in split: {int(len(y_selected) - y_selected.sum())}")

    y_prob = predict_probabilities(model, X_selected, device)
    results_df = build_results_dataframe(records_selected, y_selected, y_prob, threshold)

    all_pred_path = DIAG_OUTPUT_DIR / f"all_{args.split}_predictions.csv"
    fp_path = DIAG_OUTPUT_DIR / f"false_positive_windows_{args.split}.csv"
    summary_path = DIAG_OUTPUT_DIR / f"participant_error_summary_{args.split}.csv"
    hist_path = DIAG_OUTPUT_DIR / f"probability_histogram_by_true_label_{args.split}.png"

    false_positives = results_df[results_df["error_type"] == "FP"].copy()
    false_positives = false_positives.sort_values("probability_food", ascending=False)

    participant_summary = summarize_by_participant(results_df)

    results_df.to_csv(all_pred_path, index=False)
    false_positives.to_csv(fp_path, index=False)
    participant_summary.to_csv(summary_path, index=False)
    plot_probability_histogram(results_df, hist_path)

    print("\nOverall error counts")
    print(results_df["error_type"].value_counts().to_string())

    print("\nParticipant error summary")
    print(participant_summary.to_string(index=False))

    print("\nFalse-positive proximity to real food windows")
    if len(false_positives) > 0:
        print(f"  False positives total: {len(false_positives)}")
        print(f"  Within 8 sec of true food:  {int(false_positives['near_food_within_8s'].sum())}")
        print(f"  Within 16 sec of true food: {int(false_positives['near_food_within_16s'].sum())}")
        print(f"  Within 32 sec of true food: {int(false_positives['near_food_within_32s'].sum())}")
        print("\nTop false positives by confidence")
        print(false_positives.head(10)[[
            "participant_id", "start_time", "end_time", "probability_food",
            "seconds_to_nearest_true_food", "near_food_within_16s"
        ]].to_string(index=False))
    else:
        print("  No false positives found at this threshold.")

    print("\nSaved diagnostic files")
    print(f"  {all_pred_path.resolve()}")
    print(f"  {fp_path.resolve()}")
    print(f"  {summary_path.resolve()}")
    print(f"  {hist_path.resolve()}")

    # Generate raw signal plots for the highest-confidence false positives.
    if len(false_positives) > 0 and args.top_n > 0:
        print(f"\nGenerating plots for top {min(args.top_n, len(false_positives))} false positives...")
        annotation_root = Path(args.annotation_root)
        session_map = get_session_map(annotation_root)
        config = base.load_config(args.config)
        conn = base.connect_raw_db(config)
        try:
            for rank, (_, row) in enumerate(false_positives.head(args.top_n).iterrows(), start=1):
                safe_start = str(row["start_time"]).replace(":", "-").replace(" ", "_")
                out_path = FALSE_POSITIVE_PLOT_DIR / f"fp_{rank:03d}_{row['participant_id']}_{safe_start}.png"
                try:
                    make_false_positive_plot(
                        conn,
                        session_map,
                        row,
                        out_path,
                        context_seconds=args.context_seconds,
                    )
                    print(f"  Saved {out_path}")
                except Exception as exc:
                    print(f"  Could not plot FP rank {rank} ({row['participant_id']} {row['start_time']}): {exc}")
        finally:
            conn.close()

        print(f"\nFalse-positive plots saved in: {FALSE_POSITIVE_PLOT_DIR.resolve()}")

    print("\nHow to interpret the outputs:")
    print("  - If most FPs are within 16-32 sec of true food, they may be edge/timing windows around eating.")
    print("  - If one participant has most FPs, investigate that participant's sensor/annotation quality.")
    print("  - If high-confidence FPs show large motion spikes, the model may confuse general wrist movement with eating.")
    print("  - If probability histograms overlap heavily, the FFT features may not separate food/non-food well enough.")


if __name__ == "__main__":
    main()

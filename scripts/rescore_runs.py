r"""
rescore_runs.py - re-score finished runs on the corrected evaluation grid.

Until 20/10/26 the windows a run was scored on came from a grid anchored once
per PARTICIPANT with a relative tolerance (see ets_train.evaluation_grid).
For participants with two sessions, the second session was scored on all of
its half-overlapping windows, on none of them, or correctly, depending only
on how far apart the two recordings were. Training, early stopping and the
thresholds were not affected; only which held-out windows were counted.

Every run saved its held-out probabilities, so no retraining is needed for
windows that were scored: this keeps only the windows on the corrected grid
and recomputes every number. Sessions that were scored on NO window were
never predicted; they are listed, and only a rerun can add them.

    python rescore_runs.py results\*.json

Writes corrected copies, same file names, to results_fixed\ (the originals are
left alone). Point every other script at results_fixed\ afterwards:

    python compare_runs.py results_fixed\*x3*.json --reference results_fixed\<run>.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import ets_eval
import ets_train


def counts(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"tp": int(np.sum((y == 1) & (pred == 1))),
            "fp": int(np.sum((y == 0) & (pred == 1))),
            "fn": int(np.sum((y == 1) & (pred == 0))),
            "tn": int(np.sum((y == 0) & (pred == 0)))}


def summarise(y, pred, participants, window: float, old: dict) -> tuple:
    total = counts(y, pred)
    summary = dict(old)
    summary.update(total)
    summary["n"] = int(len(y))
    for key, value in ets_eval.metrics_from_counts(total).items():
        summary[f"pooled_{key}"] = value
    summary["false_alarms_per_hour"] = ets_eval.false_alarms_per_hour(
        total["fp"], total["tn"], window)
    per = {}
    for p in np.unique(participants):
        c = counts(y[participants == p], pred[participants == p])
        c["n"] = int(sum(c.values()))
        per[str(p)] = {**c, **ets_eval.metrics_from_counts(c)}
    f1 = np.array([m["f1"] for m in per.values()])
    summary.update(participant_f1_mean=float(f1.mean()),
                   participant_f1_std=float(f1.std(ddof=0)),
                   participant_f1_median=float(np.median(f1)),
                   participants=int(len(f1)), rescored_on_session_grid=True)
    return summary, per


def rescore(path: Path, out_dir: Path) -> None:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    npz = path.with_name(f"{record['tag']}_predictions.npz")
    if not npz.is_file():
        print(f"  skipped {path.name}: no predictions file")
        return
    data = dict(np.load(npz, allow_pickle=False))
    window = float(data["window_seconds"])
    overlap = record["config"].get("overlap", True)
    hop = window / 2.0 if overlap else window
    grid = ets_train.evaluation_grid(data["participants"], data["studies"],
                                     data["starts"], window, hop)
    old_mask = data["evaluated"].astype(bool)

    keys = np.char.add(np.char.add(data["participants"].astype(str), "|"),
                       data["studies"].astype(str))
    seeds = [r["seed"] for r in record["repeats"]]
    scored = np.isfinite(data[f"probability_seed{seeds[0]}"])
    missing = [k for k in np.unique(keys) if not np.any(scored[keys == k])]
    doubled = [k for k in np.unique(keys)
               if np.sum(old_mask[keys == k]) > np.sum(grid[keys == k]) + 1]

    combined = {}
    for index, seed in enumerate(seeds):
        probability = data[f"probability_seed{seed}"]
        threshold = data[f"threshold_seed{seed}"]
        keep = grid & np.isfinite(probability) & np.isfinite(threshold)
        y = data["y"][keep].astype(int)
        pred = (probability[keep] >= threshold[keep]).astype(int)
        summary, per = summarise(y, pred, data["participants"][keep], window,
                                 record["repeats"][index]["summary"])
        record["repeats"][index]["summary"] = summary
        record["repeats"][index]["participants"] = per
        for p, c in per.items():
            total = combined.setdefault(p, {k: 0 for k in ("tp", "fp", "fn", "tn", "n")})
            for k in total:
                total[k] += c[k]
        if index == 0:
            data["evaluated"] = keep
            record["summary"] = summary
            record["n_windows_evaluated"] = summary["n"]

    record["participants"] = {p: {**c, **ets_eval.metrics_from_counts(c)}
                              for p, c in combined.items()}
    pooled = np.array([r["summary"]["pooled_f1"] for r in record["repeats"]])
    record["pooled_f1_across_seeds"] = {"mean": float(pooled.mean()),
                                        "min": float(pooled.min()),
                                        "max": float(pooled.max())}
    record["rescore_note"] = (f"session grid; {len(doubled)} session(s) had been "
                              f"scored on overlapping windows, {len(missing)} "
                              f"never scored")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / path.name, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    np.savez_compressed(out_dir / npz.name, **data)
    s = record["summary"]
    print(f"  {record['tag']}\n      windows {int(old_mask.sum()):,} -> {s['n']:,}   "
          f"F1 {s['pooled_f1']:.3f}   acc {s['pooled_accuracy']:.3f}   "
          f"FA/h {s['false_alarms_per_hour']:.1f}   "
          f"(overlap-scored sessions: {len(doubled)}, never scored: {len(missing)})")
    if missing:
        print("      never scored: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out-dir", default="results_fixed")
    args = parser.parse_args()
    paths = sorted({Path(p) for pattern in args.runs
                    for p in (glob.glob(pattern) or [pattern])
                    if p.endswith(".json") and Path(p).is_file()})
    print(f"re-scoring {len(paths)} run(s) into {args.out_dir}\\")
    for path in paths:
        try:
            rescore(path, Path(args.out_dir))
        except (KeyError, ValueError) as exc:
            print(f"  skipped {path.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""
plot_results.py - presentation figures from train_food_intake.py results.

Grayscale only, Arial 11 pt (Liberation Sans or DejaVu Sans if Arial is not
installed), so every figure matches the document and the slides.

    python plot_results.py results\<run>.json
        per-participant F1, and a timeline of predicted probability against
        the annotation for the best, median and worst participant

    python plot_results.py results\*.json --compare
        pooled F1 of every run with its 95% interval against the reference
        (run compare_runs.py first; it writes the interval)

Figures are written next to the results as PNG (300 dpi) and PDF.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

INK = "#222222"
MID = "#777777"
LIGHT = "#d0d0d0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.edgecolor": MID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(figure, path: Path) -> None:
    # Suffixes appended, not replaced: run tags contain dots ("chew0.2"),
    # which Path.with_suffix would read as an extension and cut off.
    figure.tight_layout()
    figure.savefig(f"{path}.png", dpi=300)
    figure.savefig(f"{path}.pdf")
    plt.close(figure)
    print(f"wrote {path}.png")


def participant_figure(record: dict, out: Path) -> None:
    items = sorted(record["participants"].items(), key=lambda kv: kv[1]["f1"])
    names = [p for p, _ in items]
    f1 = np.array([m["f1"] for _, m in items])
    pooled = record["summary"]["pooled_f1"]

    figure, axis = plt.subplots(figsize=(max(6.5, 0.22 * len(names) + 2), 3.6))
    axis.bar(range(len(names)), f1, color=MID, width=0.75)
    axis.axhline(pooled, color=INK, linewidth=1, linestyle="--")
    axis.text(len(names) - 0.5, pooled + 0.015, f"pooled F1 {pooled:.3f}",
              ha="right", va="bottom")
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=90)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1")
    axis.set_xlabel("Held-out participant")
    save(figure, out)


def timeline_figure(record: dict, predictions: Path, out: Path) -> None:
    data = np.load(predictions, allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    probability = data[f"probability_seed{seed}"]
    threshold = data[f"threshold_seed{seed}"]
    evaluated = data["evaluated"]
    window = float(data["window_seconds"])

    ranked = sorted(record["participants"].items(), key=lambda kv: kv[1]["f1"])
    picks = [ranked[-1], ranked[len(ranked) // 2], ranked[0]]
    labels = ["best", "median", "worst"]

    figure, axes = plt.subplots(len(picks), 1, figsize=(8, 2.1 * len(picks)))
    for axis, (participant, scores), label in zip(np.atleast_1d(axes), picks, labels):
        rows = np.flatnonzero((data["participants"] == participant) & evaluated)
        rows = rows[np.argsort(data["starts"][rows])]
        minutes = (data["starts"][rows] - data["starts"][rows][0]) / 60.0
        truth = data["y"][rows]
        for start, eating in zip(minutes, truth):
            if eating:
                axis.axvspan(start, start + window / 60.0, color=LIGHT, linewidth=0)
        axis.step(minutes, probability[rows], where="post", color=INK, linewidth=1)
        axis.axhline(threshold[rows][0], color=MID, linewidth=1, linestyle="--")
        axis.set_ylim(0, 1)
        axis.set_ylabel("p(eating)")
        axis.set_title(f"{participant} ({label}, F1 {scores['f1']:.3f}). "
                       f"Shaded: annotated eating. Dashed: threshold.",
                       loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("Minutes from start of annotation")
    save(figure, out)


def compare_figure(csv_path: Path, out: Path) -> None:
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: float(r["f1"]))
    figure, axis = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    for index, row in enumerate(rows):
        f1 = float(row["f1"])
        reference = row["reference"] == "yes"
        axis.barh(index, f1, color=INK if reference else MID, height=0.65)
        if row.get("ci_low"):
            # The interval is for the CHANGE against the reference; drawn
            # around this run's F1 it shows the range that change could span.
            reference_f1 = f1 - float(row["f1_change"])
            low = reference_f1 + float(row["ci_low"])
            high = reference_f1 + float(row["ci_high"])
            axis.plot([low, high], [index, index], color=INK, linewidth=1.2)
        axis.text(f1 + 0.005, index, f"{f1:.3f}", va="center")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([r["run"] for r in rows])
    axis.set_xlim(0, 1.08)
    axis.set_xlabel("Pooled F1 (line: 95% interval of the change against the "
                    "reference, in black)")
    save(figure, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--compare", action="store_true",
                        help="one chart of every run, from compare_runs.py's CSV")
    parser.add_argument("--csv", default="results_summary.csv")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        if not Path(args.csv).is_file():
            raise SystemExit(f"{args.csv} not found - run compare_runs.py first")
        compare_figure(Path(args.csv), out_dir / "compare_runs")
        return 0

    paths = sorted({Path(p) for pattern in args.runs
                    for p in (glob.glob(pattern) or [pattern])})
    for path in paths:
        if path.suffix != ".json" or path.name.endswith("_predictions.json"):
            continue
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        participant_figure(record, out_dir / f"{record['tag']}_participants")
        predictions = path.with_name(f"{record['tag']}_predictions.npz")
        if predictions.is_file():
            timeline_figure(record, predictions, out_dir / f"{record['tag']}_timeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

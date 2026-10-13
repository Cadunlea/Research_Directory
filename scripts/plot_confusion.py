r"""
plot_confusion.py - confusion matrices of finished runs, side by side.

    python plot_confusion.py results\<run>.json
    python plot_confusion.py results\<default run>.json results\<fa15 run>.json --labels "Default threshold" "False-alarm budget 15/h"

Each panel: rows are what the annotation says, columns what the model said.
Every cell shows its window count and its share of that row, so the
false-positive cell reads directly as "share of non-eating time wrongly
flagged". Shading follows the row share (one gray ramp, light to dark).
Under each panel: precision, recall and false alarms per hour of non-eating.

Pooled over every held-out window of the run (its first seed), the same
numbers compare_runs.py prints. Grayscale, Arial 11 pt, like plot_results.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

import ets_eval                                                      # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 11})
INK = "#222222"


def shade(share: float) -> str:
    """Light gray for a small row share, dark gray for a large one."""
    level = int(round(242 - share * (242 - 90)))
    return f"#{level:02x}{level:02x}{level:02x}"


def panel(axis, summary: dict, window_seconds: float, label: str) -> None:
    tp, fp, fn, tn = (int(summary[k]) for k in ("tp", "fp", "fn", "tn"))
    rows = [("Eating", (tp, fn)), ("Not eating", (fp, tn))]
    for r, (_, counts) in enumerate(rows):
        total = max(sum(counts), 1)
        for c, count in enumerate(counts):
            share = count / total
            axis.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=shade(share),
                                         edgecolor="white", linewidth=2))
            colour = "white" if share > 0.55 else INK
            axis.text(c + 0.5, 1.5 - r + 0.08, f"{count:,}", ha="center",
                      va="center", fontsize=16, fontweight="bold", color=colour)
            axis.text(c + 0.5, 1.5 - r - 0.18, f"{share:.1%} of row", ha="center",
                      va="center", color=colour)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_xticks([0.5, 1.5], ["Eating", "Not eating"])
    axis.set_yticks([1.5, 0.5], [r[0] for r in rows])
    axis.xaxis.set_ticks_position("top")
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Model said")
    axis.set_ylabel("Annotation says")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal")

    metrics = ets_eval.metrics_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    alarms = ets_eval.false_alarms_per_hour(fp, tn, window_seconds)
    axis.set_title(label, pad=34, fontweight="bold")
    axis.text(1, -0.12, f"Precision {metrics['precision']:.3f}   Recall "
              f"{metrics['recall']:.3f}\n{alarms:.1f} false alarms per hour of "
              f"non-eating", ha="center", va="top", transform=axis.transData)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+", help="result JSON files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="one panel title per run (default: the run's tag)")
    parser.add_argument("--out", default="figures/confusion_matrix")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        raise SystemExit("give one --labels entry per run")

    records = []
    for path in args.runs:
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))

    figure, axes = plt.subplots(1, len(records), figsize=(4.2 * len(records), 4.6),
                                squeeze=False)
    for index, (axis, record) in enumerate(zip(axes[0], records)):
        label = args.labels[index] if args.labels else record["tag"]
        panel(axis, record["summary"], float(record["config"]["window_seconds"]), label)
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

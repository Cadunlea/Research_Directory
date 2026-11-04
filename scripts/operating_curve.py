r"""
operating_curve.py - the false-alarm / recall trade-off of finished runs.

Every run saves its held-out probabilities (results\<run>_predictions.npz).
This sweeps the decision threshold over them and reports, for each false-alarm
budget, how much eating the model still catches. No retraining; seconds.

    python operating_curve.py results\<run>.json
    python operating_curve.py results\<run>.json results\<other run>.json

With several runs, their curves go on one figure: at the same false-alarm
rate, which model catches more eating?

READ THIS BEFORE QUOTING A NUMBER. The curve is descriptive: it applies one
threshold to held-out predictions pooled across folds. Picking a point off it
means choosing the threshold on the test participants, which is optimistic.
For a number to report, train with
    --threshold-objective false-alarms --max-false-alarms N
which chooses the threshold on the validation participants instead.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import ets_eval

BUDGETS = (40, 30, 20, 15, 10, 5)
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def curve(path: Path):
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"),
                   allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    keep = data["evaluated"]
    y = data["y"][keep].astype(int)
    probability = data[f"probability_seed{seed}"][keep]
    window = float(data["window_seconds"])

    rows = []
    for threshold in GRID:
        m = ets_eval.metrics(y, probability, threshold)
        m["false_alarms_per_hour"] = ets_eval.false_alarms_per_hour(
            m["fp"], m["tn"], window)
        rows.append(m)
    return record, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()

    paths = sorted({Path(p) for pattern in args.runs
                    for p in (glob.glob(pattern) or [pattern])
                    if p.endswith(".json")})
    curves = []
    for path in paths:
        record, rows = curve(path)
        curves.append((record, rows))
        s = record["summary"]
        print(f"\n{record['tag']}")
        print(f"  as run (threshold chosen on validation participants): "
              f"{s.get('false_alarms_per_hour', float('nan')):.1f} false alarms/h, "
              f"recall {s['pooled_recall']:.3f}, precision {s['pooled_precision']:.3f}, "
              f"F1 {s['pooled_f1']:.3f}, accuracy {s['pooled_accuracy']:.3f}")
        print(f"\n  {'budget (FA/h)':>14}{'threshold':>11}{'FA/h':>7}{'recall':>8}"
              f"{'precision':>11}{'F1':>7}{'accuracy':>10}")
        for budget in BUDGETS:
            within = [r for r in rows if r["false_alarms_per_hour"] <= budget]
            if not within:
                print(f"  {budget:>14}   not reachable on this grid")
                continue
            best = max(within, key=lambda r: r["recall"])
            print(f"  {budget:>14}{best['threshold']:>11.2f}"
                  f"{best['false_alarms_per_hour']:>7.1f}{best['recall']:>8.3f}"
                  f"{best['precision']:>11.3f}{best['f1']:>7.3f}{best['accuracy']:>10.3f}")

    print("\n  Descriptive only: these thresholds are read off the test predictions."
          "\n  For a reportable number, train with --threshold-objective false-alarms.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    shades = ["#222222", "#888888", "#bbbbbb", "#555555"]
    styles = ["-", "--", ":", "-."]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for index, (record, rows) in enumerate(curves):
        colour, style = shades[index % 4], styles[index % 4]
        fa = [r["false_alarms_per_hour"] for r in rows]
        recall = [r["recall"] for r in rows]
        axis.plot(fa, recall, color=colour, linestyle=style, linewidth=1.6,
                  label=record.get("description", record["tag"]))
        s = record["summary"]
        axis.plot(s.get("false_alarms_per_hour", np.nan), s["pooled_recall"], "o",
                  color=colour, markersize=6)
    axis.set_xlabel("False alarms per hour of non-eating")
    axis.set_ylabel("Share of eating windows detected (recall)")
    axis.set_xlim(0, max(90, axis.get_xlim()[1] if curves else 90))
    axis.set_ylim(0, 1.0)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Dots: the operating point each run chose on validation participants",
                   loc="left", fontsize=11)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out / "operating_curve.png", dpi=300)
    figure.savefig(out / "operating_curve.pdf")
    print(f"\nwrote {out / 'operating_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

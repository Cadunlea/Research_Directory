r"""
plot_meals.py - eating time, chews and chewing rate per meal, from process_database.py.

Reads <results>\sessions.csv (default database_results\) and draws two
figures, grayscale, Arial 11:

    meals.png / .pdf        one bar per recording (each ETS session is one
                            meal), grouped by meal type: minutes of eating
                            detected, chews counted by the model, and chewing
                            rate (chews per minute of eating). Where the
                            session is annotated, the annotated value is
                            drawn over the bar as an open circle.
    chew_check.png / .pdf   the model's chews and chewing rate per meal
                            against the annotated ones, for annotated
                            sessions, both over the annotated stretch (the
                            model's from the windows it called eating). Filled: participants the model never
                            trained on; open: participants it trained on
                            (optimistic). Printed: correlation and median
                            error for each group.

Check chew_check before quoting chew numbers: the chew output was an
auxiliary training target, never validated on its own.

USAGE
-----
    python plot_meals.py
    python plot_meals.py --results database_results --out figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

INK, MID, LIGHT = "#222222", "#777777", "#cccccc"


def number(row, key):
    try:
        return float(row.get(key) or "nan")
    except ValueError:
        return float("nan")


def meal_figure(rows, out: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["study"], r["participant"]))
    studies = [r["study"] for r in rows]
    x = np.arange(len(rows))
    panels = (("eating_min", "annotated_eating_min", "Eating (min)"),
              ("chews", "annotated_chews", "Chews"),
              ("chews_per_min", "annotated_chews_per_min", "Chewing rate\n(chews/min)"))
    figure, axes = plt.subplots(3, 1, figsize=(max(8, 0.22 * len(rows) + 2), 8), sharex=True)
    for axis, (key, annotated_key, label) in zip(axes, panels):
        axis.bar(x, [number(r, key) for r in rows], color=MID, width=0.8)
        truth = np.array([number(r, annotated_key) for r in rows])
        has = np.isfinite(truth)
        if has.any():
            axis.plot(x[has], truth[has], "o", markerfacecolor="white",
                      markeredgecolor=INK, markersize=5, label="annotated")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False, loc="upper right")
    # Meal-type groups: a divider and a label under each group.
    edges = [0] + [i for i in range(1, len(rows)) if studies[i] != studies[i - 1]] + [len(rows)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        for axis in axes:
            if lo:
                axis.axvline(lo - 0.5, color=LIGHT, linewidth=1)
        axes[-1].text((lo + hi - 1) / 2, -0.32, studies[lo].replace("ETS_", ""),
                      transform=axes[-1].get_xaxis_transform(), ha="center", va="top")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([r["participant"] for r in rows], rotation=90, fontsize=7)
    axes[0].set_title("Per meal, final model (bars) and annotation (circles)", loc="left")
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(out / f"meals{suffix}", dpi=200)
    plt.close(figure)


def chew_figure(rows, out: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    # Both sides over the annotated stretch only, so a false detection
    # elsewhere in the recording does not count against the chew estimate.
    pairs = (("annotated_chews", "model_chews_same_span", "Chews per meal"),
             ("annotated_chews_per_min", "model_chews_per_min_same_span",
              "Chewing rate (chews/min)"))
    for axis, (true_key, model_key, title) in zip(axes, pairs):
        print(f"\n  {title}")
        top = 0.0
        for flag, name, face in ((0, "held out of training", INK),
                                 (1, "used in training", "white")):
            group = [r for r in rows if int(r["in_training"]) == flag]
            t = np.array([number(r, true_key) for r in group])
            m = np.array([number(r, model_key) for r in group])
            ok = np.isfinite(t) & np.isfinite(m) & (t > 0)
            if not ok.any():
                continue
            t, m = t[ok], m[ok]
            top = max(top, t.max(), m.max())
            axis.plot(t, m, "o", markerfacecolor=face, markeredgecolor=INK,
                      markersize=6, label=f"{name} ({ok.sum()})")
            error = np.median(np.abs(m - t) / t)
            r = np.corrcoef(t, m)[0, 1] if ok.sum() > 2 else float("nan")
            print(f"    {name:<22} {ok.sum():>3} meals: r = {r:.2f}, median error "
                  f"{error:.0%}, model/annotated total {m.sum() / t.sum():.2f}")
        axis.plot([0, top * 1.05], [0, top * 1.05], color=MID, linestyle="--", linewidth=1)
        axis.set_xlim(0, top * 1.05)
        axis.set_ylim(0, top * 1.05)
        axis.set_xlabel("Annotated")
        axis.set_ylabel("Model")
        axis.set_title(title, loc="left")
        axis.legend(frameon=False, loc="upper left", fontsize=9)
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(out / f"chew_check{suffix}", dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--results", default="database_results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    with open(Path(args.results) / "sessions.csv", newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"no processed recordings in {args.results}\\sessions.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meal_figure(rows, out)
    print(f"{len(rows)} meals; wrote {out / 'meals.png'}")
    annotated = [r for r in rows if r.get("annotated_chews")]
    if annotated:
        chew_figure(annotated, out)
        print(f"\nwrote {out / 'chew_check.png'}")
    else:
        print("no annotated sessions with chew counts; chew_check not drawn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

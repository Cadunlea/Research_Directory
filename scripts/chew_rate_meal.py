r"""
chew_rate_meal.py - chewing rate over the whole meal (Dr. Sazonov's suggestion),
next to chewing rate while chewing.

Two definitions, both computed from the annotations (the truth) and from
peak counting in the optical signal (the estimate), one value per meal:

  while chewing   chews / minutes of eating windows
                  (what chew_rate_check.py and the slides report)
  over the meal   total chews / meal duration, where the meal runs from the
                  start of its first eating window to the end of its last,
                  so pauses, talking and sips between bites count

For "over the meal", the annotated total is every annotated chew between the
meal's first and last eating window (including chews in windows not labeled
eating); the estimate counts peaks only in the eating windows, as the
pipeline would. Eating windows are the annotated ones here, as in
chew_rate_check.py, so this measures the rate method on its own. No training,
so every meal is a fair test. Each definition is also compared with a
baseline that gives every meal the same rate (the mean annotated rate).

    python chew_rate_meal.py
    python chew_rate_meal.py --trend

Reads the window cache only. Writes results_fixed\chew_rate_meal.csv and
figures\chew_rate_meal.png / .pdf (grayscale, Arial).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

import ets_data                                                 # noqa: E402
import ets_train                                                # noqa: E402
from chew_rate import optical, peak_rate                        # noqa: E402

INK, MID = "#222222", "#888888"


def stats(true: np.ndarray, est: np.ndarray):
    error = np.abs(est - true) / true
    r = np.corrcoef(true, est)[0, 1] if np.ptp(est) > 1e-9 else float("nan")
    return r, float(np.median(error))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache", default=str(Path("ets_cache") / "ets_windows_w8_h4.npz"))
    parser.add_argument("--min-windows", type=int, default=5,
                        help="meals need at least this many eating windows (default 5)")
    parser.add_argument("--trend", action="store_true",
                        help="also draw the least-squares trend line")
    parser.add_argument("--out", default=str(Path("figures") / "chew_rate_meal"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False})

    w = ets_data.load_cache(Path(args.cache))
    fs, win = ets_data.SENSOR_FS, w.window_seconds
    grid = np.flatnonzero(ets_train.evaluation_grid(w.participants, w.studies, w.starts,
                                                    win, w.hop_seconds))
    eating = grid[w.y[grid] == 1]
    peaks = np.zeros(len(w.y))
    peaks[eating] = peak_rate(optical(w.X[eating]), fs) * win / 60      # peaks per window

    meals = []
    keys = sorted({(p, s) for p, s in zip(w.participants[eating], w.studies[eating])})
    for person, study in keys:
        mine = grid[(w.participants[grid] == person) & (w.studies[grid] == study)]
        eat = mine[w.y[mine] == 1]
        if len(eat) < args.min_windows:
            continue
        first, last = w.starts[eat].min(), w.starts[eat].max() + win
        span = mine[(w.starts[mine] >= first) & (w.starts[mine] < last)]
        chewing_min = len(eat) * win / 60
        meal_min = (last - first) / 60
        meals.append({
            "chewing_min": chewing_min, "meal_min": meal_min,
            "true_while": float(w.chews[eat].sum()) / chewing_min,
            "est_while": float(peaks[eat].sum()) / chewing_min,
            "true_meal": float(w.chews[span].sum()) / meal_min,
            "est_meal": float(peaks[eat].sum()) / meal_min,
        })
    if len(meals) < 3:
        raise SystemExit(f"only {len(meals)} meals with {args.min_windows}+ eating windows")
    col = lambda k: np.array([m[k] for m in meals])                     # noqa: E731

    print(f"\n{len(meals)} meals; time spent chewing is a median "
          f"{np.median(col('chewing_min') / col('meal_min')):.0%} of the meal\n")
    print(f"  {'definition':<16}{'annotated rate':>18}{'r':>7}{'median error':>15}"
          f"{'same rate for all':>20}")
    results = {}
    for key, name in (("while", "while chewing"), ("meal", "over the meal")):
        true, est = col(f"true_{key}"), col(f"est_{key}")
        r, e = stats(true, est)
        _, base = stats(true, np.full(len(true), true.mean()))
        slope = np.polyfit(true, est, 1)[0]
        results[key] = (true, est, r, e, slope)
        print(f"  {name:<16}{true.min():>8.0f} to {true.max():<4.0f}/min{r:>7.2f}"
              f"{e:>14.0%}{base:>19.0%}   trend slope {slope:.2f}")

    out_csv = Path("results_fixed") / "chew_rate_meal.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meals[0]))
        writer.writeheader()
        writer.writerows({k: f"{v:.3f}" for k, v in m.items()} for m in meals)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    titles = {"while": "Rate while chewing", "meal": "Rate over the whole meal"}
    for axis, key in zip(axes, ("while", "meal")):
        true, est, r, e, slope = results[key]
        low, high = min(true.min(), est.min()) * 0.9, max(true.max(), est.max()) * 1.05
        axis.plot([low, high], [low, high], color=MID, linestyle="--", linewidth=1,
                  label="perfect agreement")
        axis.plot(true, est, "o", markerfacecolor="white", markeredgecolor=INK, markersize=7)
        if args.trend:
            b, a = np.polyfit(true, est, 1)
            xs = np.array([true.min(), true.max()])
            axis.plot(xs, b * xs + a, color=INK, linewidth=1.8, label=f"trend (slope {b:.2f})")
            axis.legend(loc="upper left", frameon=False, fontsize=11)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{titles[key]} (chews/min)\nr = {r:.2f}, median error {e:.0%}",
                       loc="left", fontsize=13)
        axis.set_xlabel("Annotated rate")
        axis.set_ylabel("Estimated rate")
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300)
    print(f"\nwrote {out_csv}\n      {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

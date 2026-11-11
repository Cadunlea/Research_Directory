r"""
plot_chew_rate_slide.py - the chewing-rate figure for the slides, from peak counting only.

Two panels, one dot per meal, both from counting peaks in the optical (jaw)
signal in the annotated eating windows (no model, no training, so every meal
is a fair test):

  left   chews per meal: peaks counted against annotated chews
  right  chewing rate (chews per minute of eating): the same, divided by the
         minutes of eating

The dashed line is a perfect match. Each panel's title gives r and the median
error per meal; the console also prints the constant-rate baseline (every
meal given the mean annotated rate) for comparison. Grayscale, Arial, sized
for the full width of a 16:9 slide.

    python plot_chew_rate_slide.py

Reads the window cache only. Writes figures\chew_rate_slide.png / .pdf.
"""

from __future__ import annotations

import argparse
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
                        help="meals need at least this many eating windows (default 5, "
                             "as in chew_rate_check.py)")
    parser.add_argument("--out", default=str(Path("figures") / "chew_rate_slide"))
    parser.add_argument("--trend", action="store_true",
                        help="also draw the least-squares trend line (solid) next to the "
                             "perfect-agreement line (dashed), with a legend")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False})

    w = ets_data.load_cache(Path(args.cache))
    fs = ets_data.SENSOR_FS
    grid = ets_train.evaluation_grid(w.participants, w.studies, w.starts,
                                     w.window_seconds, w.hop_seconds)
    rows = np.flatnonzero(grid & (w.y == 1))
    peaks = peak_rate(optical(w.X[rows]), fs) * w.window_seconds / 60   # peaks per window
    minutes_per_window = w.window_seconds / 60

    true_chews, est_chews, minutes = [], [], []
    keys = sorted({(p, s) for p, s in zip(w.participants[rows], w.studies[rows])})
    for person, study in keys:
        idx = np.flatnonzero((w.participants[rows] == person) & (w.studies[rows] == study))
        if len(idx) < args.min_windows:
            continue
        true_chews.append(float(w.chews[rows][idx].sum()))
        est_chews.append(float(peaks[idx].sum()))
        minutes.append(len(idx) * minutes_per_window)
    true_chews, est_chews, minutes = map(np.array, (true_chews, est_chews, minutes))
    true_rate, est_rate = true_chews / minutes, est_chews / minutes
    if len(true_rate) < 3:
        raise SystemExit(f"only {len(true_rate)} meals with {args.min_windows}+ eating windows")

    r_c, e_c = stats(true_chews, est_chews)
    r_r, e_r = stats(true_rate, est_rate)
    _, e_base = stats(true_rate, np.full(len(true_rate), true_rate.mean()))
    print(f"{len(true_rate)} meals")
    print(f"  chews per meal: r = {r_c:.2f}, median error {e_c:.0%}")
    print(f"  chewing rate:   r = {r_r:.2f}, median error {e_r:.0%} "
          f"(constant rate for every meal: {e_base:.0%})")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = ((axes[0], true_chews, est_chews, "Chews per meal", "Annotated chews",
               "Counted peaks", r_c, e_c),
              (axes[1], true_rate, est_rate, "Chewing rate (chews per minute)",
               "Annotated rate", "Estimated rate", r_r, e_r))
    for axis, true, est, title, xlabel, ylabel, r, e in panels:
        low = min(true.min(), est.min()) * 0.9
        high = max(true.max(), est.max()) * 1.05
        axis.plot([low, high], [low, high], color=MID, linestyle="--", linewidth=1,
                  label="perfect agreement")
        axis.plot(true, est, "o", markerfacecolor="white", markeredgecolor=INK, markersize=7)
        if args.trend:
            slope, intercept = np.polyfit(true, est, 1)
            xs = np.array([true.min(), true.max()])
            axis.plot(xs, slope * xs + intercept, color=INK, linewidth=1.8,
                      label=f"trend (slope {slope:.2f})")
            axis.legend(loc="upper left", frameon=False, fontsize=11)
            print(f"  {title.split(' (')[0].lower()}: trend slope {slope:.2f} "
                  f"(1.00 = no compression)")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title}\nr = {r:.2f}, median error {e:.0%}", loc="left", fontsize=13)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

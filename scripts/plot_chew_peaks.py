r"""
plot_chew_peaks.py - one 8 s window of chewing, showing how chewing rate is measured.

For a slide that explains the method to a general audience: the raw optical
(jaw-muscle) signal of one annotated eating window on top, the same signal
band-passed to the chewing band (0.94-2.17 Hz) below with the counted peaks
marked, and the resulting rate. The window is chosen automatically: an
annotated eating window whose peak count equals its annotated chew count
(--pick N takes the N-th such window instead). No participant ID is drawn
or printed. Grayscale, Arial 11.

    python plot_chew_peaks.py
    python plot_chew_peaks.py --pick 5

Reads the window cache only. Writes figures\chew_peaks_example.png / .pdf.
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
from chew_rate import BAND, optical                             # noqa: E402

INK, MID = "#222222", "#888888"


def filtered_and_peaks(sig: np.ndarray, fs: float):
    """The band-passed signal and its peaks, exactly as chew_rate.peak_rate counts them."""
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, BAND, btype="band", fs=fs)
    filtered = filtfilt(b, a, sig)
    peaks, _ = find_peaks(filtered, distance=int(fs / BAND[1]),
                          prominence=0.5 * np.std(filtered))
    return filtered, peaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache", default=str(Path("ets_cache") / "ets_windows_w8_h4.npz"))
    parser.add_argument("--pick", type=int, default=0,
                        help="which of the matching windows to draw (default 0)")
    parser.add_argument("--out", default=str(Path("figures") / "chew_peaks_example"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    w = ets_data.load_cache(Path(args.cache))
    fs = ets_data.SENSOR_FS
    grid = ets_train.evaluation_grid(w.participants, w.studies, w.starts,
                                     w.window_seconds, w.hop_seconds)
    # Fully eating windows with no missing samples and a typical number of chews.
    fraction = w.eating_fraction if w.eating_fraction is not None else w.y.astype(float)
    rows = np.flatnonzero(grid & (w.y == 1) & (fraction >= 0.99) & (w.missing == 0)
                          & (w.chews >= 8) & (w.chews <= 14))
    matches = []
    for row in rows:
        sig = optical(w.X[row:row + 1])[0]
        filtered, peaks = filtered_and_peaks(sig, fs)
        if len(peaks) == int(w.chews[row]):
            matches.append((row, sig, filtered, peaks))
    if not matches:
        raise SystemExit("no window where the peak count equals the annotated chews")
    row, sig, filtered, peaks = matches[min(args.pick, len(matches) - 1)]
    t = np.arange(len(sig)) / fs
    rate = 60 * len(peaks) / w.window_seconds
    print(f"{len(matches)} matching windows; drawing number {min(args.pick, len(matches) - 1)}: "
          f"{len(peaks)} peaks, {int(w.chews[row])} annotated chews, {rate:.0f} chews/min")

    figure, axes = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
    axes[0].plot(t, sig, color=INK, linewidth=0.9)
    axes[0].set_ylabel("Raw optical\nsignal")
    axes[0].set_yticks([])
    axes[1].plot(t, filtered, color=INK, linewidth=1.2)
    axes[1].plot(t[peaks], filtered[peaks], "v", color=INK, markersize=8, markerfacecolor="white",
                 markeredgewidth=1.4)
    axes[1].axhline(0, color=MID, linewidth=0.6)
    axes[1].set_ylabel("Filtered to the\nchewing band")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(0, w.window_seconds)
    axes[0].set_title(f"{len(peaks)} peaks in 8 s = {rate:.0f} chews per minute "
                      f"(annotators marked {int(w.chews[row])} chews)", loc="left", fontsize=11)
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

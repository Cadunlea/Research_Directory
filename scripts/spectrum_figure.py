r"""
spectrum_figure.py - average frequency content of eating and non-eating windows.

An explanatory figure, not a model result: for each sensor channel, the mean
power spectrum of every eating window against every non-eating window, after
the same per-window normalisation the model sees. If chewing is visible in
the signal, the eating curve rises above the other in the chewing band
(about 0.94-2.17 Hz, Po et al. 2011), which is shaded. It shows WHY the
frequency content looks informative, and complements the finding that the
model still does better reading the time series than the spectrum.

    python spectrum_figure.py
    python spectrum_figure.py --cache ets_cache\ets_windows_w8_h4.npz --max-hz 10

Reads the window cache the training runs built; no database needed.
Grayscale, Arial 11 pt, like the other figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

import ets_data                                                      # noqa: E402

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
CHANNELS = ("Accelerometer X", "Accelerometer Y", "Accelerometer Z", "Optical")
CHEWING_BAND = (0.94, 2.17)


def mean_power(windows: np.ndarray, batch: int = 2000) -> np.ndarray:
    """Mean |FFT|^2 per frequency bin and channel, in batches to bound memory."""
    total = None
    for start in range(0, len(windows), batch):
        x = ets_data.normalise(windows[start:start + batch])
        power = np.abs(np.fft.rfft(x, axis=1)) ** 2
        part = power.sum(axis=0)
        total = part if total is None else total + part
    return total / len(windows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache", default="ets_cache/ets_windows_w8_h4.npz")
    parser.add_argument("--max-hz", type=float, default=8.0)
    parser.add_argument("--out", default="figures/spectrum_eating_vs_not")
    args = parser.parse_args()

    windows = ets_data.load_cache(Path(args.cache))
    eating = windows.y == 1
    print(f"{len(windows):,} windows: {int(eating.sum()):,} eating, "
          f"{int((~eating).sum()):,} not")
    freqs = np.fft.rfftfreq(windows.X.shape[1], d=1.0 / ets_data.SENSOR_FS)
    shown = (freqs > 0) & (freqs <= args.max_hz)

    eat = mean_power(windows.X[eating])
    rest = mean_power(windows.X[~eating])

    figure, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True)
    for channel, axis in enumerate(axes.ravel()):
        axis.axvspan(*CHEWING_BAND, color="#e6e6e6", linewidth=0)
        axis.plot(freqs[shown], 10 * np.log10(eat[shown, channel]),
                  color="#222222", linewidth=1.6, label="Eating")
        axis.plot(freqs[shown], 10 * np.log10(rest[shown, channel]),
                  color="#888888", linewidth=1.6, linestyle="--", label="Not eating")
        axis.set_title(CHANNELS[channel], loc="left", fontsize=11)
        axis.set_ylabel("Mean power (dB)")
        band = (freqs >= CHEWING_BAND[0]) & (freqs <= CHEWING_BAND[1])
        ratio = 10 * np.log10(eat[band, channel].mean() / rest[band, channel].mean())
        print(f"  {CHANNELS[channel]:<16} eating vs not in the chewing band: "
              f"{ratio:+.1f} dB")
    for axis in axes[1]:
        axis.set_xlabel("Frequency (Hz)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    figure.text(0.01, 0.005, f"Shaded: chewing band, {CHEWING_BAND[0]}-"
                f"{CHEWING_BAND[1]} Hz (Po et al. 2011). Per-window normalised "
                f"signals, 8 s windows.", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""
chew_rate_check.py - can chewing rate be read from the optical signal's rhythm?

The model's chew output gives every meal about the same chewing rate, so it
cannot tell a fast chewer from a slow one. Chewing is a rhythm the jaw sensor
records directly, so this measures the rhythm instead, with no model:

  spectral peak   the strongest frequency of the optical signal inside the
                  chewing band (0.94-2.17 Hz, Po et al. 2011) in each eating
                  window, times 60, then the median over the meal's windows
  peak count      the optical signal band-passed to the chewing band, and its
                  peaks counted (at least 0.46 s apart, the band's upper
                  limit), per minute, over the meal's windows

Both are compared with the annotated rate of each meal: annotated chews in its
eating windows per minute of those windows. The baseline gives every meal the
same rate (the mean annotated rate); a method is only useful if it beats that.
Meals are sessions with at least --min-windows annotated eating windows.

Uses the annotated eating windows, so it measures the rate estimate alone;
in the pipeline the model's detected windows would be used instead.

    python chew_rate_check.py
    python chew_rate_check.py --cache ets_cache\ets_windows_w8_h4.npz

Reads the window cache only (no database, no model). Writes
results_fixed\chew_rate_check.csv and figures\chew_rate_check.png / .pdf.
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

BAND = (0.94, 2.17)             # Hz, natural chewing (Po et al. 2011)
PAD = 8192                      # FFT length: interpolates the 0.125 Hz bins
INK, MID = "#222222", "#777777"


def optical(X: np.ndarray) -> np.ndarray:
    """Optical channel, missing samples set to the window mean, mean removed."""
    sig = X[:, :, ets_data.OPTICAL_CHANNEL].astype(float)
    valid = sig != ets_data.MISSING
    means = np.where(valid, sig, 0).sum(1) / np.maximum(valid.sum(1), 1)
    sig = np.where(valid, sig, means[:, None])
    return sig - sig.mean(1, keepdims=True)


def spectral_rate(sig: np.ndarray, fs: float) -> np.ndarray:
    """Chews per minute per window, from the band's strongest frequency."""
    spectrum = np.abs(np.fft.rfft(sig * np.hanning(sig.shape[1]), n=PAD, axis=1))
    freqs = np.fft.rfftfreq(PAD, 1 / fs)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    return 60 * freqs[band][np.argmax(spectrum[:, band], axis=1)]


def peak_rate(sig: np.ndarray, fs: float) -> np.ndarray:
    """Chews per minute per window, from peaks of the band-passed signal."""
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, BAND, btype="band", fs=fs)
    filtered = filtfilt(b, a, sig, axis=1)
    seconds = sig.shape[1] / fs
    gap = int(fs / BAND[1])
    out = np.empty(len(sig))
    for i, row in enumerate(filtered):
        peaks, _ = find_peaks(row, distance=gap, prominence=0.5 * np.std(row))
        out[i] = 60 * len(peaks) / seconds
    return out


def summary(true: np.ndarray, est: np.ndarray) -> dict:
    error = np.abs(est - true) / true
    r = np.corrcoef(true, est)[0, 1] if np.ptp(est) > 1e-9 else float("nan")
    return {"r": r, "median_error": float(np.median(error)),
            "mean_ratio": float(np.mean(est) / np.mean(true))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache", default=str(Path("ets_cache") / "ets_windows_w8_h4.npz"))
    parser.add_argument("--min-windows", type=int, default=5,
                        help="meals need at least this many eating windows (default 5)")
    parser.add_argument("--out", default=str(Path("figures") / "chew_rate_check"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    w = ets_data.load_cache(Path(args.cache))
    grid = ets_train.evaluation_grid(w.participants, w.studies, w.starts,
                                     w.window_seconds, w.hop_seconds)
    rows = np.flatnonzero(grid & (w.y == 1))
    fs = ets_data.SENSOR_FS
    sig = optical(w.X[rows])
    spec, peak = spectral_rate(sig, fs), peak_rate(sig, fs)
    minutes = w.window_seconds / 60

    meals = []
    keys = sorted({(p, s) for p, s in zip(w.participants[rows], w.studies[rows])})
    for person, study in keys:
        idx = np.flatnonzero((w.participants[rows] == person) & (w.studies[rows] == study))
        if len(idx) < args.min_windows:
            continue
        chews = float(w.chews[rows][idx].sum())
        meals.append({"participant": str(person), "study": str(study), "windows": len(idx),
                      "annotated_rate": chews / (len(idx) * minutes),
                      "spectral_rate": float(np.median(spec[idx])),
                      "peak_rate": float(np.mean(peak[idx]))})
    if len(meals) < 3:
        raise SystemExit(f"only {len(meals)} meals with {args.min_windows}+ eating windows")
    true = np.array([m["annotated_rate"] for m in meals])
    methods = {"spectral peak": np.array([m["spectral_rate"] for m in meals]),
               "peak count": np.array([m["peak_rate"] for m in meals]),
               "same rate for all": np.full(len(meals), true.mean())}

    print(f"\n{len(meals)} meals ({len(rows):,} annotated eating windows); annotated "
          f"rate {true.min():.0f} to {true.max():.0f} chews/min, mean {true.mean():.0f}\n")
    print(f"  {'method':<20}{'r':>7}{'median error':>15}{'mean est/true':>15}")
    for name, est in methods.items():
        s = summary(true, est)
        r = "  -  " if np.isnan(s["r"]) else f"{s['r']:.2f}"
        print(f"  {name:<20}{r:>7}{s['median_error']:>14.0%}{s['mean_ratio']:>15.2f}")
    print("\n  For reference, the model's chew output (plot_meals.py, its detected windows): "
          "r = -0.23,\n  20% median error on the 8 held-out meals. These two methods use "
          "no training,\n  so every meal here is a fair test.")

    out_csv = Path("results_fixed") / "chew_rate_check.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meals[0]))
        writer.writeheader()
        writer.writerows(meals)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    top = max(true.max(), methods["spectral peak"].max(), methods["peak count"].max()) * 1.05
    low = min(true.min(), methods["spectral peak"].min(), methods["peak count"].min()) * 0.95
    for axis, name in zip(axes, ("spectral peak", "peak count")):
        s = summary(true, methods[name])
        axis.plot(true, methods[name], "o", markerfacecolor="white", markeredgecolor=INK, markersize=6)
        axis.plot([low, top], [low, top], color=MID, linestyle="--", linewidth=1)
        axis.set_xlim(low, top)
        axis.set_ylim(low, top)
        axis.set_xlabel("Annotated chewing rate (chews/min)")
        axis.set_ylabel("Estimated (chews/min)")
        axis.set_title(f"{name.capitalize()}: r = {s['r']:.2f}, median error "
                       f"{s['median_error']:.0%}", loc="left", fontsize=11)
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=200)
    print(f"\nwrote {out_csv}\n      {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

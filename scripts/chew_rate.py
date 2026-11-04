"""
chew_rate.py - chewing rate from the rhythm of the AIM-2 optical signal.

During eating the jaw sensor's signal rises and falls once per chew. Filtered
to the natural chewing band (0.94-2.17 Hz, Po et al. 2011), its peaks are
counted; peaks closer than the band's fastest period or smaller than half the
window's standard deviation are ignored. No training: the settings were fixed
from the literature before the data were seen. chew_rate_check.py measured it
against the chew annotations (r = 0.72, median error 7% per meal, against 10%
for a constant rate). spectral_rate is the alternative it was compared with.
"""

from __future__ import annotations

import numpy as np

import ets_data

BAND = (0.94, 2.17)             # Hz, natural chewing (Po et al. 2011)
PAD = 8192                      # FFT length: interpolates the 0.125 Hz bins


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

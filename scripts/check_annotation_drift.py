r"""
check_annotation_drift.py - measure the sensor/annotation time lag, per session,
straight from the database.

WHAT THIS ANSWERS
-----------------
The previous model hardcoded ANNOTATION_TIME_SHIFT_SECONDS = -8.0, because
false-positive inspection suggested food-like sensor activity showed up about
8-10 seconds BEFORE the annotation said eating had started. That was a patch
applied to a symptom: nobody measured the lag, so nobody knew its true size,
whether it varied between participants, or whether it was ever really there.

This script measures it. For each session it cross-correlates a chewing-activity
signal derived from the optical sensor against the annotated eating mask, over a
grid of candidate lags, and reports the lag that maximises agreement. No shift is
assumed anywhere.

SIGN: the reported lag is how much LATER sensor activity occurs than the
annotation claims - the seconds that would have to be ADDED to the annotation
timestamps to align them. It is therefore directly comparable to the old
ANNOTATION_TIME_SHIFT_SECONDS = -8.0.

Read the result like this:
    lag ~ 0 s across sessions      the data are aligned; no correction is needed
                                   and the old -8 s shift was an artefact
    lag ~ -8 s across sessions     the old finding is real and still present
    the same non-zero lag for all  a systematic convention error, fixable once
    lags scattered per session     per-session clock drift, a real data problem

A LIKELY CAUSE, WHICH THIS SCRIPT TESTS DIRECTLY
------------------------------------------------
The repository disagrees with itself about what data_timestamp means:

  plot_raw_data.py       data_index = (timestamp - day_start) * 128
                         -> data_timestamp is the packet's START

  the old training code  packet_start = data_timestamp - data_duration
                         -> data_timestamp is the packet's END

A packet is 8 seconds long. Choosing the wrong convention displaces every
sensor sample by exactly one packet - 8 seconds - which is precisely the size of
the shift that was hardcoded. So this script evaluates BOTH conventions and
reports the lag under each. If 'start' yields ~0 s and 'end' yields ~8 s, the
drift was never in the data at all: it was a units bug, and it is now gone.

USAGE
-----
    python check_annotation_drift.py
    python check_annotation_drift.py --participant AIM104611
    python check_annotation_drift.py --convention start
    python check_annotation_drift.py --max-lag 60 --no-plots

Outputs (in drift_check_outputs/):
    annotation_drift_per_session.csv   one row per session and convention
    drift_summary.png                  lag distribution, both conventions
    drift_<participant>_<study>.png    correlation-vs-lag curve per session

Uses only packages pinned in requirements_ETS_full.txt.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mysql.connector
except ImportError:  # pragma: no cover
    sys.exit("mysql-connector-python is not installed.\n"
             "    pip install -r requirements_ETS_full.txt")

from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.use("Agg")           # write files; never needs a display or TkAgg
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30071            # MariaDB. 30073 is phpMyAdmin, not the database.
DEFAULT_USER = "jitai_admin"

# Never store the password here: this repository lives on GitHub, and a
# committed credential survives in the history even after it is removed.
# Supply it with --config jitai_config.ini, or the ETS_DB_PASSWORD variable.
PASSWORD_ENV_VAR = "ETS_DB_PASSWORD"

RAW_DB = "aim_raw_data"
STUDY_DB = "study_data"

SENSOR_IDENTIFIER = "XYZO"
OPTICAL_CHANNEL = 3             # Acc X, Acc Y, Acc Z, Optical

SENSOR_FS = 128.0
MISSING = -1

# The chewing band. Chewing runs at roughly 1-2 Hz; the band is opened slightly
# on both sides so a fast or slow chewer is not filtered away. This matches the
# band the worker's own bout detector uses (worker_lib bout detection, 0.5-2.0 Hz)
# widened to 3 Hz to catch the harmonics that survive the optical sensor.
CHEW_BAND_LO_HZ = 0.8
CHEW_BAND_HI_HZ = 3.0
ENVELOPE_SMOOTH_S = 0.5         # envelope smoothing before downsampling

ANALYSIS_FS = 10.0              # ground truth is stored at 10 Hz; correlate there
DEFAULT_MAX_LAG_S = 30.0
LAG_STEP_S = 0.1                # one ground-truth sample

# A lag read off a correlation curve that never rises above the noise is not a
# measurement. Below this peak correlation the session is reported UNRELIABLE
# rather than contributing a number to the summary.
MIN_TRUSTWORTHY_CORRELATION = 0.10

OUTPUT_DIR = Path("drift_check_outputs")


# --------------------------------------------------------------------------- #
# time  (lib_jitai/posixtime.py convention, inlined so this script stands alone)
# --------------------------------------------------------------------------- #
def posix2datetime(timestamp: float) -> datetime:
    """UTC/GMT interpretation, exactly as lib_jitai/posixtime.py does it.

    NOT datetime.fromtimestamp(), which applies the machine's local timezone and
    would introduce a 5-6 hour error on a Windows box - a far larger version of
    the very artefact this script exists to measure.
    """
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
def connect(host: str, port: int, user: str, password: str, database: str):
    return mysql.connector.connect(host=host, port=port, user=user,
                                   password=password, database=database)


def query(conn, sql: str, params: Sequence[Any] = ()) -> List[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


def credentials_from_config(path: str) -> Tuple[str, str]:
    cfg = configparser.ConfigParser(allow_no_value=True)
    if not cfg.read(path):
        raise FileNotFoundError(f"could not open {path}")
    section = cfg["raw_database"]
    return section["db_user_name"], section["passwd"]


def resolve_credentials(args) -> Tuple[str, str]:
    """Username and password, without either being stored in this file."""
    user, password = args.user, args.password
    if args.config:
        user, password = credentials_from_config(args.config)
    if not password:
        password = os.environ.get(PASSWORD_ENV_VAR, "")
    if not password:
        raise SystemExit(
            "No database password supplied. Use one of:\n"
            f"    python {os.path.basename(sys.argv[0])} "
            f"--config path\\to\\jitai_config.ini\n"
            f"    set {PASSWORD_ENV_VAR}=...        (Windows)\n"
            f"    export {PASSWORD_ENV_VAR}=...     (WSL / Linux)\n"
            "    --password ...                   (ends up in shell history)")
    return user, password


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    participant: str
    study: str
    day_start: int              # posix, midnight of the record's day
    frequency: float            # ground-truth sampling rate, 10 Hz
    bout: np.ndarray            # BOGT, full day
    bite: np.ndarray            # BIGT, full day
    chew: Optional[np.ndarray]  # CHGT, full day (may be absent)

    def eating_mask(self) -> np.ndarray:
        """1 eating, 0 not eating, -1 not recorded.

        Eating is 'chew bout OR bite': a bout is sustained chewing and a bite is
        food entering the mouth. Chew COUNT is deliberately not part of the
        definition - it is a rate, not a state, and a window can sit inside a
        genuine eating bout while containing zero completed chews.

        A sample is -1 (unknown) only where BOTH sources are -1. Where one
        source recorded and the other did not, the recorded one decides.
        """
        bout, bite = self.bout, self.bite
        length = min(len(bout), len(bite))
        bout, bite = bout[:length], bite[:length]
        unknown = (bout == MISSING) & (bite == MISSING)
        eating = ((bout > 0) | (bite > 0)).astype(np.int8)
        eating[unknown] = MISSING
        return eating

    def annotated_range(self) -> Optional[Tuple[int, int]]:
        mask = self.eating_mask()
        present = np.flatnonzero(mask != MISSING)
        if present.size == 0:
            return None
        return int(present[0]), int(present[-1])


def load_sessions(conn, only_participant: Optional[str]) -> List[Session]:
    """One Session per (participantID, studyID) with ground truth."""
    sql = (f"SELECT participantID, studyID, data_identifier, data_timestamp, "
           f"data_sampling_frequency, data_ndarray_pickle "
           f"FROM {STUDY_DB}.numeric_data "
           f"WHERE data_identifier IN ('CHGT','BOGT','BIGT')")
    params: List[Any] = []
    if only_participant:
        sql += " AND participantID=%s"
        params.append(only_participant)
    sql += " ORDER BY participantID, studyID"

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for participant, study, identifier, timestamp, frequency, blob in query(
            conn, sql, params):
        try:
            array = np.asarray(pickle.loads(blob)).ravel()
        except Exception as exc:                                # noqa: BLE001
            print(f"  !! {participant}/{study}/{identifier}: cannot unpickle "
                  f"({type(exc).__name__}: {exc})")
            continue
        key = (str(participant), str(study))
        entry = grouped.setdefault(key, {"day_start": int(timestamp),
                                         "frequency": float(frequency or 10.0)})
        entry[identifier] = array

    sessions: List[Session] = []
    for (participant, study), entry in sorted(grouped.items()):
        if "BOGT" not in entry or "BIGT" not in entry:
            print(f"  !! {participant}/{study}: missing BOGT or BIGT; skipped")
            continue
        sessions.append(Session(participant=participant, study=study,
                                day_start=entry["day_start"],
                                frequency=entry["frequency"],
                                bout=entry["BOGT"], bite=entry["BIGT"],
                                chew=entry.get("CHGT")))
    return sessions


def load_optical(conn, participant: str, start_ts: float, end_ts: float,
                 convention: str) -> np.ndarray:
    """Optical channel at 128 Hz over [start_ts, end_ts), -1 where no sample.

    `convention` decides what data_timestamp means:
        'start'  the packet BEGINS at data_timestamp   (plot_raw_data.py)
        'end'    the packet ENDS at data_timestamp     (the old training code)
    The two differ by one packet length - 8 seconds - which is exactly the size
    of the shift the old model hardcoded, so both are measured.
    """
    total = int(round((end_ts - start_ts) * SENSOR_FS))
    optical = np.full(total, float(MISSING), dtype=np.float64)

    rows = query(conn,
                 f"SELECT data_timestamp, data_duration, data_ndarray_pickle "
                 f"FROM {RAW_DB}.numeric_data "
                 f"WHERE data_identifier=%s AND participantID=%s "
                 f"AND data_timestamp BETWEEN %s AND %s "
                 f"ORDER BY data_timestamp",
                 (SENSOR_IDENTIFIER, participant,
                  int(start_ts) - 64, int(end_ts) + 64))

    for timestamp, duration, blob in rows:
        try:
            array = np.asarray(pickle.loads(blob))
        except Exception:                                       # noqa: BLE001
            continue
        if array.ndim != 2:
            continue
        # Packets are stored 4 x N; the plotting scripts transpose to N x 4.
        channels = array if array.shape[0] == 4 else array.T
        if channels.shape[0] < 4:
            continue
        signal = np.asarray(channels[OPTICAL_CHANNEL], dtype=np.float64)

        if convention == "end":
            packet_start = float(timestamp) - float(duration or
                                                    len(signal) / SENSOR_FS)
        else:
            packet_start = float(timestamp)

        offset = int(round((packet_start - start_ts) * SENSOR_FS))
        lo_dst, hi_dst = max(offset, 0), min(offset + len(signal), total)
        if lo_dst >= hi_dst:
            continue
        lo_src = lo_dst - offset
        optical[lo_dst:hi_dst] = signal[lo_src:lo_src + (hi_dst - lo_dst)]

    return optical


# --------------------------------------------------------------------------- #
# the chewing-activity signal
# --------------------------------------------------------------------------- #
def chewing_activity(optical: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(activity at ANALYSIS_FS, validity at ANALYSIS_FS).

    Band-pass the optical signal to the chewing band, take the analytic
    envelope, smooth it, and average down to 10 Hz. The result is 'how much
    chewing-rate energy is present right now' - a continuous quantity that
    should rise and fall together with the annotated eating mask, which is
    exactly what makes a lag measurable.

    Missing samples (-1) are interpolated across before filtering, because a
    filter cannot be run over holes, and are then marked invalid so they never
    contribute to a correlation.
    """
    valid = optical != MISSING
    if valid.sum() < int(SENSOR_FS * 4):
        return np.zeros(0), np.zeros(0, dtype=bool)

    filled = optical.copy()
    if not valid.all():
        indices = np.arange(len(optical))
        filled[~valid] = np.interp(indices[~valid], indices[valid], optical[valid])

    filled = filled - filled.mean()
    nyquist = SENSOR_FS / 2.0
    b, a = butter(4, [CHEW_BAND_LO_HZ / nyquist, CHEW_BAND_HI_HZ / nyquist],
                  btype="band")
    banded = filtfilt(b, a, filled)

    envelope = np.abs(hilbert(banded))
    envelope = uniform_filter1d(envelope, size=max(1, int(ENVELOPE_SMOOTH_S * SENSOR_FS)))

    # 128 Hz -> 10 Hz.
    #
    # This MUST be done on the true time grid, not by averaging blocks of
    # samples. 128 / 10 is 12.8, and rounding that to a 13-sample block makes
    # the output 9.846 Hz while the rest of the code believes it is 10 Hz. The
    # error is not a constant offset - it grows with elapsed time, reaching
    # about 8 SECONDS by the middle of a 17-minute session. That would have
    # been reported as a genuine 8 s sensor/annotation drift, which is exactly
    # the artefact this script exists to rule out. A synthetic test with a
    # known injected lag caught it.
    #
    # The envelope has already been low-pass smoothed over ENVELOPE_SMOOTH_S,
    # so sampling it at exact 10 Hz instants is correct and introduces no
    # aliasing.
    duration = len(envelope) / SENSOR_FS
    target_times = np.arange(0.0, duration, 1.0 / ANALYSIS_FS)
    if target_times.size == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    source_times = np.arange(len(envelope)) / SENSOR_FS
    activity = np.interp(target_times, source_times, envelope)
    # A 10 Hz sample counts as valid when the 128 Hz data around it is mostly
    # real rather than interpolated across a hole.
    coverage_128 = uniform_filter1d(valid.astype(np.float64),
                                    size=max(1, int(round(SENSOR_FS / ANALYSIS_FS))))
    coverage = np.interp(target_times, source_times, coverage_128)
    return activity, coverage > 0.5


def lagged_correlation(activity: np.ndarray, activity_valid: np.ndarray,
                       eating: np.ndarray, eating_valid: np.ndarray,
                       lags: np.ndarray) -> np.ndarray:
    """Pearson r between the two series at each integer sample lag.

    SIGN CONVENTION (verified against synthetic data with a known injected
    offset, in both directions):

        the reported lag is how much LATER the sensor activity occurs than the
        annotation claims - equivalently, the number of seconds that would have
        to be ADDED to the annotation timestamps to line them up.

    So it is directly comparable to the old ANNOTATION_TIME_SHIFT_SECONDS:
    a reported -8 s here reproduces exactly the finding that hardcoded -8.0,
    namely sensor activity arriving 8 s BEFORE the annotation said eating.
    A reported 0 means no correction is warranted.

    Correlation is computed only over samples valid in both series at that lag,
    so missing sensor data and un-annotated time never invent agreement.
    """
    results = np.full(len(lags), np.nan)
    length = min(len(activity), len(eating))
    activity, activity_valid = activity[:length], activity_valid[:length]
    eating, eating_valid = eating[:length], eating_valid[:length]

    for index, lag in enumerate(lags):
        if lag >= 0:
            a = activity[lag:]
            a_ok = activity_valid[lag:]
            e = eating[:length - lag]
            e_ok = eating_valid[:length - lag]
        else:
            a = activity[:length + lag]
            a_ok = activity_valid[:length + lag]
            e = eating[-lag:]
            e_ok = eating_valid[-lag:]

        both = a_ok & e_ok
        if both.sum() < ANALYSIS_FS * 30:        # need >= 30 s of overlap
            continue
        x, y = a[both].astype(float), e[both].astype(float)
        if x.std() < 1e-9 or y.std() < 1e-9:     # one side is constant
            continue
        results[index] = float(np.corrcoef(x, y)[0, 1])
    return results


# --------------------------------------------------------------------------- #
# per-session measurement
# --------------------------------------------------------------------------- #
def measure_session(conn, session: Session, convention: str,
                    max_lag_s: float) -> Optional[dict]:
    annotated = session.annotated_range()
    if annotated is None:
        print(f"  {session.participant}/{session.study}: annotation is entirely "
              f"-1; nothing to align")
        return None

    first, last = annotated
    frequency = session.frequency
    span_start = session.day_start + first / frequency
    span_end = session.day_start + (last + 1) / frequency

    mask = session.eating_mask()[first:last + 1]
    eating = (mask == 1).astype(np.float64)
    eating_valid = mask != MISSING

    # Resample the annotation to ANALYSIS_FS if it is not already there.
    if abs(frequency - ANALYSIS_FS) > 1e-6:
        source = np.arange(len(eating)) / frequency
        target = np.arange(0, source[-1], 1.0 / ANALYSIS_FS)
        eating = np.interp(target, source, eating)
        eating_valid = np.interp(target, source,
                                 eating_valid.astype(float)) > 0.5

    # Pad the sensor window by the lag range so shifting never runs off the end.
    pad = max_lag_s + 5.0
    optical = load_optical(conn, session.participant,
                           span_start - pad, span_end + pad, convention)
    coverage = float(np.mean(optical != MISSING))
    if coverage < 0.05:
        print(f"  {session.participant}/{session.study} [{convention}]: only "
              f"{coverage:.1%} sensor coverage over the annotated window; skipped")
        return None

    activity, activity_valid = chewing_activity(optical)
    if activity.size == 0:
        print(f"  {session.participant}/{session.study} [{convention}]: no usable "
              f"optical signal")
        return None

    # The padding means activity[0] sits `pad` seconds before the annotation.
    pad_samples = int(round(pad * ANALYSIS_FS))
    activity = activity[pad_samples:]
    activity_valid = activity_valid[pad_samples:]

    max_lag_samples = int(round(max_lag_s * ANALYSIS_FS))
    step = max(1, int(round(LAG_STEP_S * ANALYSIS_FS)))
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, step)
    correlations = lagged_correlation(activity, activity_valid,
                                      eating, eating_valid, lags)

    if np.all(np.isnan(correlations)):
        print(f"  {session.participant}/{session.study} [{convention}]: "
              f"correlation undefined at every lag")
        return None

    best = int(np.nanargmax(correlations))
    best_lag_s = float(lags[best] / ANALYSIS_FS)
    best_r = float(correlations[best])
    zero_index = int(np.argmin(np.abs(lags)))
    zero_r = float(correlations[zero_index])
    reliable = best_r >= MIN_TRUSTWORTHY_CORRELATION

    eating_share = float(np.mean(eating[eating_valid] > 0.5)) if eating_valid.any() else float("nan")

    print(f"  {session.participant:<12} {session.study:<12} [{convention:<5}] "
          f"lag {best_lag_s:+6.1f}s  r={best_r:.3f}  (r at 0 lag {zero_r:.3f})  "
          f"coverage {coverage:.0%}  eating {eating_share:.0%}"
          f"{'' if reliable else '   UNRELIABLE - weak correlation'}")

    return {
        "participant": session.participant,
        "study": session.study,
        "convention": convention,
        "annotated_start": posix2datetime(span_start).isoformat(sep=" "),
        "annotated_seconds": round((last - first) / frequency, 1),
        "sensor_coverage": round(coverage, 4),
        "eating_fraction": round(eating_share, 4),
        "best_lag_seconds": round(best_lag_s, 2),
        "best_correlation": round(best_r, 4),
        "correlation_at_zero_lag": round(zero_r, 4),
        "reliable": reliable,
        "_lags": lags / ANALYSIS_FS,
        "_correlations": correlations,
    }


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_session(result: dict, output_dir: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(result["_lags"], result["_correlations"], linewidth=1.6,
             color="#1f4e79")
    plt.axvline(0, color="#888888", linestyle="--", linewidth=1,
                label="zero lag (no correction)")
    plt.axvline(result["best_lag_seconds"], color="#c0392b", linewidth=1.4,
                label=f"best lag {result['best_lag_seconds']:+.1f}s")
    plt.axvline(-8.0, color="#e67e22", linestyle=":", linewidth=1.2,
                label="old hardcoded -8s")
    plt.xlabel("lag (s)   positive = sensor activity occurs LATER than annotated")
    plt.ylabel("correlation with annotated eating")
    plt.title(f"{result['participant']}  {result['study']}  "
              f"[{result['convention']} convention]")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    name = (f"drift_{result['participant']}_{result['study']}_"
            f"{result['convention']}.png")
    plt.savefig(output_dir / name, dpi=150)
    plt.close()


def plot_summary(results: List[dict], output_dir: Path) -> None:
    conventions = sorted({r["convention"] for r in results})
    figure, axes = plt.subplots(1, len(conventions), figsize=(6 * len(conventions), 4.5),
                                squeeze=False)
    for index, convention in enumerate(conventions):
        axis = axes[0][index]
        lags = [r["best_lag_seconds"] for r in results
                if r["convention"] == convention and r["reliable"]]
        axis.axvline(0, color="#888888", linestyle="--", linewidth=1)
        axis.axvline(-8.0, color="#e67e22", linestyle=":", linewidth=1.2)
        if lags:
            axis.hist(lags, bins=np.arange(-30, 30.5, 1.0), color="#1f4e79")
            median = float(np.median(lags))
            axis.axvline(median, color="#c0392b", linewidth=1.6)
            axis.set_title(f"{convention} convention\nmedian {median:+.1f}s  "
                           f"(n={len(lags)})")
        else:
            axis.set_title(f"{convention} convention\nno reliable sessions")
        axis.set_xlabel("measured lag (s)")
        axis.set_ylabel("sessions")
        axis.grid(True, linestyle="--", alpha=0.4)
    figure.suptitle("Sensor / annotation lag, measured per session\n"
                    "dashed = zero lag, dotted = the old hardcoded -8 s",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(output_dir / "drift_summary.png", dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------- #
def verdict(results: List[dict]) -> None:
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)

    for convention in sorted({r["convention"] for r in results}):
        subset = [r for r in results if r["convention"] == convention]
        reliable = [r for r in subset if r["reliable"]]
        print(f"\n  {convention} convention  "
              f"({len(reliable)} reliable of {len(subset)} sessions)")
        if not reliable:
            print("    no session produced a trustworthy correlation peak")
            continue
        lags = np.array([r["best_lag_seconds"] for r in reliable])
        correlations = np.array([r["best_correlation"] for r in reliable])
        print(f"    median lag        : {np.median(lags):+.2f} s")
        print(f"    mean lag          : {lags.mean():+.2f} s")
        print(f"    spread (IQR)      : "
              f"{np.percentile(lags, 25):+.2f} .. {np.percentile(lags, 75):+.2f} s")
        print(f"    full range        : {lags.min():+.2f} .. {lags.max():+.2f} s")
        print(f"    median peak r     : {np.median(correlations):.3f}")
        within_one = int(np.sum(np.abs(lags) <= 1.0))
        print(f"    sessions within +-1 s of zero: {within_one}/{len(lags)}")

    reliable_all = [r for r in results if r["reliable"]]
    if not reliable_all:
        print("\n  Nothing measurable. Check sensor coverage over the annotated "
              "windows (probe_database.py --dump-session) before reading "
              "anything into this.")
        return

    best = min(sorted({r["convention"] for r in results}),
               key=lambda c: abs(np.median([r["best_lag_seconds"]
                                            for r in results
                                            if r["convention"] == c and r["reliable"]]
                                           or [999])))
    median_best = float(np.median([r["best_lag_seconds"] for r in reliable_all
                                   if r["convention"] == best]))
    print(f"\n  Closest to aligned: the '{best}' convention, median "
          f"{median_best:+.2f} s.")
    if abs(median_best) <= 1.0:
        print("  -> The data are aligned. No shift is needed, and the old -8 s")
        print("     correction should NOT be carried forward.")
    elif abs(abs(median_best) - 8.0) <= 2.0:
        print("  -> The residual lag is about one 8 s packet, which points at the")
        print("     packet timestamp convention rather than at clock drift.")
        print("     Compare the two conventions above before correcting anything.")
    else:
        print("  -> A real lag remains. Report the per-session numbers; do not")
        print("     paper over it with a constant.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the sensor/annotation lag per session.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=None,
                        help="prefer --config or the ETS_DB_PASSWORD variable")
    parser.add_argument("--config", default=None,
                        help="read credentials from a jitai_config.ini")
    parser.add_argument("--participant", default=None,
                        help="measure a single participant")
    parser.add_argument("--convention", default="both",
                        choices=["start", "end", "both"],
                        help="meaning of data_timestamp (default: test both)")
    parser.add_argument("--max-lag", type=float, default=DEFAULT_MAX_LAG_S,
                        help="lag search range in seconds (default 30)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    user, password = resolve_credentials(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"connecting to {args.host}:{args.port} as {user}")
    try:
        conn = connect(args.host, args.port, user, password, STUDY_DB)
    except mysql.connector.Error as exc:
        print(f"\nCONNECTION FAILED: {exc}")
        print("  30071 is MariaDB; 30073 is the phpMyAdmin web page.")
        return 1

    conventions = ["start", "end"] if args.convention == "both" else [args.convention]

    try:
        sessions = load_sessions(conn, args.participant)
        print(f"\n{len(sessions)} session(s) with ground truth\n")
        if not sessions:
            print("Nothing to measure. Run probe_database.py first.")
            return 1

        results: List[dict] = []
        for convention in conventions:
            print(f"--- data_timestamp treated as packet {convention.upper()} ---")
            for session in sessions:
                result = measure_session(conn, session, convention, args.max_lag)
                if result is None:
                    continue
                results.append(result)
                if not args.no_plots:
                    plot_session(result, output_dir)
            print()
    finally:
        conn.close()

    if not results:
        print("No session could be measured.")
        return 1

    csv_path = output_dir / "annotation_drift_per_session.csv"
    fields = [key for key in results[0] if not key.startswith("_")]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})
    print(f"wrote {csv_path}")

    if not args.no_plots:
        plot_summary(results, output_dir)
        print(f"wrote {output_dir / 'drift_summary.png'}")

    verdict(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

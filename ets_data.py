r"""
ets_data.py - the single place that turns the ETS database into training
windows. Imported by both training scripts so they cannot disagree about what
the data means.

WHAT THE DATABASE ACTUALLY CONTAINS  (measured by probe_database.py, 2026-09-15)
-------------------------------------------------------------------------------
aim_raw_data.numeric_data, data_identifier 'XYZO'
    one row per ~8 s packet, pickled int16 array shaped (4, 1022):
    Acc X, Acc Y, Acc Z, Optical at 128 Hz.
    data_timestamp  is the packet's START.  <- measured, not assumed; see below
    data_duration   8.0        (seconds)
    data_sampling_frequency 0.0078125     (a PERIOD, 1/128 - not a frequency)

study_data.numeric_data, data_identifier 'CHGT' / 'BOGT' / 'BIGT'
    one row per session per identifier, pickled int16 array of 864000 samples:
    a WHOLE DAY at 10 Hz, indexed from data_timestamp, which is midnight.
    data_duration            864000   (a SAMPLE COUNT, not seconds)
    data_sampling_frequency  10.0     (a real frequency, unlike the raw table)
    CHGT one flag per chew, BOGT chewing bout, BIGT bite.
    Values: 1 eating, 0 not eating, -1 NOT RECORDED.

The two tables therefore use OPPOSITE conventions for both data_duration and
data_sampling_frequency. Each is read explicitly below; nothing is shared.

Sessions are joined on (participantID, studyID). record_hash does NOT match
between the two tables - the ground-truth upload gives each of CHGT/BOGT/BIGT a
different hash, so the documented tuple_hash((studyID, participantID)) does not
hold there.

THE 8 SECOND SHIFT IS NOT APPLIED, BECAUSE IT DOES NOT EXIST
------------------------------------------------------------
The previous model hardcoded ANNOTATION_TIME_SHIFT_SECONDS = -8.0.
check_annotation_drift.py measured the true lag on all 20 sessions:

    data_timestamp read as packet START : median +0.10 s, all 20 within +-0.2 s
    data_timestamp read as packet END   : median -7.90 s, all 20 near -8 s

with the SAME peak correlation either way (0.716 vs 0.715). A genuine drift
would degrade the correlation; a units error displaces it while preserving its
shape. The -8 s was the old code computing packet_start = data_timestamp -
data_duration, i.e. reading a start timestamp as an end timestamp. This module
treats data_timestamp as the packet start and applies NO shift.

MISSING DATA
------------
-1 means NOT RECORDED, everywhere, in both tables. It is never a measurement:
  - ground truth -1  -> outside the annotated video; the window is DROPPED
  - sensor -1        -> no sample received; excluded from normalisation, and
                        windows above MAX_MISSING_FRACTION are dropped
"""

from __future__ import annotations

import configparser
import hashlib
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


# --------------------------------------------------------------------------- #
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30071            # MariaDB. 30073 is phpMyAdmin, not the database.
DEFAULT_USER = "jitai_admin"
PASSWORD_ENV_VAR = "ETS_DB_PASSWORD"

RAW_DB = "aim_raw_data"
STUDY_DB = "study_data"

SENSOR_IDENTIFIER = "XYZO"
CHANNEL_NAMES = ("acc_x", "acc_y", "acc_z", "optical")
OPTICAL_CHANNEL = 3

SENSOR_FS = 128.0
GROUND_TRUTH_FS = 10.0
PACKET_SECONDS = 8.0
MISSING = -1

# A window needs enough real sensor data to be worth training on. Every session
# currently reports 100% coverage, so this only guards future uploads.
MAX_MISSING_FRACTION = 0.5
# ...and enough annotated samples for its label to mean anything.
MIN_LABEL_COVERAGE = 0.5
# A window is EATING when more than half its annotated samples are eating.
# Same rule as the previous model, kept so the comparison stays honest.
EATING_THRESHOLD = 0.5


def posix2datetime(timestamp: float) -> datetime:
    """The project's UTC/GMT convention (lib_jitai/posixtime.py), inlined so
    this module needs no lib_jitai on sys.path. NOT datetime.fromtimestamp(),
    which applies the local Windows timezone and shifts everything 5-6 hours."""
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def credentials_from_config(path: str) -> Tuple[str, str]:
    cfg = configparser.ConfigParser(allow_no_value=True)
    if not cfg.read(path):
        raise FileNotFoundError(f"could not open {path}")
    section = cfg["raw_database"]
    return section["db_user_name"], section["passwd"]


def add_database_arguments(parser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=None,
                        help=f"prefer --config or ${PASSWORD_ENV_VAR}")
    parser.add_argument("--config", default=None,
                        help="jitai_config.ini to read credentials from")


def connect(args):
    """Open the database using whichever credential source is available.

    No password is stored in this repository: it lives on GitHub, and a
    committed credential survives in the history even after it is deleted.
    """
    user, password = args.user, args.password
    if getattr(args, "config", None):
        user, password = credentials_from_config(args.config)
    if not password:
        password = os.environ.get(PASSWORD_ENV_VAR, "")
    if not password:
        raise SystemExit(
            "No database password supplied. Use one of:\n"
            "    --config path\\to\\jitai_config.ini\n"
            f"    set {PASSWORD_ENV_VAR}=...        (Windows)\n"
            f"    export {PASSWORD_ENV_VAR}=...     (WSL / Linux)")
    return mysql.connector.connect(host=args.host, port=args.port, user=user,
                                   password=password, database=STUDY_DB)


def query(conn, sql: str, params: Sequence[Any] = ()) -> List[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    participant: str
    study: str
    day_start: int
    frequency: float
    chew: np.ndarray            # CHGT, whole day at 10 Hz
    bout: np.ndarray            # BOGT
    bite: np.ndarray            # BIGT

    def eating(self) -> np.ndarray:
        """1 eating, 0 not, -1 unknown.

        Eating is 'chewing bout OR bite'. A bout is sustained chewing; a bite is
        food entering the mouth. CHGT is excluded from the definition because it
        is a rate, not a state - a window can sit inside genuine eating and
        contain no completed chew - but it is used as the auxiliary target.
        """
        length = min(len(self.bout), len(self.bite))
        bout, bite = self.bout[:length], self.bite[:length]
        label = ((bout > 0) | (bite > 0)).astype(np.int8)
        label[(bout == MISSING) & (bite == MISSING)] = MISSING
        return label

    def annotated_range(self) -> Optional[Tuple[int, int]]:
        present = np.flatnonzero(self.eating() != MISSING)
        if present.size == 0:
            return None
        return int(present[0]), int(present[-1])

    @property
    def key(self) -> str:
        return f"{self.participant}/{self.study}"


def load_sessions(conn, participants: Optional[Sequence[str]] = None
                  ) -> List[Session]:
    """Every (participant, study) that has ground truth.

    There is no ALLOWED_PARTICIPANTS list: whoever is in the database is in the
    study. The previous model hardcoded 14 IDs, which silently went stale.
    """
    sql = (f"SELECT participantID, studyID, data_identifier, data_timestamp, "
           f"data_sampling_frequency, data_ndarray_pickle "
           f"FROM {STUDY_DB}.numeric_data "
           f"WHERE data_identifier IN ('CHGT','BOGT','BIGT')")
    params: List[Any] = []
    if participants:
        sql += " AND participantID IN (" + ",".join(["%s"] * len(participants)) + ")"
        params.extend(participants)

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for participant, study, identifier, timestamp, frequency, blob in query(
            conn, sql, params):
        try:
            array = np.asarray(pickle.loads(blob)).ravel()
        except Exception as exc:                                # noqa: BLE001
            print(f"  !! {participant}/{study}/{identifier}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        entry = grouped.setdefault((str(participant), str(study)),
                                   {"day_start": int(timestamp),
                                    "frequency": float(frequency or GROUND_TRUTH_FS)})
        entry[identifier] = array

    sessions = []
    for (participant, study), entry in sorted(grouped.items()):
        missing = [i for i in ("CHGT", "BOGT", "BIGT") if i not in entry]
        if missing:
            print(f"  !! {participant}/{study}: missing {','.join(missing)}; skipped")
            continue
        sessions.append(Session(participant=participant, study=study,
                                day_start=entry["day_start"],
                                frequency=entry["frequency"],
                                chew=entry["CHGT"], bout=entry["BOGT"],
                                bite=entry["BIGT"]))
    return sessions


def load_sensor(conn, session: Session, start_ts: float, end_ts: float
                ) -> np.ndarray:
    """(n_samples, 4) float32 at 128 Hz over [start_ts, end_ts), -1 where no
    sample was received.

    data_timestamp is the packet START. This is measured, not assumed - see the
    module docstring. Reading it as an end timestamp is what produced the
    phantom 8 second shift in the previous model.
    """
    total = int(round((end_ts - start_ts) * SENSOR_FS))
    block = np.full((total, 4), float(MISSING), dtype=np.float32)

    rows = query(conn,
                 f"SELECT data_timestamp, data_ndarray_pickle "
                 f"FROM {RAW_DB}.numeric_data "
                 f"WHERE data_identifier=%s AND participantID=%s AND studyID=%s "
                 f"AND data_timestamp BETWEEN %s AND %s "
                 f"ORDER BY data_timestamp",
                 (SENSOR_IDENTIFIER, session.participant, session.study,
                  int(start_ts) - int(PACKET_SECONDS) - 1, int(end_ts) + 1))

    for timestamp, blob in rows:
        try:
            array = np.asarray(pickle.loads(blob))
        except Exception:                                       # noqa: BLE001
            continue
        if array.ndim != 2:
            continue
        samples = (array if array.shape[0] == 4 else array.T)   # 4 x N
        if samples.shape[0] < 4:
            continue
        samples = samples[:4].T.astype(np.float32)              # N x 4

        offset = int(round((float(timestamp) - start_ts) * SENSOR_FS))
        lo, hi = max(offset, 0), min(offset + len(samples), total)
        if lo >= hi:
            continue
        block[lo:hi] = samples[lo - offset:lo - offset + (hi - lo)]

    return block


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
@dataclass
class WindowSet:
    """Windows from every session, with the metadata needed to split by
    participant and to trace any window back to a moment in a video."""
    X: np.ndarray               # (n, samples, 4) float32, RAW counts, -1 missing
    y: np.ndarray               # (n,) int8      1 eating, 0 not
    chews: np.ndarray           # (n,) float32   CHGT marks in the window
    participants: np.ndarray    # (n,) str
    studies: np.ndarray         # (n,) str
    starts: np.ndarray          # (n,) float64   posix start of the window
    missing: np.ndarray         # (n,) float32   fraction of -1 sensor samples
    window_seconds: float
    hop_seconds: float

    def __len__(self) -> int:
        return len(self.y)

    def summary(self) -> str:
        positives = int(self.y.sum())
        return (f"{len(self):,} windows of {self.window_seconds:g}s "
                f"(hop {self.hop_seconds:g}s) from "
                f"{len(np.unique(self.participants))} participants: "
                f"{positives:,} eating ({positives / max(len(self), 1):.1%}), "
                f"{len(self) - positives:,} not")


def build_windows(conn, sessions: Sequence[Session], window_seconds: float,
                  hop_seconds: Optional[float] = None,
                  verbose: bool = True) -> WindowSet:
    """Cut every session's annotated span into windows.

    Only the annotated span is used. Everything else in the day is -1, and
    treating it as 'not eating' would bury ~16 minutes of real annotation under
    24 hours of fabricated negatives - the majority-class collapse that makes a
    model look accurate while predicting one class.
    """
    hop_seconds = hop_seconds or window_seconds
    window_samples = int(round(window_seconds * SENSOR_FS))

    X_parts, y_parts, chew_parts = [], [], []
    participants, studies, starts, missing_parts = [], [], [], []

    for session in sessions:
        span = session.annotated_range()
        if span is None:
            if verbose:
                print(f"  {session.key:<26} annotation is entirely -1; skipped")
            continue
        first, last = span
        frequency = session.frequency
        span_start = session.day_start + first / frequency
        span_end = session.day_start + (last + 1) / frequency

        sensor = load_sensor(conn, session, span_start, span_end)
        eating = session.eating()
        chew = session.chew

        n_windows = 0
        kept = 0
        positives = 0
        offset = 0.0
        while offset + window_seconds <= (span_end - span_start):
            n_windows += 1
            start_sample = int(round(offset * SENSOR_FS))
            block = sensor[start_sample:start_sample + window_samples]
            if len(block) < window_samples:
                break

            gt_lo = first + int(round(offset * frequency))
            gt_hi = gt_lo + int(round(window_seconds * frequency))
            labels = eating[gt_lo:gt_hi]
            known = labels != MISSING
            offset += hop_seconds

            if known.mean() < MIN_LABEL_COVERAGE:
                continue
            missing_fraction = float(np.mean(block == MISSING))
            if missing_fraction > MAX_MISSING_FRACTION:
                continue

            label = int(labels[known].mean() > EATING_THRESHOLD)
            chew_marks = float(np.sum(chew[gt_lo:gt_hi] == 1))

            X_parts.append(block)
            y_parts.append(label)
            chew_parts.append(chew_marks)
            participants.append(session.participant)
            studies.append(session.study)
            starts.append(span_start + (start_sample / SENSOR_FS))
            missing_parts.append(missing_fraction)
            kept += 1
            positives += label

        if verbose:
            print(f"  {session.key:<26} {kept:>5} windows "
                  f"({positives:>4} eating, {kept - positives:>4} not)  "
                  f"{(span_end - span_start) / 60:.0f} min annotated")

    if not X_parts:
        raise RuntimeError(
            "No windows were built. Every session was dropped - check that "
            "sensor packets exist during the annotated windows "
            "(probe_database.py reports this).")

    return WindowSet(
        X=np.stack(X_parts).astype(np.float32),
        y=np.asarray(y_parts, dtype=np.int8),
        chews=np.asarray(chew_parts, dtype=np.float32),
        participants=np.asarray(participants),
        studies=np.asarray(studies),
        starts=np.asarray(starts, dtype=np.float64),
        missing=np.asarray(missing_parts, dtype=np.float32),
        window_seconds=window_seconds, hop_seconds=hop_seconds)


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: Path, window_seconds: float,
               hop_seconds: float) -> Path:
    tag = f"w{window_seconds:g}_h{hop_seconds:g}"
    return Path(cache_dir) / f"ets_windows_{tag}.npz"


def save_cache(windows: WindowSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=windows.X, y=windows.y, chews=windows.chews,
        participants=windows.participants, studies=windows.studies,
        starts=windows.starts, missing=windows.missing,
        window_seconds=windows.window_seconds, hop_seconds=windows.hop_seconds)


def load_cache(path: Path) -> WindowSet:
    data = np.load(path, allow_pickle=False)
    return WindowSet(
        X=data["X"], y=data["y"], chews=data["chews"],
        participants=data["participants"], studies=data["studies"],
        starts=data["starts"], missing=data["missing"],
        window_seconds=float(data["window_seconds"]),
        hop_seconds=float(data["hop_seconds"]))


def get_windows(args, window_seconds: float, hop_seconds: Optional[float] = None,
                cache_dir: str = "ets_cache", rebuild: bool = False,
                verbose: bool = True) -> WindowSet:
    """Windows from cache, or built from the database and then cached."""
    hop_seconds = hop_seconds or window_seconds
    path = cache_path(Path(cache_dir), window_seconds, hop_seconds)
    if path.is_file() and not rebuild:
        windows = load_cache(path)
        if verbose:
            print(f"loaded cache {path}")
            print(f"  {windows.summary()}")
        return windows

    conn = connect(args)
    try:
        sessions = load_sessions(conn)
        if verbose:
            print(f"{len(sessions)} session(s) with ground truth\n")
        windows = build_windows(conn, sessions, window_seconds, hop_seconds,
                                verbose=verbose)
    finally:
        conn.close()

    save_cache(windows, path)
    if verbose:
        print(f"\n{windows.summary()}")
        print(f"cached to {path}")
    return windows


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def normalise(X: np.ndarray, high_pass: bool = False) -> np.ndarray:
    """Per-window, per-channel z-score, with -1 samples excluded.

    Missing samples become 0 - the channel mean after z-scoring - rather than
    being left at -1. The previous model fed the literal -1 into the FFT, where
    it is not a neutral value at all but a large negative outlier relative to
    signals centred near 1000-3500 counts, adding a spurious step at every gap.

    `high_pass` removes the per-window mean trend (the slide deck's 1 Hz high
    pass, in its simplest defensible form): the optical DC level varies hugely
    between participants and with sensor placement, and it carries no
    information about chewing.
    """
    out = np.empty_like(X, dtype=np.float32)
    for index in range(X.shape[0]):
        window = X[index]
        for channel in range(window.shape[1]):
            values = window[:, channel]
            valid = values != MISSING
            if valid.sum() < 2:
                out[index, :, channel] = 0.0
                continue
            usable = values[valid].astype(np.float64)
            if high_pass:
                usable = usable - usable.mean()
            mean, std = usable.mean(), usable.std()
            scaled = (usable - mean) / (std if std > 1e-8 else 1.0)
            column = np.zeros(len(values), dtype=np.float32)
            column[valid] = scaled
            out[index, :, channel] = column
    return out


def fft_features(X: np.ndarray) -> np.ndarray:
    """(n, bins, 4) log-magnitude spectra - the previous model's representation.

    Normalisation happens first so missing samples are 0 rather than -1.
    """
    normalised = normalise(X)
    magnitude = np.abs(np.fft.rfft(normalised, axis=1)).astype(np.float32)
    magnitude = np.log1p(magnitude)
    mean = magnitude.mean(axis=1, keepdims=True)
    std = magnitude.std(axis=1, keepdims=True)
    return ((magnitude - mean) / np.where(std < 1e-8, 1.0, std)).astype(np.float32)


# --------------------------------------------------------------------------- #
# participant-level cross-validation
# --------------------------------------------------------------------------- #
def participant_folds(participants: np.ndarray, n_folds: int = 4,
                      seed: int = 9) -> List[np.ndarray]:
    """Split PARTICIPANTS (never windows) into folds.

    Windows from one participant are hugely correlated - consecutive 8 s slices
    of the same meal, the same person, the same sensor placement. Splitting
    windows at random would put near-duplicates on both sides and report a
    score that has nothing to do with a new participant.

    Participants are ordered by a hash of their ID and dealt round-robin into
    folds. That is deterministic for a given set of participants - the same
    database gives the same folds on every run, with no reliance on numpy's
    global random state - and it keeps the folds balanced, which matters at
    n=20 where one extra participant is 5% of the data.

    It is NOT stable under insertion: adding a participant re-deals everyone.
    Balance was preferred over insertion-stability because an unbalanced fold
    (hash-modulo assignment can easily give 3 participants in one fold and 7 in
    another) produces a noisier estimate, and because results are regenerated
    wholesale when new annotations arrive anyway. Quote the fold membership
    alongside any result so a later run can be compared like for like - the
    training scripts record it in their JSON output.
    """
    unique = np.unique(participants)
    keyed = sorted(unique, key=lambda p: hashlib.md5(
        f"{seed}:{p}".encode()).hexdigest())
    return [np.asarray(keyed[index::n_folds]) for index in range(n_folds)]


def fold_indices(windows: WindowSet, held_out: Sequence[str]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    test_mask = np.isin(windows.participants, held_out)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

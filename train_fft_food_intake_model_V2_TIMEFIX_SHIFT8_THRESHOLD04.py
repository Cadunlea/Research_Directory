r"""
train_fft_food_intake_model_V1_TIMEFIX_SHIFT8_THRESHOLD04.py

First-draft frequency-domain food-intake classifier for AIM sensor data.

This is your working TIMEFIX + SHIFT8 + THRESHOLD04 baseline, changed as
little as possible. Only three additions were made, all explicitly
requested and nothing else:

  1. [label rule] Ground-truth rule is now configurable (LABEL_RULE /
     --label-rule), default "bout_or_bite" = (Chew Bout > 0) OR (Bite > 0),
     matching your earlier instruction. The original three-way OR
     ("count_or_bout_or_bite") is still selectable if you want to compare.
     Everything else about labeling (50% window threshold, the -8s shift,
     etc.) is untouched.

  2. [missing-data logging] Per-participant mean missing fraction is now
     collected into a table and written to
     participant_missing_fraction_report.csv, on top of the console prints
     that were already there. No new exclusion logic -- the existing
     BAD_ALIGNMENT_MISSING_FRACTION=0.95 skip is unchanged.

  3. [error metrics] metrics_from_probabilities() now also returns
     "error_rate" (1 - accuracy). compute_alignment_error_seconds() /
     summarize_alignment_error() are added as reusable functions (formalizing
     the false-positive-distance check from your diagnostics) and are called
     once on the test set at the selected threshold; results are printed and
     saved alongside the other metrics.

Nothing else changed: same UTC/GMT posix conversion, same -8s annotation
shift, same fixed 0.40 threshold, same FFT feature extraction (missing
samples still left as -1 into the FFT, not interpolated), same model
architecture, same train/val/test split. If you want any of those changed
too, say so explicitly -- I'm deliberately not touching them this round.

TIMEFIX VERSION:
  - Interprets database POSIX timestamps using UTC/GMT-style conversion, matching
    Dr. Sazonov's posixtime.py workaround.
  - Does NOT use local datetime.fromtimestamp() / timestamp() for sensor alignment.
  - Prints sensor span, annotation span, overlap, and missing-data fraction so bad
    alignment is caught before trusting model metrics.
  - Adds an annotation time-shift correction for the observed sensor/annotation lag.
  - Default shift: annotation timestamps are moved 8 seconds earlier.
  - Uses a fixed decision threshold of 0.40 by default.

What this script does:
  1. Uses ONLY the 14 participant IDs listed in ALLOWED_PARTICIPANTS.
  2. Finds exactly one annotation CSV inside each participant folder under ANNOTATION_ROOT.
  3. Pulls XYZO sensor packets from the local AIM raw database.
  4. Builds 8-second, non-overlapping windows: 8 sec * 128 Hz = 1024 samples.
  5. Labels each window using the configurable label rule (default: Chew Bout OR Bite).
     label = 1 when >50% of annotation samples in the 8-sec window are food.
  6. Keeps missing sensor values as -1.
  7. Converts each 1024x4 time-domain window into FFT magnitude features.
  8. Trains a small 1D CNN over frequency bins.
  9. Splits by participant using a fixed random seed so there is no participant leakage.
 10. Uses a fixed 0.40 classification threshold by default.
 11. Still prints validation/test threshold sweeps for reference.

Expected environment:
  - Docker database running
  - database_env activated
  - jitai_config.ini available in the working directory or pass --config
  - Python packages: numpy, pandas, mysql-connector-python-rf or mysql-connector-python,
    scikit-learn, torch, matplotlib

Run example:
  cd C:\Users\cadunlea\Documents\GitHub\python_code\script_plot_participant_data
  python train_fft_food_intake_model.py --config jitai_config.ini

Notes:
  - This is intentionally a simple first draft model.
  - It avoids full VGG/Chaogram complexity for now.
  - The FFT input is directly frequency-aware and easier to justify/debug.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import pickle
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# USER SETTINGS
# =============================================================================

ANNOTATION_ROOT = Path(r"C:\Users\cadunlea\Desktop\Annotation\study_data\Eating_Trajectory")

# Use ONLY these participants, even if more participants exist in the database.
ALLOWED_PARTICIPANTS = [
    "AIM110460",
    "AIM111004",
    "AIM115896",
    "AIM116725",
    "AIM121260",
    "AIM122571",
    "AIM123808",
    "AIM125068",
    "AIM126487",
    "AIM127703",
    "AIM128555",
    "AIM128799",
    "AIM129679",
    "AIM133721",
]

SENSOR_SAMPLE_RATE_HZ = 128.0
ANNOTATION_SAMPLE_RATE_HZ = 10.0
WINDOW_SECONDS = 8
WINDOW_SAMPLES = int(SENSOR_SAMPLE_RATE_HZ * WINDOW_SECONDS)  # 1024

# IMPORTANT TIME ALIGNMENT NOTE:
# Dr. Sazonov's posixtime.py stores/reads database POSIX timestamps using
# UTC/GMT-style conversion (calendar.timegm / datetime.utcfromtimestamp).
# Therefore, this script also uses datetime.utcfromtimestamp and direct datetime
# subtraction for alignment. Do NOT use datetime.fromtimestamp() or .timestamp()
# here because those use the Windows local timezone and can create 5/6-hour shifts.
SENSOR_TIME_OFFSET_SECONDS = 0

# DIAGNOSTIC LABEL-ALIGNMENT FIX:
# False-positive inspection showed food-like sensor activity often appeared about
# 8-10 seconds BEFORE the annotation switched on. To test whether strict labels
# are delayed, this version shifts annotation timestamps earlier by 8 seconds.
# Negative means "move annotations earlier". Set to 0.0 to disable.
ANNOTATION_TIME_SHIFT_SECONDS = -8.0

# Keep the threshold fixed at 0.40, based on the earlier validation sweep.
FIXED_DECISION_THRESHOLD = 0.40

# --- Ground-truth label rule (only new setting added this round) ---
# "bout_or_bite"          : (Chew Bout > 0) OR (Bite > 0)        [default]
# "bout_only"             : (Chew Bout > 0)
# "bite_only"             : (Bite > 0)
# "count_or_bout_or_bite" : original V1 rule (all three OR'd)
LABEL_RULE = "bout_or_bite"
VALID_LABEL_RULES = ("bout_or_bite", "bout_only", "bite_only", "count_or_bout_or_bite")

RANDOM_SEED = 9
TEST_PARTICIPANT_COUNT = 2
VAL_PARTICIPANT_COUNT = 2

BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 1e-3

# Thresholds to test after training. This does not change the model; it only
# changes how cautious the final food/not-food decision is.
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]

# If a participant/session has almost entirely missing sensor values, skip it
# instead of poisoning the training set with all -1 windows.
SKIP_BAD_ALIGNMENT_SESSIONS = True
BAD_ALIGNMENT_MISSING_FRACTION = 0.95

OUTPUT_DIR = Path("fft_food_intake_outputs_v1_timefix_shift8_threshold04")
CACHE_PATH = OUTPUT_DIR / "windowed_fft_dataset_v1_timefix_shift8_threshold04.npz"
MODEL_PATH = OUTPUT_DIR / "best_fft_food_intake_model_v1_timefix_shift8_threshold04.pt"
METRICS_PATH = OUTPUT_DIR / "metrics_v1_timefix_shift8_threshold04.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_v1_timefix_shift8_threshold04.png"
TRAINING_CURVES_PATH = OUTPUT_DIR / "training_curves_v1_timefix_shift8_threshold04.png"
VAL_THRESHOLD_SWEEP_PATH = OUTPUT_DIR / "validation_threshold_sweep_v1_shift8.csv"
TEST_THRESHOLD_SWEEP_PATH = OUTPUT_DIR / "test_threshold_sweep_v1_shift8_for_reference.csv"
PARTICIPANT_DIAGNOSTICS_PATH = OUTPUT_DIR / "participant_missing_fraction_report.csv"

CHEW_COUNT_COLUMN_CANDIDATES = ["Chew Count"]
CHEW_BOUT_COLUMN_CANDIDATES = ["Chew Bout"]
BITE_COLUMN_CANDIDATES = ["Bite"]
TIMECODE_COLUMN_CANDIDATES = ["bin_timecode"]
VIDEO_TIME_COLUMN_CANDIDATES = ["video_time"]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PacketRecord:
    data_timestamp: int
    data_duration: float
    data_sampling_frequency: float
    data_ndarray_pickle: bytes


@dataclass
class AnnotationSession:
    participant_id: str
    session_id: str
    csv_path: Path


@dataclass
class WindowRecord:
    participant_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    label: int


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def load_config(config_path: str = "jitai_config.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser(allow_no_value=True)
    if len(config.read(config_path)) == 0:
        raise FileNotFoundError(f"Could not open config file: {config_path}")
    return config


def connect_raw_db(config: configparser.ConfigParser):
    host_candidates = [config["raw_database"]["host_name"], "127.0.0.1", "localhost"]
    tried = []

    for host in host_candidates:
        if host in tried:
            continue
        tried.append(host)
        try:
            return mysql.connector.connect(
                host=host,
                user=config["raw_database"]["db_user_name"],
                password=config["raw_database"]["passwd"],
                database=config["raw_database"]["database_name"],
                port=int(config["raw_database"]["port_number"]),
            )
        except mysql.connector.Error as exc:
            print(f"Could not connect using host={host}: {exc}")

    raise ConnectionError("Could not connect to the raw database using config host or localhost fallbacks.")


def get_database_participant_list(conn) -> List[str]:
    query = """
        SELECT DISTINCT participantID
        FROM numeric_data
        WHERE participantID IS NOT NULL
        ORDER BY participantID
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]


def get_sensor_packets(conn, participant_id: str) -> List[PacketRecord]:
    query = """
        SELECT data_timestamp, data_duration, data_sampling_frequency, data_ndarray_pickle
        FROM numeric_data
        WHERE participantID = %s AND data_identifier = 'XYZO'
        ORDER BY data_timestamp
    """
    with conn.cursor() as cur:
        cur.execute(query, (participant_id,))
        rows = cur.fetchall()

    packets = [
        PacketRecord(
            data_timestamp=int(row[0]),
            data_duration=float(row[1]),
            data_sampling_frequency=float(row[2]),
            data_ndarray_pickle=row[3],
        )
        for row in rows
    ]
    if not packets:
        raise ValueError(f"No XYZO packets found for participant {participant_id}.")
    return packets


# =============================================================================
# ANNOTATION CSV DISCOVERY + LOADING
# =============================================================================

def discover_annotation_sessions(annotation_root: Path, allowed_participants: Sequence[str]) -> List[AnnotationSession]:
    sessions: List[AnnotationSession] = []
    problems: List[str] = []

    for participant_id in allowed_participants:
        participant_folder = annotation_root / participant_id
        if not participant_folder.exists():
            problems.append(f"Missing folder for {participant_id}: {participant_folder}")
            continue

        csv_files = sorted(participant_folder.glob("*.csv"))
        if len(csv_files) != 1:
            problems.append(
                f"Expected exactly one CSV directly inside {participant_folder}, found {len(csv_files)}: "
                f"{[str(p.name) for p in csv_files]}"
            )
            continue

        sessions.append(
            AnnotationSession(
                participant_id=participant_id,
                session_id="main",
                csv_path=csv_files[0],
            )
        )

    if problems:
        print("\nAnnotation discovery problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise FileNotFoundError("Fix annotation folder/CSV issues before training.")

    return sessions


def choose_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Could not find any of these columns: {list(candidates)}")
    return None


def parse_timecode_to_datetime(text: str, base_date: datetime.date) -> Tuple[datetime, bool]:
    """
    Parse annotation timecodes from CSV files.

    Handles both formats seen in the AIM annotation CSVs:
      - time only:      '19:06:42.4  or  19:06:42.4
      - full datetime:  2026-02-12 14:22:18.6

    Returns:
      (parsed_datetime, used_full_datetime)
    """
    cleaned = str(text).strip().replace("'", "")
    if not cleaned or cleaned.lower() == "nan":
        raise ValueError("Empty timecode")

    # Some annotation files store bin_timecode as a full absolute datetime.
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt), True
        except ValueError:
            pass

    # Most annotation files store bin_timecode as time-of-day only.
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(cleaned, fmt).time()
            return datetime.combine(base_date, parsed_time), False
        except ValueError:
            pass

    raise ValueError(f"Could not parse bin_timecode value: {text!r}")


def parse_video_time_to_seconds(text: str) -> float:
    cleaned = str(text).strip()
    dt = datetime.strptime(cleaned, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000.0


def build_annotation_datetimes(df: pd.DataFrame, base_date: datetime.date) -> pd.Series:
    """
    This follows the working plotting script: bin_timecode is treated as local clock time.
    After parsing, this script can apply a fixed annotation shift to test/compensate
    for observed sensor-vs-annotation delay.
    """
    timecode_col = choose_column(df, TIMECODE_COLUMN_CANDIDATES, required=False)
    if timecode_col is not None:
        abs_times = []
        day_offset = 0
        prev_seconds = None
        for raw in df[timecode_col]:
            dt, used_full_datetime = parse_timecode_to_datetime(raw, base_date=base_date)
            sec = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000.0

            # If the CSV only gives time-of-day, detect midnight rollover.
            # If the CSV gives full datetimes, do not alter its date.
            if not used_full_datetime:
                if prev_seconds is not None and sec < prev_seconds:
                    day_offset += 1
                dt = dt + timedelta(days=day_offset)

            prev_seconds = sec
            abs_times.append(dt)
        return pd.Series(abs_times, index=df.index)

    video_time_col = choose_column(df, VIDEO_TIME_COLUMN_CANDIDATES, required=True)
    first_zero = parse_video_time_to_seconds(df[video_time_col].iloc[0])
    start_dt = datetime.combine(base_date, datetime.min.time()) + timedelta(seconds=first_zero)
    return pd.Series(
        [start_dt + timedelta(seconds=parse_video_time_to_seconds(v) - first_zero) for v in df[video_time_col]],
        index=df.index,
    )


def series_from_candidates(df: pd.DataFrame, candidates: Sequence[str], default_value: float = 0.0) -> pd.Series:
    col = choose_column(df, candidates, required=False)
    if col is None:
        return pd.Series(np.full(len(df), default_value), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default_value).astype(float)


def compute_food_indicator(
    chew_count: pd.Series, chew_bout: pd.Series, bite: pd.Series, label_rule: str = LABEL_RULE
) -> pd.Series:
    """The only change to labeling logic this round: the OR rule is now
    selectable instead of hard-coded to the original three-way OR."""
    if label_rule == "bout_or_bite":
        food = (chew_bout > 0) | (bite > 0)
    elif label_rule == "bout_only":
        food = (chew_bout > 0)
    elif label_rule == "bite_only":
        food = (bite > 0)
    elif label_rule == "count_or_bout_or_bite":
        food = (chew_count > 0) | (chew_bout > 0) | (bite > 0)
    else:
        raise ValueError(f"Unknown label_rule {label_rule!r}; must be one of {VALID_LABEL_RULES}")
    return food.astype(float)


def load_annotation_csv(csv_path: Path, base_date: datetime.date, label_rule: str = LABEL_RULE) -> Dict[str, pd.Series]:
    df = pd.read_csv(csv_path)

    chew_count = series_from_candidates(df, CHEW_COUNT_COLUMN_CANDIDATES)
    chew_bout = series_from_candidates(df, CHEW_BOUT_COLUMN_CANDIDATES)
    bite = series_from_candidates(df, BITE_COLUMN_CANDIDATES)

    food = compute_food_indicator(chew_count, chew_bout, bite, label_rule=label_rule)
    times = build_annotation_datetimes(df, base_date=base_date)

    if ANNOTATION_TIME_SHIFT_SECONDS != 0:
        times = times + timedelta(seconds=ANNOTATION_TIME_SHIFT_SECONDS)

    return {
        "times": times,
        "food": food,
        "chew_count": chew_count,
        "chew_bout": chew_bout,
        "bite": bite,
    }


# =============================================================================
# SENSOR PACKET LOADING + WINDOWING
# =============================================================================

def to_n_by_4(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D packet array, got shape {arr.shape}")

    if arr.shape[0] == 4 and arr.shape[1] != 4:
        arr = arr.T
    elif arr.shape[1] == 4:
        pass
    elif 4 in arr.shape:
        arr = arr.reshape(4, -1).T
    else:
        raise ValueError(f"Could not coerce packet to N x 4. Shape was {arr.shape}")

    return arr.astype(np.float32, copy=False)


def load_packet_array(packet: PacketRecord) -> np.ndarray:
    arr = pickle.loads(packet.data_ndarray_pickle)
    arr = to_n_by_4(arr)

    # Preserve the working plotting script's expected-length logic.
    if packet.data_sampling_frequency > 0:
        expected_len = int(round(packet.data_duration / packet.data_sampling_frequency))
    else:
        expected_len = int(round(packet.data_duration * SENSOR_SAMPLE_RATE_HZ))

    if expected_len <= 0:
        return arr

    if arr.shape[0] < expected_len:
        pad_len = expected_len - arr.shape[0]
        pad_value = arr[-1:, :] if arr.shape[0] > 0 else np.full((1, 4), -1, dtype=np.float32)
        arr = np.vstack([arr, np.repeat(pad_value, pad_len, axis=0)])
    elif arr.shape[0] > expected_len:
        arr = arr[:expected_len, :]

    return arr.astype(np.float32, copy=False)


def db_posix_to_datetime(timestamp: float) -> datetime:
    """
    Convert database POSIX-style timestamps to naive datetimes using the same
    UTC/GMT interpretation as Dr. Sazonov's posixtime.py:

        posix2datetime(timestamp) -> datetime.utcfromtimestamp(timestamp)

    This intentionally avoids datetime.fromtimestamp(), which applies the local
    Windows timezone and can shift sensor data by 5/6 hours.
    """
    return datetime.utcfromtimestamp(float(timestamp))


def get_packet_start_end(packet: PacketRecord) -> Tuple[datetime, datetime]:
    """
    The existing plotting code treats data_timestamp as the packet end time and
    data_duration as the packet duration in seconds. Keep that convention, but
    convert using UTC/GMT-style datetime logic.
    """
    packet_end_dt = db_posix_to_datetime(packet.data_timestamp + SENSOR_TIME_OFFSET_SECONDS)
    packet_start_dt = db_posix_to_datetime(
        packet.data_timestamp + SENSOR_TIME_OFFSET_SECONDS - float(packet.data_duration)
    )
    return packet_start_dt, packet_end_dt


def get_sensor_span(packets: Sequence[PacketRecord]) -> Tuple[datetime, datetime]:
    first_start, _ = get_packet_start_end(packets[0])
    _, last_end = get_packet_start_end(packets[-1])
    return first_start, last_end


def build_sensor_window_array(packets: Sequence[PacketRecord], window_start: datetime, window_end: datetime) -> np.ndarray:
    """Return exactly WINDOW_SAMPLES x 4. Missing data remain -1."""
    total_samples = int(round((window_end - window_start).total_seconds() * SENSOR_SAMPLE_RATE_HZ))
    if total_samples != WINDOW_SAMPLES:
        total_samples = WINDOW_SAMPLES

    sensor_window = np.full((total_samples, 4), -1.0, dtype=np.float32)

    for packet in packets:
        packet_arr = load_packet_array(packet)
        packet_start_dt, _ = get_packet_start_end(packet)

        # Use direct naive datetime subtraction. Do NOT use .timestamp() here,
        # because .timestamp() applies the local Windows timezone.
        packet_start_idx = int(round((packet_start_dt - window_start).total_seconds() * SENSOR_SAMPLE_RATE_HZ))
        packet_end_idx = packet_start_idx + packet_arr.shape[0]

        dst_start = max(packet_start_idx, 0)
        dst_end = min(packet_end_idx, sensor_window.shape[0])
        if dst_start >= dst_end:
            continue

        src_start = dst_start - packet_start_idx
        src_end = src_start + (dst_end - dst_start)
        sensor_window[dst_start:dst_end, :] = packet_arr[src_start:src_end, :]

    return sensor_window


def make_windows_from_annotation_times(times: pd.Series) -> List[Tuple[datetime, datetime]]:
    start = times.iloc[0]
    end = times.iloc[-1]

    # Align to the first annotation timestamp and use non-overlapping 8-sec windows.
    windows = []
    current = start
    while current + timedelta(seconds=WINDOW_SECONDS) <= end:
        windows.append((current, current + timedelta(seconds=WINDOW_SECONDS)))
        current += timedelta(seconds=WINDOW_SECONDS)
    return windows


def label_window(annotation_data: Dict[str, pd.Series], window_start: datetime, window_end: datetime) -> Optional[int]:
    times = annotation_data["times"]
    food = annotation_data["food"]

    mask = (times >= window_start) & (times < window_end)
    if mask.sum() == 0:
        return None

    food_fraction = float((food[mask] > 0).mean())
    return int(food_fraction > 0.5)


# =============================================================================
# FFT FEATURE EXTRACTION
# =============================================================================

def zscore_keep_missing_minus_one(window: np.ndarray) -> np.ndarray:
    """
    Z-score each channel using only non-missing samples.
    Missing samples remain exactly -1, per user requirement.
    """
    out = window.astype(np.float32, copy=True)
    for ch in range(out.shape[1]):
        x = out[:, ch]
        valid = x != -1.0
        if valid.sum() < 2:
            continue
        mean = float(x[valid].mean())
        std = float(x[valid].std())
        if std < 1e-8:
            std = 1.0
        x_valid = (x[valid] - mean) / std
        out[valid, ch] = x_valid
        out[~valid, ch] = -1.0
    return out


def fft_magnitude_features(window: np.ndarray) -> np.ndarray:
    """
    Input:  window shape = (1024, 4)
    Output: feature shape = (4, 513), one FFT magnitude vector per channel.
    """
    if window.shape != (WINDOW_SAMPLES, 4):
        raise ValueError(f"Expected window shape {(WINDOW_SAMPLES, 4)}, got {window.shape}")

    window = zscore_keep_missing_minus_one(window)
    fft_vals = np.fft.rfft(window, axis=0)  # (513, 4)
    mag = np.abs(fft_vals).astype(np.float32)

    # Log compression helps keep extreme FFT magnitudes from dominating.
    mag = np.log1p(mag)

    # Per-channel normalization over frequency bins.
    for ch in range(mag.shape[1]):
        m = float(mag[:, ch].mean())
        s = float(mag[:, ch].std())
        if s < 1e-8:
            s = 1.0
        mag[:, ch] = (mag[:, ch] - m) / s

    return mag.T.astype(np.float32)  # (4, 513)


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_dataset(
    conn, sessions: Sequence[AnnotationSession], label_rule: str = LABEL_RULE
) -> Tuple[np.ndarray, np.ndarray, List[str], List[WindowRecord], List[dict]]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    participant_list: List[str] = []
    records: List[WindowRecord] = []
    global_raw_missing_fractions: List[float] = []
    skipped_sessions: List[str] = []
    participant_diagnostics: List[dict] = []  # new: per-participant table for CSV export

    db_participants = set(get_database_participant_list(conn))
    missing_in_db = [s.participant_id for s in sessions if s.participant_id not in db_participants]
    if missing_in_db:
        raise ValueError(f"These allowed participants have annotation folders but are missing from numeric_data: {missing_in_db}")

    for session in sessions:
        print(f"\nBuilding windows for {session.participant_id} ({session.session_id})")
        packets = get_sensor_packets(conn, session.participant_id)
        sensor_start, sensor_end = get_sensor_span(packets)
        annotation_data = load_annotation_csv(session.csv_path, base_date=sensor_start.date(), label_rule=label_rule)

        annotation_start = annotation_data["times"].iloc[0]
        annotation_end = annotation_data["times"].iloc[-1]
        overlap_seconds = max(
            0.0,
            (min(sensor_end, annotation_end) - max(sensor_start, annotation_start)).total_seconds(),
        )

        windows = make_windows_from_annotation_times(annotation_data["times"])
        print(f"  CSV: {session.csv_path.name}")
        print(f"  Sensor span:     {sensor_start} to {sensor_end}")
        print(f"  Annotation span: {annotation_start} to {annotation_end}")
        print(f"  Span overlap:    {overlap_seconds:.1f} seconds")
        print(f"  Candidate 8-sec windows: {len(windows)}")

        session_X: List[np.ndarray] = []
        session_y: List[int] = []
        session_participants: List[str] = []
        session_records: List[WindowRecord] = []
        session_missing_fractions: List[float] = []

        kept = 0
        positives = 0
        for window_start, window_end in windows:
            label = label_window(annotation_data, window_start, window_end)
            if label is None:
                continue

            sensor_window = build_sensor_window_array(packets, window_start, window_end)
            missing_fraction = float(np.mean(sensor_window == -1.0))
            session_missing_fractions.append(missing_fraction)
            features = fft_magnitude_features(sensor_window)

            session_X.append(features)
            session_y.append(label)
            session_participants.append(session.participant_id)
            session_records.append(
                WindowRecord(
                    participant_id=session.participant_id,
                    session_id=session.session_id,
                    start_time=window_start,
                    end_time=window_end,
                    label=label,
                )
            )
            kept += 1
            positives += int(label)

        mean_missing = float(np.mean(session_missing_fractions)) if session_missing_fractions else float("nan")
        print(f"  Kept windows: {kept}; positives: {positives}; negatives: {kept - positives}")
        print(f"  Mean missing fraction: {mean_missing:.4f}")

        excluded = False
        if mean_missing > BAD_ALIGNMENT_MISSING_FRACTION:
            message = (
                "  WARNING: This participant/session is almost entirely -1. "
                "Sensor/annotation time alignment is likely wrong."
            )
            if SKIP_BAD_ALIGNMENT_SESSIONS:
                print(message)
                print("  Skipping this participant/session so it does not poison training.")
                skipped_sessions.append(f"{session.participant_id}:{session.session_id}")
                excluded = True
            else:
                print(message)

        # New: log this participant/session into the diagnostics table regardless
        # of whether it was kept or skipped.
        participant_diagnostics.append({
            "participant_id": session.participant_id,
            "session_id": session.session_id,
            "csv_name": session.csv_path.name,
            "sensor_span_start": sensor_start.isoformat(),
            "sensor_span_end": sensor_end.isoformat(),
            "annotation_span_start": annotation_start.isoformat(),
            "annotation_span_end": annotation_end.isoformat(),
            "span_overlap_seconds": overlap_seconds,
            "candidate_windows": len(windows),
            "kept_windows": kept,
            "positive_windows": positives,
            "negative_windows": kept - positives,
            "mean_missing_fraction": mean_missing,
            "excluded": excluded,
        })

        if excluded:
            continue

        X_list.extend(session_X)
        y_list.extend(session_y)
        participant_list.extend(session_participants)
        records.extend(session_records)
        global_raw_missing_fractions.extend(session_missing_fractions)

    if skipped_sessions:
        print("\nSkipped sessions because of bad alignment / all-missing sensor windows:")
        for item in skipped_sessions:
            print(f"  {item}")

    if not X_list:
        raise RuntimeError("No windows were created. Check annotation timing and sensor alignment.")

    global_missing_fraction = float(np.mean(global_raw_missing_fractions)) if global_raw_missing_fractions else float("nan")
    print(f"\nGlobal mean raw missing fraction before FFT cache save: {global_missing_fraction:.4f}")
    if global_missing_fraction > BAD_ALIGNMENT_MISSING_FRACTION:
        raise RuntimeError(
            "Global mean raw missing fraction is > 0.95, so the model would train on almost all -1 values. "
            "Do not trust training until sensor/annotation time alignment is fixed."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y, participant_list, records, participant_diagnostics


def save_dataset_cache(
    X: np.ndarray, y: np.ndarray, participants: List[str], records: List[WindowRecord],
    participant_diagnostics: List[dict], path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record_json = [
        {
            "participant_id": r.participant_id,
            "session_id": r.session_id,
            "start_time": r.start_time.isoformat(),
            "end_time": r.end_time.isoformat(),
            "label": int(r.label),
        }
        for r in records
    ]
    np.savez_compressed(
        path,
        X=X,
        y=y,
        participants=np.asarray(participants, dtype=object),
        records=np.asarray([json.dumps(r) for r in record_json], dtype=object),
        participant_diagnostics=np.asarray([json.dumps(d) for d in participant_diagnostics], dtype=object),
    )
    print(f"\nSaved dataset cache to {path.resolve()}")


def load_dataset_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[dict], List[dict]]:
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    participants = [str(p) for p in data["participants"]]
    records = [json.loads(str(r)) for r in data["records"]]
    diag_key = "participant_diagnostics"
    diagnostics = [json.loads(str(d)) for d in data[diag_key]] if diag_key in data else []
    return X, y, participants, records, diagnostics


# =============================================================================
# PARTICIPANT SPLIT
# =============================================================================

def split_participants(participants: Sequence[str], seed: int) -> Tuple[List[str], List[str], List[str]]:
    unique_participants = sorted(set(participants))
    if len(unique_participants) < TEST_PARTICIPANT_COUNT + VAL_PARTICIPANT_COUNT + 1:
        raise ValueError("Not enough participants for the requested train/val/test split.")

    rng = random.Random(seed)
    shuffled = unique_participants[:]
    rng.shuffle(shuffled)

    test_pids = sorted(shuffled[:TEST_PARTICIPANT_COUNT])
    val_pids = sorted(shuffled[TEST_PARTICIPANT_COUNT:TEST_PARTICIPANT_COUNT + VAL_PARTICIPANT_COUNT])
    train_pids = sorted(shuffled[TEST_PARTICIPANT_COUNT + VAL_PARTICIPANT_COUNT:])

    return train_pids, val_pids, test_pids


def indices_for_participants(participant_per_window: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    selected_set = set(selected)
    return np.asarray([i for i, p in enumerate(participant_per_window) if p in selected_set], dtype=int)


# =============================================================================
# PYTORCH DATASET + MODEL
# =============================================================================

class FFTWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class FrequencyCNN(nn.Module):
    """Small first-draft CNN over FFT frequency bins."""
    def __init__(self, input_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(1)  # logits


# =============================================================================
# TRAINING + EVALUATION
# =============================================================================

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(FFTWindowDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_true.extend(yb.numpy().tolist())

    y_true = np.asarray(all_true).astype(int)
    y_prob = np.asarray(all_probs)
    return metrics_from_probabilities(y_true, y_prob, threshold)


def metrics_from_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    accuracy = float(accuracy_score(y_true, y_pred))
    # "Absolute error" on the raw probability output, distinct from the
    # thresholded classification error_rate above and from the seconds-based
    # alignment error below -- mean(|y_true - probability|).
    mean_absolute_error = float(np.mean(np.abs(y_true - y_prob))) if len(y_true) else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,  # classification "error" = 1 - accuracy
        "mean_absolute_error": mean_absolute_error,  # "absolute error" on probabilities
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive": int(y_pred.sum()),
        "predicted_negative": int(len(y_pred) - y_pred.sum()),
        "probability_min": float(y_prob.min()) if len(y_prob) else float("nan"),
        "probability_mean": float(y_prob.mean()) if len(y_prob) else float("nan"),
        "probability_max": float(y_prob.max()) if len(y_prob) else float("nan"),
    }


def collect_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_true.extend(yb.numpy().tolist())
    return np.asarray(all_true).astype(int), np.asarray(all_probs, dtype=float)


def threshold_sweep_from_probabilities(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]) -> pd.DataFrame:
    rows = [metrics_from_probabilities(y_true, y_prob, float(th)) for th in thresholds]
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, selection_metric: str = "balanced_accuracy") -> float:
    """
    Pick threshold using validation data only.

    Default is balanced_accuracy because raw accuracy can look good even when the
    model predicts only the majority class. Ties are resolved using F1, then accuracy.
    """
    if selection_metric not in sweep.columns:
        raise ValueError(f"Unknown threshold selection metric: {selection_metric}")

    ranked = sweep.sort_values(
        by=[selection_metric, "f1", "accuracy"],
        ascending=[False, False, False],
    )
    return float(ranked.iloc[0]["threshold"])


def compute_alignment_error_seconds(window_records: Sequence[dict], y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    New: for every window, find the distance (seconds) to the nearest TRUE
    positive window start for the same participant. Formalizes the manual
    false-positive-distance diagnostic ("41/62 FPs within 8s...") as reusable
    code. Returns a per-window DataFrame; use summarize_alignment_error() for
    the headline numbers.
    """
    df = pd.DataFrame(window_records)
    df["y_true"] = np.asarray(y_true).astype(int)
    df["y_pred"] = np.asarray(y_pred).astype(int)
    df["start_time"] = pd.to_datetime(df["start_time"])

    results = []
    for participant_id, group in df.groupby("participant_id"):
        true_times = group.loc[group["y_true"] == 1, "start_time"].to_numpy()
        for _, row in group.iterrows():
            if len(true_times) == 0:
                nearest = np.nan
            else:
                diffs = np.abs((true_times - np.datetime64(row["start_time"])) / np.timedelta64(1, "s"))
                nearest = float(diffs.min())
            results.append({
                "participant_id": participant_id,
                "start_time": row["start_time"],
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"]),
                "correct": int(row["y_true"] == row["y_pred"]),
                "nearest_true_food_seconds": nearest,
            })

    return pd.DataFrame(results)


def summarize_alignment_error(alignment_df: pd.DataFrame) -> dict:
    """New: headline numbers for false positives specifically -- mirrors
    "66% of false positives were within 8 seconds of labeled food" etc."""
    fp = alignment_df[(alignment_df["y_pred"] == 1) & (alignment_df["y_true"] == 0)]
    if len(fp) == 0:
        return {"n_false_positives": 0}
    return {
        "n_false_positives": int(len(fp)),
        "mean_abs_seconds_to_nearest_true_food": float(fp["nearest_true_food_seconds"].mean()),
        "median_abs_seconds_to_nearest_true_food": float(fp["nearest_true_food_seconds"].median()),
        "fraction_within_8s": float((fp["nearest_true_food_seconds"] <= 8).mean()),
        "fraction_within_16s": float((fp["nearest_true_food_seconds"] <= 16).mean()),
        "fraction_within_32s": float((fp["nearest_true_food_seconds"] <= 32).mean()),
    }


def train_model(X_train, y_train, X_val, y_val, device: torch.device) -> Tuple[nn.Module, Dict[str, List[float]]]:
    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = FrequencyCNN(input_channels=X_train.shape[1]).to(device)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    if positive_count > 0:
        pos_weight_value = negative_count / positive_count
    else:
        pos_weight_value = 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    best_val_loss = math.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_metrics = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | val F1 {val_metrics['f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Food", "Food"])
    plt.yticks([0, 1], ["No Food", "Food"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    global ANNOTATION_TIME_SHIFT_SECONDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="jitai_config.ini", help="Path to jitai_config.ini")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild dataset cache from DB and CSVs")
    parser.add_argument("--annotation-root", default=str(ANNOTATION_ROOT), help="Root folder containing AIM participant annotation folders")
    parser.add_argument("--threshold", type=float, default=FIXED_DECISION_THRESHOLD, help="Fixed decision threshold. Default is 0.40 for the shift-corrected V1 test.")
    parser.add_argument("--threshold-selection", default="balanced_accuracy", choices=["balanced_accuracy", "f1", "accuracy", "specificity", "recall"], help="Validation metric used only if --threshold is passed as None by editing the script.")
    parser.add_argument("--annotation-shift-seconds", type=float, default=ANNOTATION_TIME_SHIFT_SECONDS, help="Seconds to shift annotation timestamps. Negative moves annotations earlier. Default: -8.0.")
    parser.add_argument("--label-rule", default=LABEL_RULE, choices=VALID_LABEL_RULES, help="Ground-truth label rule. Default: bout_or_bite.")
    args = parser.parse_args()

    ANNOTATION_TIME_SHIFT_SECONDS = float(args.annotation_shift_seconds)

    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotation_root = Path(args.annotation_root)

    if CACHE_PATH.exists() and not args.rebuild_cache:
        print(f"Loading existing dataset cache: {CACHE_PATH.resolve()}")
        X, y, participant_per_window, records, participant_diagnostics = load_dataset_cache(CACHE_PATH)
    else:
        sessions = discover_annotation_sessions(annotation_root, ALLOWED_PARTICIPANTS)
        print("Found annotation sessions:")
        for s in sessions:
            print(f"  {s.participant_id}: {s.csv_path}")

        config = load_config(args.config)
        conn = connect_raw_db(config)
        try:
            X, y, participant_per_window, records, participant_diagnostics = build_dataset(conn, sessions, label_rule=args.label_rule)
        finally:
            conn.close()
        save_dataset_cache(X, y, participant_per_window, records, participant_diagnostics, CACHE_PATH)

    # New: write the per-participant missing-fraction table to CSV.
    if participant_diagnostics:
        diag_df = pd.DataFrame(participant_diagnostics)
        diag_df.to_csv(PARTICIPANT_DIAGNOSTICS_PATH, index=False)
        print(f"\nSaved per-participant missing-coverage report to {PARTICIPANT_DIAGNOSTICS_PATH.resolve()}")
        print(diag_df[["participant_id", "kept_windows", "mean_missing_fraction", "excluded"]].to_string(index=False))

    print("\nDataset summary")
    print(f"  X shape: {X.shape}  # windows x channels x frequency_bins")
    print(f"  y shape: {y.shape}")
    print(f"  Total positives: {int(y.sum())}")
    print(f"  Total negatives: {int(len(y) - y.sum())}")
    print(f"  Participants: {sorted(set(participant_per_window))}")

    train_pids, val_pids, test_pids = split_participants(participant_per_window, RANDOM_SEED)
    print("\nParticipant split")
    print(f"  Train: {train_pids}")
    print(f"  Val:   {val_pids}")
    print(f"  Test:  {test_pids}")

    train_idx = indices_for_participants(participant_per_window, train_pids)
    val_idx = indices_for_participants(participant_per_window, val_pids)
    test_idx = indices_for_participants(participant_per_window, test_pids)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\nWindow split")
    print(f"  Train windows: {len(y_train)}  positives={int(y_train.sum())} negatives={int(len(y_train)-y_train.sum())}")
    print(f"  Val windows:   {len(y_val)}  positives={int(y_val.sum())} negatives={int(len(y_val)-y_val.sum())}")
    print(f"  Test windows:  {len(y_test)}  positives={int(y_test.sum())} negatives={int(len(y_test)-y_test.sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model, history = train_model(X_train, y_train, X_val, y_val, device=device)

    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    y_val_true, y_val_prob = collect_probabilities(model, val_loader, device=device)
    val_sweep = threshold_sweep_from_probabilities(y_val_true, y_val_prob, THRESHOLD_GRID)
    VAL_THRESHOLD_SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    val_sweep.to_csv(VAL_THRESHOLD_SWEEP_PATH, index=False)

    if args.threshold is not None:
        selected_threshold = float(args.threshold)
        print(f"\nUsing manually supplied threshold: {selected_threshold:.2f}")
    else:
        selected_threshold = choose_threshold(val_sweep, selection_metric=args.threshold_selection)
        print(f"\nSelected threshold from validation sweep: {selected_threshold:.2f}")
        print(f"Threshold selection metric: {args.threshold_selection}")

    print(f"Saved validation threshold sweep to {VAL_THRESHOLD_SWEEP_PATH.resolve()}")
    print("\nValidation threshold sweep")
    display_cols = ["threshold", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "fp", "fn", "tp", "tn"]
    print(val_sweep[display_cols].to_string(index=False))

    y_test_true, y_test_prob = collect_probabilities(model, test_loader, device=device)
    test_sweep = threshold_sweep_from_probabilities(y_test_true, y_test_prob, THRESHOLD_GRID)
    test_sweep.to_csv(TEST_THRESHOLD_SWEEP_PATH, index=False)
    test_metrics = metrics_from_probabilities(y_test_true, y_test_prob, selected_threshold)

    print("\nTest metrics at selected threshold")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")

    # New: alignment-error diagnostic on the test set at the selected threshold.
    test_records = [r for r, p in zip(records, participant_per_window) if p in test_pids]
    y_pred_test = (y_test_prob >= selected_threshold).astype(int)
    alignment_df = compute_alignment_error_seconds(test_records, y_test_true, y_pred_test)
    alignment_summary = summarize_alignment_error(alignment_df)
    print("\nAlignment-error summary (false positives only):")
    print(json.dumps(alignment_summary, indent=2))

    print(f"\nSaved test threshold sweep for reference to {TEST_THRESHOLD_SWEEP_PATH.resolve()}")
    print("Note: choose/report the selected-threshold test metrics above. The test sweep is diagnostic only.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "allowed_participants": ALLOWED_PARTICIPANTS,
            "train_participants": train_pids,
            "val_participants": val_pids,
            "test_participants": test_pids,
            "window_seconds": WINDOW_SECONDS,
            "sensor_sample_rate_hz": SENSOR_SAMPLE_RATE_HZ,
            "selected_threshold": selected_threshold,
            "threshold_selection_metric": args.threshold_selection,
            "annotation_time_shift_seconds": ANNOTATION_TIME_SHIFT_SECONDS,
            "label_rule": args.label_rule,
            "metrics": test_metrics,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH.resolve()}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_metrics": test_metrics,
                "alignment_error_summary": alignment_summary,
                "selected_threshold": selected_threshold,
                "threshold_selection_metric": args.threshold_selection,
                "annotation_time_shift_seconds": ANNOTATION_TIME_SHIFT_SECONDS,
                "label_rule": args.label_rule,
                "train_participants": train_pids,
                "val_participants": val_pids,
                "test_participants": test_pids,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {METRICS_PATH.resolve()}")

    plot_training_curves(history, TRAINING_CURVES_PATH)
    plot_confusion_matrix(test_metrics, CONFUSION_MATRIX_PATH)
    print(f"Saved training curves to {TRAINING_CURVES_PATH.resolve()}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH.resolve()}")


if __name__ == "__main__":
    main()

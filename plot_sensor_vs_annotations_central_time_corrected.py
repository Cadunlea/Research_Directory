import configparser
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd


# =========================
# USER SETTINGS
# =========================
ANNOTATION_CSV_PATH = r"C:\Users\cadunlea\Desktop\Annotation\study_data\Eating_Trajectory\AIM110460\AIM110460_annotation_v00.csv"
PARTICIPANT_ID: Optional[str] = "AIM110460"  # Set to e.g. "AIM111004" to skip menu
SAVE_FIGURE = False
OUTPUT_DIR = Path("plots")
WINDOW_PADDING_MINUTES = 15

# Positive values move sensor timestamps later.
SENSOR_TIME_OFFSET_HOURS = 6
SENSOR_TIME_OFFSET_SECONDS = SENSOR_TIME_OFFSET_HOURS * 3600

FOOD_COLUMN_CANDIDATES = [
    "Food Intake Detection",
    "Food Intake",
    "Food",
    "FI",
]
CHEW_COUNT_COLUMN_CANDIDATES = ["Chew Count"]
CHEW_BOUT_COLUMN_CANDIDATES = ["Chew Bout"]
BITE_COLUMN_CANDIDATES = ["Bite"]
TIMECODE_COLUMN_CANDIDATES = ["bin_timecode"]
VIDEO_TIME_COLUMN_CANDIDATES = ["video_time"]

SENSOR_SAMPLE_RATE_HZ = 128.0
ANNOTATION_SAMPLE_RATE_HZ = 10.0


@dataclass
class PacketRecord:
    data_timestamp: int
    data_duration: float
    data_sampling_frequency: float
    data_ndarray_pickle: bytes


def load_config(config_path: str = "jitai_config.ini") -> configparser.ConfigParser:
    config = configparser.ConfigParser(allow_no_value=True)
    if len(config.read(config_path)) == 0:
        raise FileNotFoundError(f"Could not open {config_path}")
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
        except mysql.connector.Error:
            pass

    raise ConnectionError("Could not connect to the raw database using config host or localhost fallbacks.")


def choose_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Could not find any of these columns: {list(candidates)}")
    return None


def get_participant_list(conn) -> List[str]:
    query = """
        SELECT DISTINCT participantID
        FROM numeric_data
        WHERE participantID IS NOT NULL
        ORDER BY participantID
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]


def select_participant(conn, participant_id: Optional[str]) -> str:
    participants = get_participant_list(conn)
    if not participants:
        raise ValueError("No participants found in numeric_data.")

    if participant_id is not None:
        if participant_id not in participants:
            raise ValueError(f"Participant {participant_id} not found. Available: {participants}")
        return participant_id

    print("Participants found in numeric_data:")
    for i, pid in enumerate(participants, start=1):
        print(f"{i}. {pid}")

    while True:
        raw = input("Select participant number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(participants):
                return participants[idx - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


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

    return arr


def load_packet_array(packet: PacketRecord) -> np.ndarray:
    arr = pickle.loads(packet.data_ndarray_pickle)
    arr = to_n_by_4(arr)

    if packet.data_sampling_frequency > 0:
        expected_len = int(round(packet.data_duration / packet.data_sampling_frequency))
    else:
        expected_len = int(round(packet.data_duration * SENSOR_SAMPLE_RATE_HZ))

    if arr.shape[0] < expected_len:
        pad_len = expected_len - arr.shape[0]
        pad_value = arr[-1:, :] if arr.shape[0] > 0 else np.full((1, 4), -1)
        arr = np.vstack([arr, np.repeat(pad_value, pad_len, axis=0)])
    elif arr.shape[0] > expected_len:
        arr = arr[:expected_len, :]

    return arr


def get_sensor_span(packets: Sequence[PacketRecord]) -> Tuple[datetime, datetime]:
    first_packet_start_ts = (packets[0].data_timestamp + SENSOR_TIME_OFFSET_SECONDS) - float(packets[0].data_duration)
    last_packet_end_ts = packets[-1].data_timestamp + SENSOR_TIME_OFFSET_SECONDS
    return datetime.fromtimestamp(first_packet_start_ts), datetime.fromtimestamp(last_packet_end_ts)
#bottom of things that can be replaced with access to the python code

def parse_timecode_to_time(text: str) -> datetime.time:
    cleaned = str(text).strip().replace("'", "")
    if not cleaned:
        raise ValueError("Empty timecode")
    try:
        return datetime.strptime(cleaned, "%H:%M:%S.%f").time()
    except ValueError:
        return datetime.strptime(cleaned, "%H:%M:%S").time()


def parse_video_time_to_seconds(text: str) -> float:
    cleaned = str(text).strip()
    dt = datetime.strptime(cleaned, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000.0


def build_annotation_datetimes(df: pd.DataFrame, base_date: datetime.date) -> pd.Series:
    timecode_col = choose_column(df, TIMECODE_COLUMN_CANDIDATES, required=False)
    if timecode_col is not None:
        abs_times = []
        day_offset = 0
        prev_seconds = None
        for raw in df[timecode_col]:
            t = parse_timecode_to_time(raw)
            sec = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000.0
            if prev_seconds is not None and sec < prev_seconds:
                day_offset += 1
            prev_seconds = sec
            abs_times.append(datetime.combine(base_date, t) + timedelta(days=day_offset))
        return pd.Series(abs_times, index=df.index)

    video_time_col = choose_column(df, VIDEO_TIME_COLUMN_CANDIDATES, required=True)
    first_zero = parse_video_time_to_seconds(df[video_time_col].iloc[0])
    start_dt = datetime.combine(base_date, datetime.min.time()) + timedelta(seconds=first_zero)
    return pd.Series(
        [start_dt + timedelta(seconds=parse_video_time_to_seconds(v) - first_zero) for v in df[video_time_col]],
        index=df.index,
    )


def shift_annotation_datetimes_to_best_overlap(times: pd.Series, sensor_start: datetime, sensor_end: datetime) -> pd.Series:
    best_times = times
    best_score = float("-inf")

    for day_shift in (-1, 0, 1):
        for hour_shift in range(-8, 9):
            shifted = times + timedelta(days=day_shift, hours=hour_shift)
            overlap_start = max(shifted.iloc[0], sensor_start)
            overlap_end = min(shifted.iloc[-1], sensor_end)
            overlap = max((overlap_end - overlap_start).total_seconds(), 0.0)

            center_distance = abs(
                (
                    (shifted.iloc[0] + (shifted.iloc[-1] - shifted.iloc[0]) / 2)
                    - (sensor_start + (sensor_end - sensor_start) / 2)
                ).total_seconds()
            )

            score = overlap - 1e-3 * center_distance
            if score > best_score:
                best_score = score
                best_times = shifted

    return best_times


def series_from_candidates(df: pd.DataFrame, candidates: Sequence[str], default_value: float = 0.0) -> pd.Series:
    col = choose_column(df, candidates, required=False)
    if col is None:
        return pd.Series(np.full(len(df), default_value), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default_value).astype(float)


def get_food_series(df: pd.DataFrame, chew_count: pd.Series, chew_bout: pd.Series, bite: pd.Series) -> pd.Series:
    food_col = choose_column(df, FOOD_COLUMN_CANDIDATES, required=False)
    if food_col is not None:
        return pd.to_numeric(df[food_col], errors="coerce").fillna(0.0).astype(float)

    return ((chew_count > 0) | (chew_bout > 0) | (bite > 0)).astype(float)


def load_annotation_data(annotation_csv_path: str, sensor_start: datetime, sensor_end: datetime):
    df = pd.read_csv(annotation_csv_path)

    chew_count = series_from_candidates(df, CHEW_COUNT_COLUMN_CANDIDATES)
    chew_bout = series_from_candidates(df, CHEW_BOUT_COLUMN_CANDIDATES)
    bite = series_from_candidates(df, BITE_COLUMN_CANDIDATES)
    food = get_food_series(df, chew_count, chew_bout, bite)

    times = build_annotation_datetimes(df, base_date=sensor_start.date())
    # times = shift_annotation_datetimes_to_best_overlap(times, sensor_start, sensor_end)
    # above should remain commented out. as of 4/13/26. it shifts the annotation time based on the sensors, causing delay issues.
    # annotations should not be shifted because it has the local timestamp, while the sensors are on GMT timestamp.

    return {
        "times": times,
        "food_series": food,
        "chew_bout_series": chew_bout,
        "chew_count_series": chew_count,
        "bite_series": bite,
        "annotation_start": times.iloc[0],
        "annotation_end": times.iloc[-1],
    }


def build_sensor_window_arrays(packets: Sequence[PacketRecord], window_start: datetime, window_end: datetime) -> np.ndarray:
    total_seconds = (window_end - window_start).total_seconds()
    total_samples = int(np.ceil(total_seconds * SENSOR_SAMPLE_RATE_HZ))
    sensor_full = np.full((total_samples, 4), -1.0, dtype=float)

    window_start_ts = window_start.timestamp()

    for packet in packets:
        packet_arr = load_packet_array(packet)

        packet_start_ts = (packet.data_timestamp + SENSOR_TIME_OFFSET_SECONDS) - float(packet.data_duration)
        packet_start_idx = int(round((packet_start_ts - window_start_ts) * SENSOR_SAMPLE_RATE_HZ))
        packet_end_idx = packet_start_idx + packet_arr.shape[0]

        dst_start = max(packet_start_idx, 0)
        dst_end = min(packet_end_idx, sensor_full.shape[0])
        if dst_start >= dst_end:
            continue

        src_start = dst_start - packet_start_idx
        src_end = src_start + (dst_end - dst_start)
        sensor_full[dst_start:dst_end, :] = packet_arr[src_start:src_end, :]

    return sensor_full


def build_annotation_window_arrays(annotation_data: dict, window_start: datetime, window_end: datetime):
    total_seconds = (window_end - window_start).total_seconds()
    total_samples = int(np.ceil(total_seconds * ANNOTATION_SAMPLE_RATE_HZ))

    chew_count_full = np.full(total_samples, -1.0, dtype=float)
    chew_bout_full = np.full(total_samples, -1.0, dtype=float)
    bite_full = np.full(total_samples, -1.0, dtype=float)
    food_full = np.full(total_samples, -1.0, dtype=float)

    times = annotation_data["times"]
    chew_count = annotation_data["chew_count_series"]
    chew_bout = annotation_data["chew_bout_series"]
    bite = annotation_data["bite_series"]
    food = annotation_data["food_series"]

    for idx, ts in enumerate(times):
        sample_idx = int(round((ts - window_start).total_seconds() * ANNOTATION_SAMPLE_RATE_HZ))
        if 0 <= sample_idx < total_samples:
            chew_count_full[sample_idx] = float(chew_count.iloc[idx])
            chew_bout_full[sample_idx] = float(chew_bout.iloc[idx])
            bite_full[sample_idx] = float(bite.iloc[idx])
            food_full[sample_idx] = float(food.iloc[idx])

    return {
        "food": food_full,
        "chew_bout": chew_bout_full,
        "chew_count": chew_count_full,
        "bite": bite_full,
        "annotation_start": annotation_data["annotation_start"],
        "annotation_end": annotation_data["annotation_end"],
    }


def make_plot(participant_id: str, window_start: datetime, window_end: datetime,
              sensor_data: np.ndarray, annotations: dict):
    sensor_x = [window_start + timedelta(seconds=i / SENSOR_SAMPLE_RATE_HZ) for i in range(sensor_data.shape[0])]
    anno_x = [window_start + timedelta(seconds=i / ANNOTATION_SAMPLE_RATE_HZ) for i in range(len(annotations["food"]))]

    fig, axes = plt.subplots(8, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(
        f"{participant_id}\n"
        f"Window: {window_start.strftime('%Y-%m-%d %H:%M:%S')} to {window_end.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Annotation span: {annotations['annotation_start'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} to "
        f"{annotations['annotation_end'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    )

    axes[0].plot(sensor_x, sensor_data[:, 0], color="brown", linewidth=0.9)
    axes[0].set_ylabel("Acc X")
    axes[0].legend(["Acc X"], loc="upper right")

    axes[1].plot(sensor_x, sensor_data[:, 1], color="blue", linewidth=0.9)
    axes[1].set_ylabel("Acc Y")
    axes[1].legend(["Acc Y"], loc="upper right")

    axes[2].plot(sensor_x, sensor_data[:, 2], color="green", linewidth=0.9)
    axes[2].set_ylabel("Acc Z")
    axes[2].legend(["Acc Z"], loc="upper right")

    axes[3].plot(sensor_x, sensor_data[:, 3], color="orange", linewidth=0.9)
    axes[3].set_ylabel("Optical")
    axes[3].legend(["Optical"], loc="upper right")

    axes[4].step(anno_x, annotations["food"], where="post", color="red", linewidth=1.0)
    axes[4].set_ylabel("Food")
    axes[4].legend(["Food intake"], loc="upper right")
    axes[4].set_ylim(-1.2, 1.2)

    axes[5].step(anno_x, annotations["chew_bout"], where="post", color="purple", linewidth=1.0)
    axes[5].set_ylabel("Bout")
    axes[5].legend(["Chew bout"], loc="upper right")
    axes[5].set_ylim(-1.2, 1.2)

    axes[6].step(anno_x, annotations["chew_count"], where="post", color="darkgreen", linewidth=1.0)
    axes[6].set_ylabel("Count")
    axes[6].legend(["Chew count"], loc="upper right")
    axes[6].set_ylim(-1.2, 1.2)

    axes[7].step(anno_x, annotations["bite"], where="post", color="black", linewidth=1.0)
    axes[7].set_ylabel("Bite")
    axes[7].legend(["Bite"], loc="upper right")
    axes[7].set_ylim(-1.2, 1.2)

    locator = mdates.MinuteLocator(interval=30)
    formatter = mdates.DateFormatter("%H:%M")

    for ax in axes:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlim(window_start, window_end)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    axes[-1].tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("Clock time")

    fig.text(0.01, 0.01, "Missing data are filled with -1, without overwriting existing samples.", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig


def main():
    annotation_path = Path(ANNOTATION_CSV_PATH)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_path}")

    config = load_config()
    conn = connect_raw_db(config)
    try:
        participant_id = select_participant(conn, PARTICIPANT_ID)
        print(f"Selected participant: {participant_id}")

        packets = get_sensor_packets(conn, participant_id)
        sensor_start, sensor_end = get_sensor_span(packets)
        annotation_data = load_annotation_data(str(annotation_path), sensor_start, sensor_end)

        window_start = min(sensor_start, annotation_data["annotation_start"]) - timedelta(minutes=WINDOW_PADDING_MINUTES)
        window_end = max(sensor_end, annotation_data["annotation_end"]) + timedelta(minutes=WINDOW_PADDING_MINUTES)

        sensor_data = build_sensor_window_arrays(packets, window_start, window_end)
        annotations = build_annotation_window_arrays(annotation_data, window_start, window_end)

        fig = make_plot(participant_id, window_start, window_end, sensor_data, annotations)

        if SAVE_FIGURE:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"{participant_id}_sensor_vs_annotations.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {out_path.resolve()}")

        plt.show()
    finally:
        conn.close()


if __name__ == "__main__":
    main()

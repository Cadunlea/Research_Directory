r"""
probe_database.py - READ-ONLY inspection of the ETS databases.

WHY THIS EXISTS
---------------
Everything downstream (the drift check and both training scripts) reads its
data from the database rather than from annotation CSVs. Before writing code
against that data, this script reports what is ACTUALLY stored, so the rest is
built against facts instead of assumptions.

It answers, specifically:

  Q1  Where does the sensor signal live, and where does the ground truth live?
  Q2  What is inside a ground-truth record - array length, dtype, and above all
      WHICH VALUE fills the part of the day that was never annotated.
      -1 (not recorded) and 0 (recorded as not-eating) mean completely
      different things: -1 windows must be DROPPED, 0 windows must be KEPT as
      negatives. Getting this backwards either throws away most of the data or
      buries ~17 minutes of eating under 24 hours of fabricated negatives.
  Q3  Do CHGT / BOGT / BIGT correspond to Chew Count / Chew Bout / Bite?
  Q4  How many sessions have BOTH sensor data and ground truth - the real
      dataset size.
  Q5  Do raw and study rows share a studyID / record_hash, so the two can be
      joined, or must they be matched on participant plus overlapping time?

NOTHING IS WRITTEN. Every statement is a SELECT.

USAGE
-----
    python probe_database.py                        # 127.0.0.1:30071
    python probe_database.py --port 30071
    python probe_database.py --config jitai_config.ini   # read creds from ini
    python probe_database.py --dump-session AIM104611    # detail one participant

If you browse the database in a web browser you use port 30073 - that is
phpMyAdmin, the admin web page. Python connects to MariaDB itself on 30071.
Both numbers are correct, for different things.

Requires only mysql-connector-python and numpy, both already pinned in
requirements_ETS_full.txt.
"""

from __future__ import annotations

import argparse
import configparser
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mysql.connector
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit("mysql-connector-python is not installed in this interpreter.\n"
             "    pip install -r requirements_ETS_full.txt")


# --------------------------------------------------------------------------- #
# Defaults. The port is the HOST-published MariaDB port for this study:
# 30000 + study_number * 10 + 1, and study_ets is study_number 7 -> 30071.
# --------------------------------------------------------------------------- #
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30071
DEFAULT_USER = "jitai_admin"

# No password is stored in this file. This repository is on GitHub, and a
# credential committed to it stays in the history even after it is deleted.
# The password is read, in order of preference, from:
#   --config jitai_config.ini   (the file the rest of the project already uses)
#   the ETS_DB_PASSWORD environment variable
#   --password on the command line (visible in shell history; last resort)
PASSWORD_ENV_VAR = "ETS_DB_PASSWORD"

RAW_DB = "aim_raw_data"
STUDY_DB = "study_data"
PROCESSED_DB = "aim_processed_data"

SENSOR_IDENTIFIER = "XYZO"
GT_IDENTIFIERS = ("CHGT", "BOGT", "BIGT")

MISSING = -1


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def posix2datetime(timestamp: float) -> datetime:
    """The project's UTC/GMT convention, matching lib_jitai/posixtime.py.

    Reimplemented inline rather than imported so this script is self-contained
    and does not need lib_jitai (and therefore settings.py, isdocker, pytz) on
    sys.path. Deliberately NOT datetime.fromtimestamp(), which would apply the
    Windows local timezone and shift everything by 5-6 hours.
    """
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).replace(tzinfo=None)


def fmt_ts(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "-"
    return f"{posix2datetime(timestamp).isoformat(sep=' ')} ({int(timestamp)})"


def fmt_span_seconds(seconds: float) -> str:
    if seconds < 0:
        return f"{seconds:.0f}s"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def connect(host: str, port: int, user: str, password: str, database: str):
    return mysql.connector.connect(host=host, port=port, user=user,
                                   password=password, database=database)


def query(conn, sql: str, params: Sequence[Any] = ()) -> List[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchall()


def resolve_credentials(args) -> Tuple[str, str]:
    """Username and password, without either being stored in this file."""
    user, password = args.user, args.password
    if args.config:
        creds = credentials_from_config(args.config)
        user, password = creds["user"], creds["password"]
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


def credentials_from_config(path: str) -> Dict[str, Any]:
    """Read host/user/password out of a jitai_config.ini.

    The PORT in that file is the container-side 3306, which from the host
    belongs to a different study's database entirely, so it is deliberately
    ignored here in favour of the host-published port.
    """
    cfg = configparser.ConfigParser(allow_no_value=True)
    if not cfg.read(path):
        raise FileNotFoundError(f"could not open {path}")
    section = cfg["raw_database"]
    return {
        "user": section["db_user_name"],
        "password": section["passwd"],
    }


# --------------------------------------------------------------------------- #
# array unpacking
# --------------------------------------------------------------------------- #
def unpickle(blob: bytes) -> Optional[np.ndarray]:
    try:
        return np.asarray(pickle.loads(blob))
    except Exception as exc:                                    # noqa: BLE001
        print(f"      !! could not unpickle: {type(exc).__name__}: {exc}")
        return None


def describe_array(arr: Optional[np.ndarray], indent: str = "      ") -> None:
    if arr is None:
        return
    print(f"{indent}shape={arr.shape}  dtype={arr.dtype}  "
          f"size={arr.size:,}")
    flat = arr.ravel()
    finite = flat[np.isfinite(flat)] if np.issubdtype(arr.dtype, np.floating) else flat
    if finite.size:
        print(f"{indent}min={finite.min()}  max={finite.max()}  "
              f"mean={float(finite.mean()):.4f}")


def value_histogram(arr: np.ndarray, indent: str = "      ",
                    max_distinct: int = 12) -> None:
    """Exact value counts when the array is discrete - which is what tells us
    whether the un-annotated part of the day is -1 or 0."""
    values, counts = np.unique(arr, return_counts=True)
    total = arr.size
    if len(values) > max_distinct:
        print(f"{indent}{len(values):,} distinct values (continuous); "
              f"count of exactly -1: {int(np.sum(arr == MISSING)):,} "
              f"({100.0 * np.sum(arr == MISSING) / total:.2f}%)")
        return
    for value, count in zip(values, counts):
        share = 100.0 * count / total
        note = ""
        if value == MISSING:
            note = "   <- NOT RECORDED (must be dropped, never trained on)"
        elif value == 0:
            note = "   <- recorded as NOT eating"
        elif value == 1:
            note = "   <- recorded as EATING"
        print(f"{indent}value {value!s:>6}: {count:>10,}  ({share:6.2f}%){note}")


def annotated_span(arr: np.ndarray) -> Optional[Tuple[int, int]]:
    """First and last index that is not the -1 missing sentinel."""
    present = np.flatnonzero(arr != MISSING)
    if present.size == 0:
        return None
    return int(present[0]), int(present[-1])


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def report_server(conn) -> None:
    banner("SERVER")
    version = query(conn, "SELECT VERSION()")[0][0]
    print(f"  server version : {version}")
    databases = [row[0] for row in query(conn, "SHOW DATABASES")]
    print(f"  databases      : {', '.join(databases)}")
    for name in (RAW_DB, STUDY_DB, PROCESSED_DB):
        mark = "ok" if name in databases else "MISSING"
        print(f"    {name:<22} {mark}")


def report_table_shape(conn, database: str, table: str) -> List[str]:
    rows = query(conn,
                 "SELECT column_name, column_type FROM information_schema.columns "
                 "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
                 (database, table))
    if not rows:
        print(f"  {database}.{table}: TABLE NOT FOUND")
        return []
    print(f"  {database}.{table} columns:")
    for column, coltype in rows:
        print(f"    {column:<28} {coltype}")
    return [r[0] for r in rows]


def report_identifiers(conn, database: str) -> List[tuple]:
    """Every data_identifier in a numeric_data table, with row counts."""
    rows = query(conn,
                 f"SELECT data_identifier, COUNT(*), COUNT(DISTINCT participantID), "
                 f"COUNT(DISTINCT studyID) "
                 f"FROM {database}.numeric_data GROUP BY data_identifier "
                 f"ORDER BY data_identifier")
    print(f"\n  {database}.numeric_data, by data_identifier:")
    print(f"    {'ident':<8}{'rows':>10}{'participants':>14}{'studyIDs':>10}")
    for ident, count, participants, studies in rows:
        print(f"    {ident:<8}{count:>10,}{participants:>14}{studies:>10}")
    if not rows:
        print("    (table is empty)")
    return rows


def report_studies(conn, database: str) -> None:
    rows = query(conn,
                 f"SELECT studyID, COUNT(*), COUNT(DISTINCT participantID) "
                 f"FROM {database}.numeric_data GROUP BY studyID ORDER BY studyID")
    print(f"\n  {database}.numeric_data, by studyID:")
    for study, count, participants in rows:
        print(f"    {study:<20}{count:>10,} rows{participants:>6} participants")


def sample_record(conn, database: str, identifier: str) -> Optional[dict]:
    rows = query(conn,
                 f"SELECT studyID, participantID, deviceID, record_hash, "
                 f"data_timestamp, data_duration, data_sampling_frequency, "
                 f"data_processed_by_method, data_ndarray_pickle "
                 f"FROM {database}.numeric_data WHERE data_identifier=%s "
                 f"ORDER BY sequence_id LIMIT 1", (identifier,))
    if not rows:
        return None
    (study, participant, device, record_hash, timestamp, duration,
     frequency, method, blob) = rows[0]
    return {
        "studyID": study, "participantID": participant, "deviceID": device,
        "record_hash": record_hash, "data_timestamp": int(timestamp),
        "data_duration": duration, "data_sampling_frequency": frequency,
        "data_processed_by_method": method, "array": unpickle(blob),
    }


def report_ground_truth(conn) -> None:
    banner("Q2/Q3  GROUND TRUTH RECORDS  (study_data.numeric_data)")
    for identifier in GT_IDENTIFIERS:
        record = sample_record(conn, STUDY_DB, identifier)
        print(f"\n  --- {identifier} ---")
        if record is None:
            print("    no rows with this data_identifier")
            continue
        print(f"    studyID / participant : {record['studyID']} / "
              f"{record['participantID']}")
        print(f"    record_hash           : {record['record_hash']}")
        print(f"    data_timestamp        : {fmt_ts(record['data_timestamp'])}")
        print(f"    data_duration         : {record['data_duration']}")
        print(f"    sampling_frequency    : {record['data_sampling_frequency']}")
        print(f"    processed_by_method   : {record['data_processed_by_method']}")

        arr = record["array"]
        if arr is None:
            continue
        describe_array(arr, indent="    ")

        flat = arr.ravel()
        print("    value counts:")
        value_histogram(flat, indent="      ")

        span = annotated_span(flat)
        if span is None:
            print("      ** every sample is -1: this record annotates nothing **")
            continue
        first, last = span
        frequency = float(record["data_sampling_frequency"] or 10.0)
        start_ts = record["data_timestamp"] + first / frequency
        end_ts = record["data_timestamp"] + last / frequency
        print(f"    annotated (non -1) index range: {first:,} .. {last:,}")
        print(f"      -> {fmt_ts(start_ts)}")
        print(f"      -> {fmt_ts(end_ts)}")
        print(f"      -> span {fmt_span_seconds((last - first) / frequency)}, "
              f"{100.0 * (last - first + 1) / flat.size:.2f}% of the record")

        inside = flat[first:last + 1]
        outside_count = flat.size - inside.size
        outside_missing = int(np.sum(flat[:first] == MISSING) +
                              np.sum(flat[last + 1:] == MISSING))
        print(f"    OUTSIDE the annotated span: {outside_count:,} samples, "
              f"{outside_missing:,} of them are -1")
        if outside_count and outside_missing == outside_count:
            print("      ANSWER: un-annotated time is -1. Those windows will be "
                  "DROPPED.")
        elif outside_count:
            print("      WARNING: un-annotated time is NOT all -1. Some of it is "
                  "stored as a real value, which would become fabricated "
                  "negatives. Report this back.")


def report_sensor(conn) -> None:
    banner("Q1  SENSOR PACKETS  (aim_raw_data.numeric_data, XYZO)")
    record = sample_record(conn, RAW_DB, SENSOR_IDENTIFIER)
    if record is None:
        print("  no XYZO rows found in aim_raw_data.numeric_data")
        return
    print(f"  studyID / participant : {record['studyID']} / "
          f"{record['participantID']}")
    print(f"  deviceID              : {record['deviceID']}")
    print(f"  record_hash           : {record['record_hash']}")
    print(f"  data_timestamp        : {fmt_ts(record['data_timestamp'])}")
    print(f"  data_duration         : {record['data_duration']}")
    print(f"  sampling_frequency    : {record['data_sampling_frequency']}")
    arr = record["array"]
    describe_array(arr, indent="  ")
    if arr is None:
        return
    # Channel order matters: the plotting scripts treat the packet as 4 x N and
    # read the optical channel as row 3, so confirm the orientation here.
    if arr.ndim == 2:
        rows, cols = arr.shape
        print(f"  orientation           : {rows} x {cols}"
              f"  -> {'4 channels x N samples' if rows == 4 else 'N samples x 4 channels' if cols == 4 else 'UNEXPECTED'}")
        channels = arr if rows == 4 else arr.T
        names = ["Acc X", "Acc Y", "Acc Z", "Optical"]
        for index in range(min(4, channels.shape[0])):
            channel = channels[index]
            print(f"    {names[index]:<8} min={channel.min():>10} "
                  f"max={channel.max():>10} mean={float(channel.mean()):>12.2f} "
                  f"n_missing(-1)={int(np.sum(channel == MISSING)):,}")

    # Packet cadence: the training code assumes contiguous 8 s packets.
    steps = query(conn,
                  f"SELECT data_timestamp FROM {RAW_DB}.numeric_data "
                  f"WHERE data_identifier=%s AND participantID=%s "
                  f"ORDER BY data_timestamp LIMIT 200",
                  (SENSOR_IDENTIFIER, record["participantID"]))
    timestamps = [int(r[0]) for r in steps]
    if len(timestamps) > 1:
        deltas = Counter(b - a for a, b in zip(timestamps, timestamps[1:]))
        common = ", ".join(f"{delta}s x{count}"
                           for delta, count in deltas.most_common(5))
        print(f"  packet time steps (first {len(timestamps)}): {common}")


def report_join(conn) -> None:
    """Q4/Q5 - how sensor and ground truth line up, per session."""
    banner("Q4/Q5  SESSION INVENTORY  (sensor + ground truth overlap)")

    gt_rows = query(conn,
                    f"SELECT studyID, participantID, record_hash, data_identifier, "
                    f"data_timestamp, data_duration, data_sampling_frequency "
                    f"FROM {STUDY_DB}.numeric_data "
                    f"WHERE data_identifier IN ('CHGT','BOGT','BIGT') "
                    f"ORDER BY participantID, studyID, data_identifier")
    if not gt_rows:
        print("  no ground-truth rows found")
        return

    sessions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for study, participant, record_hash, identifier, timestamp, _dur, _freq in gt_rows:
        key = (str(participant), str(study))
        entry = sessions.setdefault(key, {"identifiers": set(),
                                          "record_hash": record_hash,
                                          "data_timestamp": int(timestamp)})
        entry["identifiers"].add(identifier)

    print(f"  ground-truth sessions (participant x studyID): {len(sessions)}")
    participants = sorted({p for p, _ in sessions})
    print(f"  distinct participants with ground truth     : {len(participants)}")

    # Do the raw rows carry the same studyID and record_hash?
    raw_pairs = query(conn,
                      f"SELECT DISTINCT studyID, participantID, record_hash "
                      f"FROM {RAW_DB}.numeric_data WHERE data_identifier=%s",
                      (SENSOR_IDENTIFIER,))
    raw_keys = {(str(p), str(s)) for s, p, _h in raw_pairs}
    raw_hashes = {(str(p), str(s)): h for s, p, h in raw_pairs}
    raw_participants = {p for p, _s in raw_keys}

    print(f"  sensor sessions (participant x studyID)     : {len(raw_keys)}")
    print(f"  distinct participants with sensor data      : "
          f"{len(raw_participants)}")

    shared_keys = set(sessions) & raw_keys
    print(f"\n  sessions matching on (participantID, studyID): {len(shared_keys)}")
    if shared_keys:
        hash_matches = sum(1 for key in shared_keys
                           if raw_hashes.get(key) == sessions[key]["record_hash"])
        print(f"    of those, record_hash also matches        : {hash_matches}")
        if hash_matches == len(shared_keys):
            print("    ANSWER: raw and study share studyID AND record_hash; "
                  "join on record_hash.")
        else:
            print("    NOTE: studyID matches but record_hash does not for some "
                  "sessions; join on (participantID, studyID) instead.")
    else:
        print("    ANSWER: no (participantID, studyID) pair is in both tables.")
        print("    The studyID differs between raw and study; sessions must be "
              "matched on participantID plus overlapping time.")
        shared_participants = {p for p, _ in sessions} & raw_participants
        print(f"    participants present in both tables: "
              f"{len(shared_participants)}")

    # Per-session detail. The question that decides whether a session is usable
    # is not "does this participant have sensor data somewhere" but "are there
    # sensor packets DURING the annotated window". A participant can have
    # thousands of packets and still none that overlap the video.
    print(f"\n  {'participant':<12}{'study':<14}{'annotated window (UTC)':<38}"
          f"{'mins':>6}{'pkts in window':>15}{'cover':>7}  status")
    usable = 0
    for (participant, study), entry in sorted(sessions.items()):
        span = annotated_window(conn, participant, study)
        if span is None:
            print(f"  {participant:<12}{study:<14}"
                  f"{'annotation is entirely -1':<38}{'':>6}{'':>15}{'':>7}  UNUSABLE")
            continue
        start_ts, end_ts, eating_share = span
        minutes = (end_ts - start_ts) / 60.0

        count = query(conn,
                      f"SELECT COUNT(*) FROM {RAW_DB}.numeric_data "
                      f"WHERE data_identifier=%s AND participantID=%s "
                      f"AND data_timestamp BETWEEN %s AND %s",
                      (SENSOR_IDENTIFIER, participant,
                       int(start_ts) - 16, int(end_ts) + 16))[0][0]
        expected = max(1, int((end_ts - start_ts) / 8))
        coverage = count / expected
        if count == 0:
            status = "NO SENSOR DATA IN WINDOW"
        elif coverage < 0.5:
            status = "partial"
        else:
            status = "ok"
            usable += 1

        window = (f"{posix2datetime(start_ts).strftime('%Y-%m-%d %H:%M')} .. "
                  f"{posix2datetime(end_ts).strftime('%Y-%m-%d %H:%M')}")
        print(f"  {participant:<12}{study:<14}{window:<38}{minutes:>6.0f}"
              f"{count:>15,}{coverage:>6.0%}  {status}")

    print(f"\n  USABLE SESSIONS (sensor data covering the annotation): "
          f"{usable} of {len(sessions)}")
    if usable < len(sessions):
        print("  Sessions marked NO SENSOR DATA IN WINDOW have ground truth whose")
        print("  day does not line up with any sensor packets. Check the upload")
        print("  before dismissing them - the annotation date may be wrong.")


def annotated_window(conn, participant: str,
                     study: str) -> Optional[Tuple[float, float, float]]:
    """(start_ts, end_ts, eating_fraction) of the annotated video for a session.

    Eating is 'chew bout OR bite' - a bout is sustained chewing and a bite is
    food entering the mouth. Chew count is excluded: it is a rate, not a state.
    """
    rows = query(conn,
                 f"SELECT data_identifier, data_timestamp, data_sampling_frequency, "
                 f"data_ndarray_pickle FROM {STUDY_DB}.numeric_data "
                 f"WHERE participantID=%s AND studyID=%s "
                 f"AND data_identifier IN ('BOGT','BIGT')",
                 (participant, study))
    arrays, day_start, frequency = {}, None, 10.0
    for identifier, timestamp, freq, blob in rows:
        array = unpickle(blob)
        if array is None:
            continue
        arrays[identifier] = array.ravel()
        day_start, frequency = int(timestamp), float(freq or 10.0)
    if "BOGT" not in arrays or "BIGT" not in arrays or day_start is None:
        return None

    bout, bite = arrays["BOGT"], arrays["BIGT"]
    length = min(len(bout), len(bite))
    bout, bite = bout[:length], bite[:length]
    known = (bout != MISSING) | (bite != MISSING)
    present = np.flatnonzero(known)
    if present.size == 0:
        return None
    first, last = int(present[0]), int(present[-1])
    eating = ((bout > 0) | (bite > 0))[first:last + 1]
    return (day_start + first / frequency,
            day_start + (last + 1) / frequency,
            float(eating.mean()))


def dump_session(conn, participant: str) -> None:
    """Every ground-truth record for one participant, in full detail, with the
    overlapping sensor coverage. Use this once the summary raises a question."""
    banner(f"DETAIL  {participant}")
    rows = query(conn,
                 f"SELECT studyID, data_identifier, data_timestamp, data_duration, "
                 f"data_sampling_frequency, data_ndarray_pickle "
                 f"FROM {STUDY_DB}.numeric_data "
                 f"WHERE participantID=%s AND data_identifier IN "
                 f"('CHGT','BOGT','BIGT') ORDER BY studyID, data_identifier",
                 (participant,))
    if not rows:
        print("  no ground-truth rows for this participant")
        return

    for study, identifier, timestamp, duration, frequency, blob in rows:
        arr = unpickle(blob)
        if arr is None:
            continue
        flat = arr.ravel()
        span = annotated_span(flat)
        print(f"\n  {study} / {identifier}")
        print(f"    day start   : {fmt_ts(timestamp)}")
        print(f"    samples     : {flat.size:,} @ {frequency} Hz "
              f"(duration field says {duration})")
        if span is None:
            print("    ** all -1 **")
            continue
        first, last = span
        freq = float(frequency or 10.0)
        inside = flat[first:last + 1]
        print(f"    annotated   : {fmt_ts(timestamp + first / freq)} .. "
              f"{fmt_ts(timestamp + last / freq)}"
              f"  ({fmt_span_seconds((last - first) / freq)})")
        print("    inside the annotated span:")
        value_histogram(inside, indent="      ")

        packets = query(conn,
                        f"SELECT COUNT(*) FROM {RAW_DB}.numeric_data "
                        f"WHERE data_identifier=%s AND participantID=%s "
                        f"AND data_timestamp BETWEEN %s AND %s",
                        (SENSOR_IDENTIFIER, participant,
                         int(timestamp + first / freq) - 16,
                         int(timestamp + last / freq) + 16))
        covering = packets[0][0]
        expected = int((last - first) / freq / 8) + 1
        print(f"    sensor packets covering that span: {covering:,} "
              f"(about {expected:,} expected at 8 s each)")
        if covering == 0:
            print("      ** NO SENSOR DATA over the annotated window - this "
                  "session cannot be used **")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only probe of the ETS databases.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="host-published MariaDB port (default 30071; "
                             "30073 is phpMyAdmin, not the database)")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=None,
                        help=f"database password; prefer --config or the "
                             f"{PASSWORD_ENV_VAR} environment variable")
    parser.add_argument("--config", default=None,
                        help="read user/password from a jitai_config.ini "
                             "(its port is container-side and is ignored)")
    parser.add_argument("--dump-session", default=None, metavar="AIMxxxxxx",
                        help="print full detail for one participant")
    args = parser.parse_args()

    user, password = resolve_credentials(args)

    print(f"connecting to {args.host}:{args.port} as {user}")
    try:
        conn = connect(args.host, args.port, user, password, STUDY_DB)
    except mysql.connector.Error as exc:
        print(f"\nCONNECTION FAILED: {exc}")
        print("\n  - is the study_ets database container running?")
        print("  - 30071 is MariaDB; 30073 is the phpMyAdmin web page")
        print("  - from WSL, 127.0.0.1 reaches ports Docker published on Windows")
        return 1

    try:
        report_server(conn)

        banner("TABLE SHAPES")
        report_table_shape(conn, RAW_DB, "numeric_data")
        print()
        report_table_shape(conn, STUDY_DB, "numeric_data")

        banner("WHAT IS STORED")
        report_identifiers(conn, RAW_DB)
        report_studies(conn, RAW_DB)
        report_identifiers(conn, STUDY_DB)
        report_studies(conn, STUDY_DB)

        report_sensor(conn)
        report_ground_truth(conn)
        report_join(conn)

        if args.dump_session:
            dump_session(conn, args.dump_session)

        banner("DONE")
        print("  Nothing was written. Paste this whole output back.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

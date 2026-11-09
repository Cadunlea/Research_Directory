r"""
write_results_to_database.py - store the model's output in the study database.

The last step of inference. process_database.py writes, for every recording,
the model's decision on each 8 s window (database_results\sessions\*_windows.csv).
This stores that output in study_data.numeric_data, the same table that holds
the annotations, in the SAME FORMAT as the annotations (CHGT/BOGT/BIGT), so any
lab code that reads an annotation can read the model's output unchanged:

    one row per recording per day per record type
    data_timestamp            midnight (same convention as the annotation rows)
    data_duration             864000   (a sample count, as in the annotation rows)
    data_sampling_frequency   10.0
    data_ndarray_pickle       pickled int16 array, 864000 samples (a day at 10 Hz)
                              -1 wherever the model did not score (no sensor
                              data, or outside the recording), as in the
                              annotations

Three new record types (data_identifier is CHAR(4), like CHGT/BOGT/BIGT/XYZO),
none of which exists yet:

    EATD   model decision: 1 eating, 0 not eating          (compare with BOGT/BIGT)
    EATP   model probability of eating x 1000 (0 to 1000)
    CHRT   chewing rate from the optical sensor, chews per minute x 10,
             in windows the model calls eating; 0 in windows it does not

Each 8 s window's value fills its 80 samples. Nothing already in the table is
read for writing, changed or deleted.

SAFE BY DEFAULT
  * Dry run unless --write: it connects, reads the table's columns, checks the
    record types are unused and fit the column, builds every array, checks the
    minutes of eating against sessions.csv, and prints what it would insert.
  * Insert only: no UPDATE or DELETE anywhere in this script.
  * Refuses if any row with one of these data_identifiers already exists.
  * All rows go in one transaction: on any error nothing is kept.
  * data_processed_by_method (if the table has it) records the model and
    threshold that produced the rows.

USAGE
-----
    python write_results_to_database.py --config "%CFG%"            (dry run)
    python write_results_to_database.py --config "%CFG%" --write
    python write_results_to_database.py --config "%CFG%" --identifiers EAD2 EAP2 CHR2

process_database.py runs the same step at the end with --database dry-run or
--database write, so one command does the whole inference.

To find the rows afterwards:
    SELECT participantID, studyID, data_identifier, data_timestamp
    FROM study_data.numeric_data WHERE data_identifier IN ('EATD','EATP','CHRT');
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ets_data

TABLE = f"{ets_data.STUDY_DB}.numeric_data"
FS = 10.0                                   # samples per second, as the annotations
DAY = 86400
DAY_SAMPLES = int(DAY * FS)                 # 864000
IDENTIFIERS = ("EATD", "EATP", "CHRT")
NOT_SCORED = -1
REQUIRED = ("participantID", "studyID", "data_identifier", "data_timestamp",
            "data_duration", "data_sampling_frequency", "data_ndarray_pickle")


def posix(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def day_arrays(windows, day_offset: int = 0):
    """Turn one recording's windows into whole-day 10 Hz arrays.

    windows: iterable of (start, end, probability, eating, rate), posix seconds;
    probability NaN means the window was not scored. day_offset: where midnight
    falls in posix seconds modulo 86400 (0 for UTC midnight).
    Returns {day_start: (decision, probability x 1000, rate x 10)} as int16."""
    days = {}
    for start, end, probability, eating, rate in windows:
        if not np.isfinite(probability):
            continue
        values = (int(bool(eating)), int(round(probability * 1000)),
                  (int(round(rate * 10)) if np.isfinite(rate) else NOT_SCORED)
                  if eating else 0)
        a = int(round((start - day_offset) * FS))
        b = int(round((end - day_offset) * FS))
        while a < b:                                # a window may cross midnight
            day = a // DAY_SAMPLES
            stop = min(b, (day + 1) * DAY_SAMPLES)
            arrays = days.setdefault(day * DAY + day_offset, tuple(
                np.full(DAY_SAMPLES, NOT_SCORED, dtype=np.int16) for _ in range(3)))
            for array, value in zip(arrays, values):
                array[a - day * DAY_SAMPLES:stop - day * DAY_SAMPLES] = value
            a = stop
    return days


def read_windows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        for w in csv.DictReader(handle):
            yield (posix(w["start_utc"]), posix(w["end_utc"]), number(w["probability"]),
                   w["eating"] == "1", number(w.get("chews_per_min_optical")))


def text(value) -> str:
    """DESCRIBE can return bytes, depending on the connector version."""
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def column_width(sql_type) -> int | None:
    m = re.search(r"char\((\d+)\)", text(sql_type).lower())
    return int(m.group(1)) if m else None


def store(args, results: Path, identifiers=IDENTIFIERS, write: bool = False) -> int:
    """Check the table, build the arrays and insert them (or only report, when
    write is False). Returns the number of rows inserted."""
    identifiers = tuple(identifiers)
    if len(identifiers) != 3 or len(set(identifiers)) != 3:
        raise SystemExit("give three different identifiers: decision, probability, rate")
    with open(results / "sessions.csv", newline="", encoding="utf-8") as handle:
        sessions = [r for r in csv.DictReader(handle) if r.get("status") == "ok"]
    model = {}
    if (results / "model.json").exists():
        with open(results / "model.json", encoding="utf-8") as handle:
            model = json.load(handle)
    method = (f"food_intake_final model trained {model.get('created_utc', '?')}, "
              f"threshold {float(model.get('threshold', 'nan')):.3f}, 8 s windows")
    print(f"{len(sessions)} recordings in {results / 'sessions.csv'}; {method}")

    conn = ets_data.connect(args)
    try:
        described = {text(d[0]): d for d in ets_data.query(conn, f"DESCRIBE {TABLE}")}
        missing = [c for c in REQUIRED if c not in described]
        if missing:
            raise SystemExit(f"{TABLE} has no column(s) {missing}; columns are "
                             f"{list(described)}. Nothing written.")
        width = column_width(described["data_identifier"][1])
        if width and max(len(i) for i in identifiers) > width:
            raise SystemExit(f"data_identifier holds {width} characters; "
                             f"{identifiers} do not fit. Pass shorter ones with --identifiers.")

        columns = list(REQUIRED)
        optional = {}
        if "data_processed_by_method" in described:
            w = column_width(described["data_processed_by_method"][1])
            optional["data_processed_by_method"] = method[:w] if w else method
        if "deviceID" in described:
            optional["deviceID"] = None                 # filled per recording below
        hash_needed = ("record_hash" in described and text(described["record_hash"][2]).upper() == "NO"
                       and described["record_hash"][4] is None)
        if hash_needed:
            w = column_width(described["record_hash"][1])
            kind = text(described["record_hash"][1]).lower()
            if not ("char" in kind or "text" in kind) or (w and w < 32):
                raise SystemExit(f"record_hash is required and is {described['record_hash'][1]}; "
                                 f"ask Siavash what it should hold. Nothing written.")
            optional["record_hash"] = None
        columns += list(optional)
        left_out = [c for c, d in described.items()
                    if c not in columns and text(d[2]).upper() == "NO" and d[4] is None
                    and "auto_increment" not in text(d[5]).lower()]
        if left_out:
            raise SystemExit(f"{TABLE} requires values for {left_out}, which this script "
                             f"does not write. Ask Siavash what they should hold.")

        existing = ets_data.query(
            conn, f"SELECT data_identifier, COUNT(*) FROM {TABLE} WHERE data_identifier IN "
                  f"({','.join(['%s'] * 3)}) GROUP BY data_identifier", identifiers)
        if existing:
            raise SystemExit(f"rows already exist: {dict(existing)}; refusing to add more. "
                             f"Use new --identifiers.")
        midnight = ets_data.query(conn, f"SELECT data_timestamp FROM {TABLE} "
                                        f"WHERE data_identifier='BOGT' LIMIT 1")
        offset = int(midnight[0][0]) % DAY if midnight else 0
        print(f"\n{TABLE}: columns OK; {', '.join(identifiers)} unused; "
              f"midnight at posix % 86400 = {offset} (from the annotation rows)")

        def recording_rows(s):
            """The rows for one recording, built when needed (each is ~1.7 MB)."""
            participant, study = s["participant"], s["study"]
            path = results / "sessions" / f"{participant}_{study}_windows.csv"
            if not path.exists():
                raise SystemExit(f"{path} is missing; rerun process_database.py")
            days = day_arrays(read_windows(path), offset)
            device = None
            if "deviceID" in optional:
                found = ets_data.query(
                    conn, f"SELECT deviceID FROM {ets_data.RAW_DB}.numeric_data "
                          f"WHERE data_identifier=%s AND participantID=%s AND studyID=%s LIMIT 1",
                    (ets_data.SENSOR_IDENTIFIER, participant, study))
                device = found[0][0] if found else None
            minutes = sum(int(np.sum(a[0] == 1)) for a in days.values()) / FS / 60
            rows = []
            for day, arrays in sorted(days.items()):
                for identifier, array in zip(identifiers, arrays):
                    extra = []
                    for c in optional:
                        if c == "deviceID":
                            extra.append(device)
                        elif c == "record_hash":
                            extra.append(hashlib.md5(f"{study}|{participant}|{identifier}|{day}"
                                                     .encode()).hexdigest())
                        else:
                            extra.append(optional[c])
                    rows.append((participant, study, identifier, int(day), DAY_SAMPLES, FS,
                                 pickle.dumps(array, protocol=4), *extra))
            return rows, minutes

        checked, n_rows, size, largest = [], 0, 0, 0
        for s in sessions:
            rows, minutes = recording_rows(s)
            checked.append((s["participant"], s["study"], minutes, number(s.get("eating_min"))))
            n_rows += len(rows)
            size += sum(len(r[6]) for r in rows)
            largest = max([largest] + [len(r[6]) for r in rows])
        off = [c for c in checked if np.isfinite(c[3]) and abs(c[2] - c[3]) > 0.15]
        print(f"eating minutes rebuilt from the arrays match sessions.csv for "
              f"{len(checked) - len(off)} of {len(checked)} recordings")
        for p, st, got, want in off[:10]:
            print(f"  !! {p}/{st}: {got:.1f} min in the arrays, {want:.1f} in sessions.csv")
        if off:
            raise SystemExit("arrays disagree with sessions.csv; nothing written")
        print(f"{n_rows} rows ({n_rows // 3} recording-days x 3 record types), "
              f"{size / 1e6:.0f} MB in all; columns: {', '.join(columns)}")
        packet = ets_data.query(conn, "SELECT @@max_allowed_packet")[0][0]
        if int(packet) < 4 * largest:
            print(f"  !! the server accepts {int(packet) / 1e6:.1f} MB per statement and a row "
                  f"is {largest / 1e6:.1f} MB (up to twice that once escaped). If the write "
                  f"fails with 'packet too large', ask Siavash to raise max_allowed_packet.")
        for p, st, got, _ in checked[:5]:
            print(f"  {p}/{st}: {got:.1f} min eating")
        if not write:
            print(f"\nDRY RUN: nothing written. To insert the {n_rows} rows, run again with "
                  f"--write (or process_database.py --database write).")
            return 0

        sql = (f"INSERT INTO {TABLE} ({', '.join(columns)}) "
               f"VALUES ({', '.join(['%s'] * len(columns))})")
        first = None
        conn.autocommit = False                         # one transaction for every row
        try:
            done = 0
            with conn.cursor() as cur:
                for s in sessions:
                    rows, _ = recording_rows(s)
                    for row in rows:                    # one row per statement
                        cur.execute(sql, row)
                    done += len(rows)
                    first = first or (rows[0] if rows else None)
                    print(f"  {s['participant']}/{s['study']}: {len(rows)} rows ({done}/{n_rows})")
            conn.commit()
        except Exception:
            conn.rollback()
            print("  error: rolled back, nothing kept")
            raise
        written = ets_data.query(
            conn, f"SELECT COUNT(*) FROM {TABLE} WHERE data_identifier IN "
                  f"({','.join(['%s'] * 3)})", identifiers)[0][0]
        back = ets_data.query(
            conn, f"SELECT data_ndarray_pickle FROM {TABLE} WHERE data_identifier=%s AND "
                  f"participantID=%s AND studyID=%s AND data_timestamp=%s",
            (identifiers[0], first[0], first[1], first[3]))
        same = bool(back) and np.array_equal(pickle.loads(back[0][0]), pickle.loads(first[6]))
        print(f"\ninserted {written} rows; first row read back {'identical' if same else 'DIFFERENT'}")
        return written
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--results", default="database_results",
                        help="folder written by process_database.py")
    parser.add_argument("--identifiers", nargs=3, default=list(IDENTIFIERS),
                        metavar=("DECISION", "PROBABILITY", "RATE"),
                        help="data_identifier for each record type (must be unused)")
    parser.add_argument("--write", action="store_true",
                        help="actually insert; without it nothing is written")
    args = parser.parse_args()
    store(args, Path(args.results), args.identifiers, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""
write_results_to_database.py - store detected eating episodes in the study database.

The last step of inference: process_database.py writes every detected eating
episode to database_results\all_episodes.csv; this stores them in
study_data.numeric_data as a NEW record type, so nothing already there is
touched. The format is a proposal until Dr. Sazonov and Siavash confirm it.

One row per episode, in the columns the annotations already use:

    participantID, studyID      the recording
    data_identifier             the new record type (default EATDET)
    data_timestamp              episode start (posix seconds, database clock)
    data_sampling_frequency     0 (an episode is not a sampled signal)
    data_ndarray_pickle         pickled float array:
                                [end, duration_s, windows, mean_probability,
                                 chews, chews_per_min, chews_per_min_optical]
                                (NaN where a value is missing)

SAFE BY DEFAULT
  * Dry run unless --write: it connects, reads the table's columns, checks
    the record type is unused, and prints what it would insert. Nothing is
    written.
  * Insert only: no UPDATE or DELETE anywhere in this script.
  * Refuses if any row with this data_identifier already exists.
  * All rows go in one transaction: on any error nothing is kept.
  * The model that produced the episodes (database_results\model.json) is
    printed, and the rows can be found again by their data_identifier.

USAGE
-----
    python write_results_to_database.py --config "%CFG%"            (dry run)
    python write_results_to_database.py --config "%CFG%" --write    (after the OK)
    python write_results_to_database.py --config "%CFG%" --identifier EATDET2
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ets_data

TABLE = f"{ets_data.STUDY_DB}.numeric_data"
COLUMNS = ("participantID", "studyID", "data_identifier", "data_timestamp",
           "data_sampling_frequency", "data_ndarray_pickle")
FIELDS = ("end", "duration_s", "windows", "mean_probability", "chews", "chews_per_min",
          "chews_per_min_optical")


def posix(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rows_from_csv(path: Path, identifier: str):
    with open(path, newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    rows = []
    for e in episodes:
        start = posix(e["start_utc"])
        values = np.array([posix(e["end_utc"])] + [number(e.get(k)) for k in FIELDS[1:]],
                          dtype=float)
        rows.append((e["participant"], e["study"], identifier, int(round(start)), 0.0,
                     pickle.dumps(values, protocol=4)))
    return episodes, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--results", default="database_results",
                        help="folder written by process_database.py")
    parser.add_argument("--identifier", default="EATDET",
                        help="data_identifier for the new rows (must be unused)")
    parser.add_argument("--write", action="store_true",
                        help="actually insert; without it nothing is written")
    args = parser.parse_args()

    results = Path(args.results)
    episodes, rows = rows_from_csv(results / "all_episodes.csv", args.identifier)
    model = {}
    if (results / "model.json").exists():
        with open(results / "model.json", encoding="utf-8") as handle:
            model = json.load(handle)
    recordings = {(r[0], r[1]) for r in rows}
    print(f"{len(rows)} episodes from {len(recordings)} recordings "
          f"({results / 'all_episodes.csv'}); model trained {model.get('created_utc', '?')}, "
          f"threshold {model.get('threshold', '?')}")

    conn = ets_data.connect(args)
    try:
        described = ets_data.query(conn, f"DESCRIBE {TABLE}")
        names = [str(d[0]) for d in described]
        missing = [c for c in COLUMNS if c not in names]
        if missing:
            raise SystemExit(f"{TABLE} has no column(s) {missing}; columns are {names}. "
                             f"The format needs agreeing before anything is written.")
        # Columns this script leaves out must have a default or be nullable.
        required = [str(d[0]) for d in described
                    if str(d[0]) not in COLUMNS and str(d[2]).upper() == "NO"
                    and d[4] is None and "auto_increment" not in str(d[5]).lower()]
        if required:
            raise SystemExit(f"{TABLE} requires values for {required}, which this script "
                             f"does not write. Ask Siavash what they should hold.")
        existing = ets_data.query(conn, f"SELECT COUNT(*) FROM {TABLE} WHERE data_identifier=%s",
                                  (args.identifier,))[0][0]
        if existing:
            raise SystemExit(f"{existing} rows with data_identifier '{args.identifier}' already "
                             f"exist; refusing to add more. Use a new --identifier.")

        print(f"\n{TABLE}: columns OK; data_identifier '{args.identifier}' is unused.")
        print("first rows that would be inserted:")
        for e in episodes[:5]:
            print(f"  {e['participant']}/{e['study']}  {e['start_utc']} to {e['end_utc']}  "
                  f"{e['duration_s']} s, chews {e.get('chews') or '-'}, "
                  f"optical rate {e.get('chews_per_min_optical') or '-'}")
        if not args.write:
            print(f"\nDRY RUN: nothing written. Add --write to insert {len(rows)} rows.")
            return 0

        sql = (f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) "
               f"VALUES ({', '.join(['%s'] * len(COLUMNS))})")
        try:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        written = ets_data.query(conn, f"SELECT COUNT(*) FROM {TABLE} WHERE data_identifier=%s",
                                 (args.identifier,))[0][0]
        print(f"\ninserted {written} rows with data_identifier '{args.identifier}'.")
        print(f"To find them: SELECT * FROM {TABLE} WHERE data_identifier='{args.identifier}';")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""
process_database.py - run the saved model on every recording in the database.

The batch form of detect_eating.py, and the "process this database" step:
finds every participant and study with AIM-2 sensor data, runs the final model
over each whole recording, and writes into --out (default database_results\):

    sessions\<participant>_<study>_windows.csv    per window, as detect_eating.py
    sessions\<participant>_<study>_episodes.csv   per episode, as detect_eating.py
    all_episodes.csv     every detected eating episode in the database, one row
                         each: participant, study, start, end, duration, mean
                         probability, chews, chewing rate. This is the table
                         meant to be written back to the database once its
                         place there is agreed.
    sessions.csv         one row per recording: hours recorded and scored,
                         minutes of eating, episodes, chews and chewing rate;
                         for annotated sessions also F1, precision, recall and
                         the annotated eating time and chews, and whether the
                         participant was used to train the model
    model.json           a copy of the model's record, so the results say
                         which model and threshold produced them

Recordings already in --out are skipped, so an interrupted run picks up where
it stopped; --overwrite redoes them. A recording that fails is reported and
the rest carry on. plot_meals.py draws the results.

USAGE
-----
    python process_database.py --config "%CFG%" --list
    python process_database.py --config "%CFG%"
    python process_database.py --config "%CFG%" --participants AIM146271 AIM147560
    python process_database.py --config "%CFG%" --studies ETS_lunch ETS_snack
    python process_database.py --config "%CFG%" --overwrite

Scores against annotations are optimistic for participants the model trained
on (in_training = 1). The fair estimate for new people is the
leave-one-subject-out result; held-out participants here are in_training = 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import ets_data
from detect_eating import (_SessionRef, annotation_scores, chew_scale, episode_rows,
                           episodes, load_model, predict, recording_range, utc,
                           write_outputs)

SESSION_FIELDS = ["participant", "study", "start_utc", "end_utc", "hours_recorded",
                  "hours_scored", "eating_min", "episodes", "chews", "chews_per_min",
                  "in_training", "annotated_windows", "f1", "precision", "recall",
                  "annotated_eating_min", "annotated_chews", "annotated_chews_per_min",
                  "model_chews_same_span", "model_chews_per_min_same_span", "status"]
EPISODE_FIELDS = ["participant", "study", "start_utc", "end_utc", "duration_s",
                  "windows", "mean_probability", "chews", "chews_per_min"]


def recordings(conn, participants=None, studies=None):
    """(participant, study) pairs that have sensor data."""
    sql = (f"SELECT DISTINCT participantID, studyID FROM {ets_data.RAW_DB}.numeric_data "
           f"WHERE data_identifier=%s")
    params = [ets_data.SENSOR_IDENTIFIER]
    for column, values in (("participantID", participants), ("studyID", studies)):
        if values:
            sql += f" AND {column} IN (" + ",".join(["%s"] * len(values)) + ")"
            params.extend(values)
    return sorted((str(p), str(s)) for p, s in ets_data.query(conn, sql, params))


def read_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process(conn, model, transform, record, config, scale, annotated, participant,
            study, prefix: Path, merge_gap: float, min_episode: float):
    threshold = float(record["threshold"])
    window = float(config.window_seconds)
    start, end = recording_range(conn, participant, study)
    starts, probability, chews, missing = predict(
        conn, model, transform, _SessionRef(participant=participant, study=study),
        start, end, window, scale)
    scored = np.isfinite(probability)
    eating = scored & (np.nan_to_num(probability) >= threshold)
    found = episodes(starts, eating, window, merge_gap, min_episode)
    write_outputs(prefix, starts, probability, chews, eating, missing, found, window)

    eating_min = eating.sum() * window / 60
    counted = chews[eating & np.isfinite(chews)]
    row = {"participant": participant, "study": study,
           "start_utc": utc(start), "end_utc": utc(end),
           "hours_recorded": f"{(end - start) / 3600:.2f}",
           "hours_scored": f"{scored.sum() * window / 3600:.2f}",
           "eating_min": f"{eating_min:.1f}", "episodes": len(found),
           "chews": f"{counted.sum():.0f}" if len(counted) else "",
           "chews_per_min": f"{counted.sum() / eating_min:.1f}"
           if len(counted) and eating_min else "",
           "in_training": int(participant in set(record.get("training_participants", []))),
           "status": "ok"}
    session = annotated.get((participant, study))
    if session is not None and session.annotated_range() is not None:
        m = annotation_scores(session, starts, probability, chews, threshold, window)
        if m is not None:
            row.update({"annotated_windows": m["annotated_windows"],
                        "f1": f"{m['f1']:.4f}", "precision": f"{m['precision']:.4f}",
                        "recall": f"{m['recall']:.4f}",
                        "annotated_eating_min": f"{m['annotated_eating_min']:.1f}",
                        "annotated_chews": f"{m['annotated_chews']:.0f}",
                        "annotated_chews_per_min":
                            f"{m['annotated_chews'] / m['annotated_eating_min']:.1f}"
                            if m["annotated_eating_min"] else ""})
            if "model_chews_same_span" in m:
                row["model_chews_same_span"] = f"{m['model_chews_same_span']:.0f}"
                if m["model_eating_min_same_span"]:
                    row["model_chews_per_min_same_span"] = (
                        f"{m['model_chews_same_span'] / m['model_eating_min_same_span']:.1f}")
    episode_list = [{"participant": participant, "study": study, **e}
                    for e in episode_rows(starts, probability, chews, eating, found, window)]
    return row, episode_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--model", default=str(Path("models") / "food_intake_final"))
    parser.add_argument("--out", default="database_results")
    parser.add_argument("--participants", nargs="*", help="only these participants")
    parser.add_argument("--studies", nargs="*", help="only these studies (e.g. ETS_lunch)")
    parser.add_argument("--merge-gap", type=float, default=32.0)
    parser.add_argument("--min-episode", type=float, default=0.0)
    parser.add_argument("--list", action="store_true",
                        help="only list the recordings that would be processed")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo recordings that already have results")
    args = parser.parse_args()

    model, config, record, transform = load_model(Path(args.model))
    scale = chew_scale(record, config)
    out = Path(args.out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    with open(out / "model.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, default=str)
    if scale is None:
        print("  chew counts unavailable (no chew output, or no window cache to scale it)")

    done = {} if args.overwrite else {
        (r["participant"], r["study"]): r for r in read_rows(out / "sessions.csv")
        if r.get("status") == "ok"}
    kept_episodes = [] if args.overwrite else [
        r for r in read_rows(out / "all_episodes.csv")
        if (r["participant"], r["study"]) in done]

    conn = ets_data.connect(args)
    try:
        todo = recordings(conn, args.participants, args.studies)
        if args.list:
            for participant, study in todo:
                start, end = recording_range(conn, participant, study)
                print(f"  {participant}/{study}: {utc(start)} to {utc(end)} UTC "
                      f"({(end - start) / 3600:.2f} h)")
            print(f"\n{len(todo)} recordings")
            return 0
        annotated = {(s.participant, s.study): s for s in ets_data.load_sessions(conn)}
        print(f"{len(todo)} recordings with sensor data, {len(annotated)} annotated; "
              f"{sum(k in done for k in todo)} already done; model {args.model}, "
              f"threshold {float(record['threshold']):.3f}\n")
        rows = dict(done)
        all_episodes = kept_episodes
        for i, (participant, study) in enumerate(todo, 1):
            if (participant, study) in done:
                continue
            began = time.time()
            try:
                row, found = process(conn, model, transform, record, config, scale,
                                     annotated, participant, study,
                                     out / "sessions" / f"{participant}_{study}",
                                     args.merge_gap, args.min_episode)
            except (Exception, SystemExit) as exc:                  # noqa: BLE001
                print(f"  [{i}/{len(todo)}] {participant}/{study}: FAILED "
                      f"({type(exc).__name__}: {exc})")
                rows[(participant, study)] = {"participant": participant, "study": study,
                                              "status": f"failed: {exc}"}
                continue
            rows[(participant, study)] = row
            all_episodes = [e for e in all_episodes
                            if (e["participant"], e["study"]) != (participant, study)]
            all_episodes.extend(found)
            score = f", F1 {float(row['f1']):.3f}" if row.get("f1") else ""
            print(f"  [{i}/{len(todo)}] {participant}/{study}: {row['hours_scored']} h scored, "
                  f"{row['eating_min']} min eating, {row['episodes']} episodes"
                  f"{score}  ({time.time() - began:.0f} s)")
            # Saved after every recording, so an interruption loses at most one.
            write_rows(out / "sessions.csv", SESSION_FIELDS,
                       [rows[k] for k in sorted(rows)])
            write_rows(out / "all_episodes.csv", EPISODE_FIELDS,
                       sorted(all_episodes, key=lambda e: (e["participant"], e["study"],
                                                           e["start_utc"])))
    finally:
        conn.close()

    write_rows(out / "sessions.csv", SESSION_FIELDS, [rows[k] for k in sorted(rows)])
    ok = [r for r in rows.values() if r.get("status") == "ok"]
    failed = [r for r in rows.values() if r.get("status", "ok") != "ok"]
    print(f"\n{len(ok)} recordings processed, {len(failed)} failed; "
          f"{len(all_episodes)} eating episodes in total")
    scored = [r for r in ok if r.get("f1")]
    for flag, name in ((0, "held out of training"), (1, "used in training (optimistic)")):
        group = [float(r["f1"]) for r in scored if int(r["in_training"]) == flag]
        if group:
            print(f"  annotated, {name}: {len(group)} sessions, median F1 {np.median(group):.3f}")
    print(f"\nwrote {out / 'sessions.csv'}\n      {out / 'all_episodes.csv'}\n"
          f"      {out / 'sessions'}\\ (per-recording windows and episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

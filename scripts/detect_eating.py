r"""
detect_eating.py - run the saved model on a recording and report eating episodes.

Loads the model folder written by train_final_model.py, reads the AIM-2
sensor data for one participant and study from the database, cuts it into
non-overlapping 8 s windows, and writes:

    <out>_windows.csv    one row per window: start and end (UTC), the model's
                         probability, the decision (1 eating), and the share
                         of missing sensor samples (windows more than half
                         missing are not scored)
    <out>_episodes.csv   one row per detected eating episode: start, end,
                         duration, number of windows, mean probability, and
                         the model's chew count and chewing rate (chews per
                         minute) over the episode's eating windows

Consecutive eating windows form an episode, and pauses of up to --merge-gap
seconds (default 32, four windows) are bridged, as in episode_metrics.py.
Episodes shorter than --min-episode seconds (default 0: keep all) are dropped.

The time range is the whole recording by default, or --start/--end in UTC.
With --compare, a session that has annotations is also scored against them
over its annotated span.

Chew counts come from the model's second output, trained on the annotated
chews (CHGT) and scaled back to chews per window. chew_check.py measures how
well they match the annotations; check that before reporting them.

USAGE
-----
    python detect_eating.py --config "%CFG%" --participant AIM142507 --study ETS_lunch
    python detect_eating.py --config "%CFG%" --participant AIM142507 --study ETS_lunch --compare
    python detect_eating.py --config "%CFG%" --participant AIM142507 --study ETS_lunch --start "2026-03-10 12:00:00" --end "2026-03-10 13:00:00"
    python detect_eating.py --config "%CFG%" --participant AIM142507 --study ETS_lunch --model models\food_intake_fa15

The model was trained on these participants; scoring one of them again shows
the tool working, not how well it generalises. That number is the
leave-one-subject-out result.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ets_data
import ets_eval
from episode_metrics import merge, runs_of

CHUNK_SECONDS = 3600.0       # read the sensor data an hour at a time


class _SessionRef(SimpleNamespace):
    """What ets_data.load_sensor needs to find a recording."""


def load_model(model_dir: Path):
    import train_food_intake as tfi
    with open(model_dir / "model.json", encoding="utf-8") as handle:
        record = json.load(handle)
    parser = tfi.build_parser()
    config = parser.parse_args([])
    for key, value in record["config"].items():
        setattr(config, key, value)
    config = tfi.normalise_config(config)
    model = tfi.ARCHITECTURES[config.architecture](tuple(record["input_shape"]), config)
    import warnings
    with warnings.catch_warnings():
        # The saved weights include the optimizer's state, which inference
        # does not need; Keras warns that it skips it.
        warnings.simplefilter("ignore", UserWarning)
        model.load_weights(str(model_dir / "model.weights.h5"))
    return model, config, record, tfi.make_transform(config)


def chew_scale(record, config):
    """Chews per window that the chew output's 1.0 stands for.

    Training divided the chew target by the 99th percentile of the annotated
    counts. model.json records it from now on; for a model saved before that,
    it is recomputed from the same window cache, which gives the same value.
    None when the model has no chew output or the cache is not here."""
    if float(config.chew_weight) <= 0:
        return None
    if record.get("chew_scale"):
        return float(record["chew_scale"])
    hop = config.window_seconds / 2.0 if config.overlap else config.window_seconds
    path = ets_data.cache_path(Path(config.cache_dir), config.window_seconds, hop)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return float(np.percentile(data["chews"], 99)) or 1.0


def parse_utc(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp()


def utc(timestamp: float) -> str:
    return ets_data.posix2datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def recording_range(conn, participant: str, study: str):
    rows = ets_data.query(
        conn, f"SELECT MIN(data_timestamp), MAX(data_timestamp) "
              f"FROM {ets_data.RAW_DB}.numeric_data WHERE data_identifier=%s "
              f"AND participantID=%s AND studyID=%s",
        (ets_data.SENSOR_IDENTIFIER, participant, study))
    if not rows or rows[0][0] is None:
        raise SystemExit(f"no sensor data for {participant}/{study}")
    return float(rows[0][0]), float(rows[0][1]) + ets_data.PACKET_SECONDS


def predict(conn, model, transform, ref, start: float, end: float,
            window: float, scale=None):
    """(starts, probabilities, chews, missing fractions) for every window in
    range. Chews are NaN without a chew scale, and wherever the window is not
    scored."""
    samples = int(round(window * ets_data.SENSOR_FS))
    starts, probabilities, chews, missing = [], [], [], []
    chunk_start = start
    while chunk_start + window <= end:
        chunk_end = min(chunk_start + CHUNK_SECONDS, end)
        chunk_end = chunk_start + np.floor((chunk_end - chunk_start) / window) * window
        block = ets_data.load_sensor(conn, ref, chunk_start, chunk_end)
        n = len(block) // samples
        if n == 0:
            break
        X = block[:n * samples].reshape(n, samples, 4)
        gap = np.mean(X == ets_data.MISSING, axis=(1, 2))
        usable = gap <= ets_data.MAX_MISSING_FRACTION
        p, c = np.full(n, np.nan), np.full(n, np.nan)
        if usable.any():
            output = model(transform(X[usable]), training=False)
            if isinstance(output, dict):
                if scale is not None:
                    c[usable] = np.asarray(output["chews"]).ravel() * scale
                output = output["food"]
            p[usable] = np.asarray(output).ravel()
        starts.extend(chunk_start + window * np.arange(n))
        probabilities.extend(p)
        chews.extend(c)
        missing.extend(gap)
        chunk_start = chunk_end
    return (np.asarray(starts), np.asarray(probabilities), np.asarray(chews),
            np.asarray(missing))


def episodes(starts, eating, window: float, merge_gap: float, min_seconds: float):
    breaks = np.abs(np.diff(starts) - window) > 0.5 * window
    spans = merge(runs_of(eating, breaks), starts, window, merge_gap, breaks)
    out = []
    for first, last in spans:
        duration = starts[last] + window - starts[first]
        if duration >= min_seconds:
            out.append((first, last, duration))
    return out


def window_labels(session, starts, window):
    """Per window: annotated label (1, 0, or -1 where under half is
    annotated) and the number of annotated chews."""
    eating = session.eating()
    labels, chews = [], []
    for s in starts:
        lo = int(round((s - session.day_start) * session.frequency))
        hi = lo + int(round(window * session.frequency))
        part = eating[max(lo, 0):max(hi, 0)]
        known = part[part != ets_data.MISSING]
        enough = len(part) and len(known) >= ets_data.MIN_LABEL_COVERAGE * len(part)
        labels.append(int(known.mean() > ets_data.EATING_THRESHOLD) if enough else -1)
        chews.append(float(np.sum(session.chew[max(lo, 0):max(hi, 0)] == 1)))
    return np.asarray(labels), np.asarray(chews)


def annotation_scores(session, starts, probability, chews, threshold, window):
    """Scores against the session's annotations over the windows that are
    both annotated and scored; None if there are none."""
    labels, true_chews = window_labels(session, starts, window)
    keep = (labels >= 0) & np.isfinite(probability)
    if not keep.any():
        return None
    m = ets_eval.metrics(labels[keep], probability[keep], threshold)
    eating = keep & (probability >= threshold)
    out = {"annotated_windows": int(keep.sum()), "f1": m["f1"],
           "precision": m["precision"], "recall": m["recall"],
           "annotated_eating_min": float(np.sum(keep & (labels == 1)) * window / 60),
           "annotated_chews": float(true_chews[keep].sum())}
    if np.isfinite(chews[keep]).any():
        out["model_chews_same_span"] = float(np.nansum(chews[eating]))
        out["model_eating_min_same_span"] = float(eating.sum() * window / 60)
    return out


def compare(conn, participant, study, starts, probability, chews, threshold, window):
    sessions = [s for s in ets_data.load_sessions(conn, [participant])
                if s.study == study]
    if not sessions:
        print("\n  --compare: this session has no annotations")
        return
    if sessions[0].annotated_range() is None:
        print("\n  --compare: annotations are all missing")
        return
    m = annotation_scores(sessions[0], starts, probability, chews, threshold, window)
    if m is None:
        print("\n  --compare: no scored window falls inside the annotated span")
        return
    print(f"\n  against the annotation ({m['annotated_windows']} annotated windows): "
          f"F1 {m['f1']:.3f}, precision {m['precision']:.3f}, recall {m['recall']:.3f}")
    if "model_chews_same_span" in m:
        print(f"  chews in the annotated span: {m['annotated_chews']:.0f} annotated, "
              f"{m['model_chews_same_span']:.0f} counted by the model in its eating windows")
    print("  (this participant may be in the training set; see the note above)")


def write_outputs(prefix: Path, starts, probability, chews, eating, missing,
                  found, window: float) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}_windows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_utc", "end_utc", "probability", "eating", "chews",
                         "missing_fraction"])
        for s, p, e, c, g in zip(starts, probability, eating, chews, missing):
            writer.writerow([utc(s), utc(s + window),
                             "" if not np.isfinite(p) else f"{p:.4f}",
                             "" if not np.isfinite(p) else int(e),
                             "" if not np.isfinite(c) else f"{c:.2f}", f"{g:.3f}"])
    with open(f"{prefix}_episodes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_utc", "end_utc", "duration_s", "windows",
                         "mean_probability", "chews", "chews_per_min"])
        for row in episode_rows(starts, probability, chews, eating, found, window):
            writer.writerow([row["start_utc"], row["end_utc"], row["duration_s"],
                             row["windows"], row["mean_probability"], row["chews"],
                             row["chews_per_min"]])


def episode_rows(starts, probability, chews, eating, found, window: float):
    """One dict per episode. Chews are summed over the episode's eating
    windows only, so the bridged pauses add none; the rate is per minute of
    those windows."""
    rows = []
    for first, last, duration in found:
        span = slice(first, last + 1)
        eat = eating[span]
        c = chews[span][eat]
        counted = c[np.isfinite(c)]
        minutes = eat.sum() * window / 60
        rows.append({"start_utc": utc(starts[first]), "end_utc": utc(starts[last] + window),
                     "duration_s": f"{duration:.0f}", "windows": last - first + 1,
                     "mean_probability": f"{np.nanmean(probability[span]):.3f}",
                     "chews": f"{counted.sum():.0f}" if len(counted) else "",
                     "chews_per_min": f"{counted.sum() / minutes:.1f}"
                     if len(counted) and minutes else ""})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ets_data.add_database_arguments(parser)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--model", default=str(Path("models") / "food_intake_final"))
    parser.add_argument("--start", help='UTC, "YYYY-MM-DD HH:MM:SS" (default: recording start)')
    parser.add_argument("--end", help='UTC, "YYYY-MM-DD HH:MM:SS" (default: recording end)')
    parser.add_argument("--merge-gap", type=float, default=32.0)
    parser.add_argument("--min-episode", type=float, default=0.0)
    parser.add_argument("--compare", action="store_true",
                        help="also score against the session's annotations")
    parser.add_argument("--out", default=None,
                        help="output prefix (default detections\\<participant>_<study>)")
    args = parser.parse_args()

    model, config, record, transform = load_model(Path(args.model))
    threshold = float(record["threshold"])
    scale = chew_scale(record, config)
    window = float(config.window_seconds)
    conn = ets_data.connect(args)
    try:
        if args.start and args.end:
            start, end = parse_utc(args.start), parse_utc(args.end)
        else:
            start, end = recording_range(conn, args.participant, args.study)
        ref = _SessionRef(participant=args.participant, study=args.study)
        print(f"{args.participant}/{args.study}: {utc(start)} to {utc(end)} UTC "
              f"({(end - start) / 3600:.2f} h); model {args.model}, threshold "
              f"{threshold:.3f} ({record['threshold_objective']})")
        starts, probability, chews, missing = predict(conn, model, transform, ref,
                                                      start, end, window, scale)
        scored = np.isfinite(probability)
        eating = scored & (np.nan_to_num(probability) >= threshold)
        found = episodes(starts, eating, window, args.merge_gap, args.min_episode)
        if args.compare:
            compare(conn, args.participant, args.study, starts, probability,
                    chews, threshold, window)
    finally:
        conn.close()

    prefix = Path(args.out) if args.out else Path("detections") / f"{args.participant}_{args.study}"
    write_outputs(prefix, starts, probability, chews, eating, missing, found, window)

    hours = scored.sum() * window / 3600
    print(f"\n  {int(scored.sum()):,} windows scored ({hours:.2f} h), "
          f"{int((~scored).sum()):,} skipped for missing data")
    print(f"  eating: {int(eating.sum()):,} windows ({eating.sum() * window / 60:.1f} min), "
          f"{len(found)} episode(s) (pauses up to {args.merge_gap:g} s bridged)")
    for first, last, duration in found[:10]:
        print(f"    {utc(starts[first])} - {utc(starts[last] + window)}   {duration / 60:5.1f} min")
    if len(found) > 10:
        print(f"    ... {len(found) - 10} more in the episodes file")
    print(f"\nwrote {prefix}_windows.csv\n      {prefix}_episodes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

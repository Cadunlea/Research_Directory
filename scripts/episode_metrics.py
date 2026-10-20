r"""
episode_metrics.py - how a finished run behaves at the level of eating EPISODES.

Window F1 counts 8 s windows. A user experiences episodes: did the device
notice that I was eating, how long did it take, and how often did it claim I
was eating when I was not? This reads the held-out predictions a run saved
(results\<run>_predictions.npz) and answers those questions. No retraining.

    python episode_metrics.py results\<run>.json
    python episode_metrics.py results\<default>.json results\<fa15>.json
    python episode_metrics.py results\<run>.json --merge-gap 0 32 64

EPISODES. Within each session, consecutive eating windows form an episode.
Two episodes separated by a gap of at most --merge-gap seconds count as one
(0 = bouts exactly as annotated at 8 s resolution; 32 = pauses of up to four
windows are part of the same episode). The same rule builds the predicted
episodes from the model's decisions. A gap in the recording always ends an
episode. Windows and thresholds are exactly those of the run itself.

REPORTED, per run and merge gap
    episodes detected   true episodes with at least one window predicted eating
    by length           the same, split by episode duration
    delay               seconds from a detected episode's start to the first
                        window predicted eating (median and 90th percentile)
    false episodes / h  predicted episodes overlapping no true eating and not
                        touching it either, per hour of non-eating time
    edge episodes / h   predicted episodes touching, but not overlapping, true
                        eating: a meal reported one window early or late
    episode precision   predicted episodes that overlap true eating

Each session here is roughly one meal, so "meals detected" is reported too,
but it is close to trivial; the episode rows are the informative ones.
Writes results\episode_metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from error_breakdown import sessions

LENGTH_BINS = ((8, 8, "8 s"), (16, 24, "16-24 s"), (32, 56, "32-56 s"),
               (64, None, "64 s or more"))


def runs_of(flags: np.ndarray, breaks: np.ndarray) -> List[Tuple[int, int]]:
    """(first, last) positions of each run of True, never across a break.

    breaks[i] is True when position i+1 does not follow position i in time.
    """
    out, start = [], None
    for i, flag in enumerate(flags):
        if not flag:
            continue
        if start is None:
            start = i
        if i == len(flags) - 1 or not flags[i + 1] or breaks[i]:
            out.append((start, i))
            start = None
    return out


def merge(spans: List[Tuple[int, int]], times: np.ndarray, window: float,
          gap: float, breaks: np.ndarray) -> List[Tuple[int, int]]:
    """Join spans whose silence is at most `gap` seconds and that have no
    recording break between them."""
    if not spans:
        return []
    merged = [spans[0]]
    for first, last in spans[1:]:
        prev_first, prev_last = merged[-1]
        silence = times[first] - (times[prev_last] + window)
        broken = bool(np.any(breaks[prev_last:first]))
        if silence <= gap + 1e-6 and not broken:
            merged[-1] = (prev_first, last)
        else:
            merged.append((first, last))
    return merged


def session_episodes(t: np.ndarray, y: np.ndarray, pred: np.ndarray,
                     window: float, gap: float) -> Dict[str, list]:
    breaks = np.abs(np.diff(t) - window) > 0.5 * window
    true = merge(runs_of(y == 1, breaks), t, window, gap, breaks)
    said = merge(runs_of(pred == 1, breaks), t, window, gap, breaks)

    detected, delays, lengths = [], [], []
    for first, last in true:
        span = slice(first, last + 1)
        hits = np.flatnonzero(pred[span] == 1)
        lengths.append(t[last] + window - t[first])
        detected.append(len(hits) > 0)
        if len(hits):
            delays.append(t[first + hits[0]] - t[first])

    eating = y == 1
    false, edge, real = 0, 0, 0
    for first, last in said:
        if eating[first:last + 1].any():
            real += 1
            continue
        before = first > 0 and not breaks[first - 1] and eating[first - 1]
        after = last < len(y) - 1 and not breaks[last] and eating[last + 1]
        if before or after:
            edge += 1
        else:
            false += 1
    return {"detected": detected, "lengths": lengths, "delays": delays,
            "false": false, "edge": edge, "real": real}


def evaluate(path: Path, gap: float) -> Dict[str, float]:
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"),
                   allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    probability = data[f"probability_seed{seed}"]
    threshold = data[f"threshold_seed{seed}"]
    keep = data["evaluated"] & np.isfinite(probability) & np.isfinite(threshold)
    rows = np.flatnonzero(keep)
    y = data["y"].astype(int)
    pred = (probability >= threshold).astype(int)
    window = float(data["window_seconds"])
    starts = data["starts"]

    detected, lengths, delays = [], [], []
    false = edge = real = 0
    meals = meals_hit = 0
    for order in sessions(data["participants"][rows], data["studies"][rows],
                          starts[rows]):
        r = rows[order]
        out = session_episodes(starts[r], y[r], pred[r], window, gap)
        detected += out["detected"]
        lengths += out["lengths"]
        delays += out["delays"]
        false += out["false"]
        edge += out["edge"]
        real += out["real"]
        if y[r].any():
            meals += 1
            meals_hit += int(np.any((y[r] == 1) & (pred[r] == 1)))

    non_eating_hours = float(np.sum(y[rows] == 0)) * window / 3600.0
    detected = np.asarray(detected, dtype=bool)
    lengths = np.asarray(lengths)
    result = {
        "run": record["tag"], "merge_gap_s": gap,
        "true_episodes": int(len(detected)),
        "episodes_detected": float(detected.mean()) if len(detected) else 0.0,
        "delay_median_s": float(np.median(delays)) if delays else 0.0,
        "delay_p90_s": float(np.percentile(delays, 90)) if delays else 0.0,
        "predicted_episodes": int(false + edge + real),
        "episode_precision": real / max(false + edge + real, 1),
        "false_episodes_per_h": false / non_eating_hours,
        "edge_episodes_per_h": edge / non_eating_hours,
        "meals_detected": f"{meals_hit}/{meals}",
        "non_eating_hours": non_eating_hours,
    }
    for low, high, label in LENGTH_BINS:
        inside = (lengths >= low - 1e-6) & ((lengths <= high + 1e-6)
                                            if high else True)
        result[f"detected_{label}"] = (f"{detected[inside].mean():.0%} of "
                                       f"{int(inside.sum())}"
                                       if inside.any() else "none")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+", help="result JSON files")
    parser.add_argument("--merge-gap", type=float, nargs="+", default=[0.0, 32.0],
                        help="seconds of pause that still count as the same "
                             "episode (default: 0 and 32)")
    parser.add_argument("--csv", default="results/episode_metrics.csv")
    args = parser.parse_args()

    results = [evaluate(Path(run), gap) for run in args.runs
               for gap in args.merge_gap]
    for r in results:
        print(f"\n{r['run']}   (episodes merged across pauses up to "
              f"{r['merge_gap_s']:g} s)")
        print(f"  true eating episodes      {r['true_episodes']:,}")
        print(f"  episodes detected         {r['episodes_detected']:.1%}")
        for _, _, label in LENGTH_BINS:
            print(f"      {label:<20}  {r[f'detected_{label}']}")
        print(f"  delay to first detection  median {r['delay_median_s']:.0f} s, "
              f"90th percentile {r['delay_p90_s']:.0f} s")
        print(f"  predicted episodes        {r['predicted_episodes']:,}  "
              f"(episode precision {r['episode_precision']:.1%})")
        print(f"  false episodes / hour     {r['false_episodes_per_h']:.1f}   "
              f"(away from any eating)")
        print(f"  edge episodes / hour      {r['edge_episodes_per_h']:.1f}   "
              f"(touching a real episode: reported early or late)")
        print(f"  meals (sessions) with any detection  {r['meals_detected']}")
        print(f"  non-eating time           {r['non_eating_hours']:.1f} h")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

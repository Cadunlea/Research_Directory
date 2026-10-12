r"""
error_breakdown.py - where does a finished run go wrong?

Reads the held-out predictions a run saved (results\<run>_predictions.npz) and
sorts every wrong window by its neighbours in the same session. No retraining;
seconds.

    python error_breakdown.py results\<run>.json
    python error_breakdown.py results\<run>.json --all-participants
    python error_breakdown.py results\<run>.json --threshold 0.5

Each fold's threshold is chosen on its own inner validation participants, so
it differs between participants; the table shows it. --threshold applies one
fixed value to every fold instead, as a sensitivity check: a value picked after
looking at these results is chosen on the test participants, so it is not a
reportable result.

MISSED EATING (false negatives)
    edge        a neighbour is eating and a neighbour is not: first or last
                window of a bout
    mid-meal    no neighbour is non-eating: inside a meal
    short bout  no neighbour is eating: a bout one window long

FALSE ALARMS (false positives), as runs of consecutive false windows
    edge        the run touches real eating: it extends a meal
    isolated    one window, not touching eating
    sustained   two or more windows, not touching eating

Neighbours are the windows directly before and after on the evaluation grid;
a window next to a gap (missing data, end of the annotated span) has fewer.

For each category it also prints the F1 the run would reach if every error of
that category were corrected: an upper bound on what a method aimed at that
category could gain. "False episodes per hour" counts isolated and sustained
runs per hour of non-eating, the false-alarm figure a user would experience.

Writes results\<run>_errors.csv with every participant.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

import ets_eval

FN_KINDS = ("edge", "mid-meal", "short bout")
FP_KINDS = ("edge", "isolated", "sustained")


def sessions(participants: np.ndarray, studies: np.ndarray,
             starts: np.ndarray) -> List[np.ndarray]:
    """Indices of each session, in time order."""
    keys = np.char.add(np.char.add(participants.astype(str), "|"),
                       studies.astype(str))
    out = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        out.append(rows[np.argsort(starts[rows])])
    return out


def classify(order: np.ndarray, y: np.ndarray, pred: np.ndarray,
             starts: np.ndarray, window: float) -> Dict[int, tuple]:
    """(type, kind) for every wrong window of one session."""
    t = starts[order]
    contiguous = np.abs(np.diff(t) - window) < 0.5 * window   # i and i+1 adjacent
    n = len(order)

    def neighbours(i: int) -> List[int]:
        out = []
        if i > 0 and contiguous[i - 1]:
            out.append(i - 1)
        if i < n - 1 and contiguous[i]:
            out.append(i + 1)
        return out

    labels = y[order]
    predicted = pred[order]
    result: Dict[int, tuple] = {}

    for i in range(n):
        if labels[i] == 1 and predicted[i] == 0:
            near = [labels[j] for j in neighbours(i)]
            if 0 in near and 1 in near:
                kind = "edge"
            elif 0 in near:
                kind = "short bout"
            else:
                kind = "mid-meal"
            result[order[i]] = ("fn", kind)

    false = (labels == 0) & (predicted == 1)
    i = 0
    while i < n:
        if not false[i]:
            i += 1
            continue
        run = [i]
        while run[-1] < n - 1 and contiguous[run[-1]] and false[run[-1] + 1]:
            run.append(run[-1] + 1)
        touches = any(labels[j] == 1 for k in run for j in neighbours(k))
        kind = "edge" if touches else ("isolated" if len(run) == 1 else "sustained")
        for k in run:
            result[order[k]] = ("fp", kind)
        result[order[run[0]]] = ("fp", kind, "run_start")
        i = run[-1] + 1
    return result


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run", help="result JSON of a finished run")
    parser.add_argument("--all-participants", action="store_true",
                        help="print every participant, not only the 10 weakest")
    parser.add_argument("--threshold", type=float, default=None,
                        help="one fixed threshold for every fold instead of each "
                             "fold's own (sensitivity check only)")
    args = parser.parse_args()

    path = Path(args.run)
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"),
                   allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    keep = data["evaluated"]
    y = data["y"].astype(int)
    probability = data[f"probability_seed{seed}"]
    threshold = data[f"threshold_seed{seed}"]
    keep = keep & np.isfinite(probability) & np.isfinite(threshold)
    fold_threshold = threshold.copy()
    if args.threshold is not None:
        threshold = np.full_like(threshold, args.threshold)
    pred = (probability >= threshold).astype(int)
    window = float(data["window_seconds"])
    participants = data["participants"].astype(str)
    starts = data["starts"]

    rows = np.flatnonzero(keep)
    labels: Dict[int, tuple] = {}
    for order in sessions(participants[rows], data["studies"][rows], starts[rows]):
        labels.update(classify(rows[order], y, pred, starts, window))

    tp = int(np.sum((y == 1) & (pred == 1) & keep))
    fp = int(np.sum((y == 0) & (pred == 1) & keep))
    fn = int(np.sum((y == 1) & (pred == 0) & keep))
    tn = int(np.sum((y == 0) & (pred == 0) & keep))
    non_eating_hours = (fp + tn) * window / 3600.0

    fn_count = Counter(v[1] for v in labels.values() if v[0] == "fn")
    fp_count = Counter(v[1] for v in labels.values() if v[0] == "fp")
    fp_runs = Counter(v[1] for v in labels.values() if len(v) == 3)

    print(f"\n{record['tag']}")
    print(f"  {int(keep.sum()):,} held-out windows of {window:g} s; "
          f"F1 {f1(tp, fp, fn):.3f} (run reported "
          f"{record['summary']['pooled_f1']:.3f})")
    per_participant = {p: float(fold_threshold[rows][participants[rows] == p][0])
                       for p in np.unique(participants[rows])}
    values = np.array(list(per_participant.values()))
    print(f"  fold thresholds across participants: min {values.min():.2f}, "
          f"median {np.median(values):.2f}, max {values.max():.2f}")
    if args.threshold is not None:
        print(f"  SENSITIVITY CHECK: every fold scored at a fixed threshold of "
              f"{args.threshold:g}, not its own")

    print(f"\n  missed eating: {fn:,} windows ({fn / max(tp + fn, 1):.1%} of eating)")
    print(f"  {'':14s}{'windows':>9s}{'share':>8s}{'F1 if fixed':>13s}")
    for kind in FN_KINDS:
        c = fn_count[kind]
        print(f"  {kind:14s}{c:9,d}{c / max(fn, 1):8.1%}"
              f"{f1(tp + c, fp, fn - c):13.3f}")

    print(f"\n  false alarms: {fp:,} windows "
          f"({fp / non_eating_hours:.1f} per hour of non-eating)")
    print(f"  {'':14s}{'windows':>9s}{'share':>8s}{'runs':>7s}{'F1 if fixed':>13s}")
    for kind in FP_KINDS:
        c = fp_count[kind]
        print(f"  {kind:14s}{c:9,d}{c / max(fp, 1):8.1%}{fp_runs[kind]:7,d}"
              f"{f1(tp, fp - c, fn):13.3f}")

    edges = fn_count["edge"] + fp_count["edge"]
    episodes = fp_runs["isolated"] + fp_runs["sustained"]
    print(f"\n  all edge errors forgiven:  F1 {f1(tp + fn_count['edge'], fp - fp_count['edge'], fn - fn_count['edge']):.3f}"
          f"  ({edges / max(fn + fp, 1):.0%} of all errors are at bout edges)")
    print(f"  false episodes per hour of non-eating: {episodes / non_eating_hours:.1f}"
          f"  (isolated + sustained runs; edge runs extend a real meal)")

    # Per participant.
    table = []
    for participant in np.unique(participants[rows]):
        mine = rows[participants[rows] == participant]
        p_tp = int(np.sum((y[mine] == 1) & (pred[mine] == 1)))
        p_fp = int(np.sum((y[mine] == 0) & (pred[mine] == 1)))
        p_fn = int(np.sum((y[mine] == 1) & (pred[mine] == 0)))
        kinds = Counter(labels[i][:2] for i in mine if i in labels)
        row = {"participant": participant, "windows": len(mine),
               "threshold": round(per_participant[participant], 3),
               "f1": f1(p_tp, p_fp, p_fn), "missed": p_fn, "false_alarms": p_fp}
        for kind in FN_KINDS:
            row[f"missed_{kind}"] = kinds[("fn", kind)]
        for kind in FP_KINDS:
            row[f"false_{kind}"] = kinds[("fp", kind)]
        table.append(row)
    table.sort(key=lambda r: r["f1"])

    shown = table if args.all_participants else table[:10]
    print(f"\n  {'weakest participants' if not args.all_participants else 'participants'}"
          f" (counts of windows)")
    print(f"  {'':12s}{'':6s}{'':6s}{'':6s}{'---- missed eating ----':>27s}"
          f"{'---- false alarms ----':>28s}")
    print(f"  {'participant':12s}{'F1':>6s}{'thr':>6s}{'wins':>6s}"
          f"{'edge':>8s}{'mid':>8s}{'short':>8s}{'  ':3s}"
          f"{'edge':>8s}{'isol':>8s}{'sust':>8s}")
    for r in shown:
        print(f"  {r['participant']:12s}{r['f1']:6.3f}{r['threshold']:6.2f}{r['windows']:6d}"
              f"{r['missed_edge']:8d}{r['missed_mid-meal']:8d}{r['missed_short bout']:8d}{'':3s}"
              f"{r['false_edge']:8d}{r['false_isolated']:8d}{r['false_sustained']:8d}")

    suffix = "" if args.threshold is None else f"_thr{args.threshold:g}"
    out = path.with_name(f"{record['tag']}_errors{suffix}.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

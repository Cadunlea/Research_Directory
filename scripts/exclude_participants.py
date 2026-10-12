r"""
exclude_participants.py - how much do the weakest participants move the numbers?

Recomputes the pooled metrics of a run with and without a set of participants,
and puts the difference next to the run's own uncertainty: a 95% bootstrap
interval for pooled F1, resampling participants. If the value without them
falls inside that interval, removing them is not a meaningful change.

With several runs, it also repeats the paired comparison against the reference
on the remaining participants, to show whether any conclusion (which changes
help, which hurt) depends on them.

The excluded set is chosen ONCE, from the reference run, and applied to every
run. Choosing it separately per run, from that run's own scores, would
favour every run equally badly and bias each comparison.

USAGE
-----
    python exclude_participants.py results\<run>.json
    python exclude_participants.py results\<run>.json --below 0.6
    python exclude_participants.py results\<run>.json --exclude AIM147560 AIM112400
    python exclude_participants.py results\*loso*.json --reference results\<run>.json

Reporting: excluding participants because their scores are low is not by itself
a reason to exclude them. Report the full-cohort numbers as the result; numbers
without a participant belong in the paper only with an independent reason
(sensor fault, annotation error) found by looking at the data.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

import ets_eval
from compare_runs import load, pick_reference


def pooled(participants: Dict[str, Dict[str, float]], keep: List[str],
           window_seconds: float) -> Dict[str, float]:
    counts = {k: sum(participants[p][k] for p in keep)
              for k in ("tp", "fp", "fn", "tn")}
    metrics = ets_eval.metrics_from_counts(counts)
    metrics["fa_per_hour"] = ets_eval.false_alarms_per_hour(
        counts["fp"], counts["tn"], window_seconds)
    metrics["mean_participant_f1"] = float(
        np.mean([participants[p]["f1"] for p in keep]))
    metrics["windows"] = int(sum(counts.values()))
    return metrics


def bootstrap_f1(participants: Dict[str, Dict[str, float]], keep: List[str],
                 n_boot: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for index in range(n_boot):
        sample = [keep[i] for i in rng.integers(0, len(keep), len(keep))]
        values[index] = pooled(participants, sample, 1.0)["f1"]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+", help="result JSON files or patterns")
    parser.add_argument("--reference", default="",
                        help="run that defines the excluded participants "
                             "(default: the default configuration)")
    parser.add_argument("--below", type=float, default=0.55,
                        help="exclude participants whose F1 in the reference "
                             "run is below this (default 0.55)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="exclude exactly these participant IDs instead")
    args = parser.parse_args()

    runs = load(args.runs)
    if not runs:
        raise SystemExit("no result files found")
    reference = pick_reference(runs, args.reference)
    ref_parts = reference["participants"]
    everyone = sorted(ref_parts)

    if args.exclude is not None:
        excluded = sorted(p for p in args.exclude if p in ref_parts)
        missing = sorted(set(args.exclude) - set(ref_parts))
        if missing:
            print(f"  not in the reference run: {', '.join(missing)}")
    else:
        excluded = sorted(p for p in everyone if ref_parts[p]["f1"] < args.below)
    if not excluded:
        raise SystemExit("no participants to exclude")
    kept = [p for p in everyone if p not in excluded]

    window_seconds = float(reference["config"]["window_seconds"])
    full = pooled(ref_parts, everyone, window_seconds)
    reduced = pooled(ref_parts, kept, window_seconds)
    low, high = bootstrap_f1(ref_parts, everyone)

    print(f"\nreference: {reference['tag']}")
    print(f"excluded ({len(excluded)} of {len(everyone)} participants, "
          f"{1 - reduced['windows'] / full['windows']:.1%} of windows):")
    for p in excluded:
        print(f"    {p}  F1 {ref_parts[p]['f1']:.3f}  "
              f"({int(sum(ref_parts[p][k] for k in ('tp', 'fp', 'fn', 'tn')))} windows)")

    print(f"\n  {'':24s}{'all':>9s}{'without':>9s}{'change':>9s}")
    for key, label in (("f1", "pooled F1"), ("accuracy", "accuracy"),
                       ("balanced_accuracy", "balanced accuracy"),
                       ("precision", "precision"), ("recall", "recall"),
                       ("fa_per_hour", "false alarms / h"),
                       ("mean_participant_f1", "mean participant F1")):
        digits = 1 if key == "fa_per_hour" else 3
        print(f"  {label:24s}{full[key]:9.{digits}f}{reduced[key]:9.{digits}f}"
              f"{reduced[key] - full[key]:+9.{digits}f}")

    inside = low <= reduced["f1"] <= high
    print(f"\n  95% interval of pooled F1, all participants: [{low:.3f}, {high:.3f}]")
    print(f"  pooled F1 without them ({reduced['f1']:.3f}) is "
          f"{'INSIDE' if inside else 'OUTSIDE'} that interval")

    others = [r for r in runs if r is not reference]
    if others:
        print(f"\n  paired comparison against the reference, "
              f"all participants vs without the excluded ones")
        print(f"  {'F1 change [95% CI], all':>30s}  {'without':>26s}  run")
        for run in others:
            cand = run["participants"]
            shared_all = {p: ref_parts[p] for p in everyone if p in cand}
            shared_kept = {p: ref_parts[p] for p in kept if p in cand}
            cells = []
            for subset in (shared_all, shared_kept):
                c = ets_eval.paired_comparison(subset, {p: cand[p] for p in subset})
                cells.append(f"{c['pooled_f1_difference']:+.3f} "
                             f"[{c['pooled_f1_difference_ci_low']:+.3f}, "
                             f"{c['pooled_f1_difference_ci_high']:+.3f}]")
            print(f"  {cells[0]:>30s}  {cells[1]:>26s}  {run['tag']}")
    print("\n  Full-cohort numbers remain the result; see the note at the top "
          "of this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

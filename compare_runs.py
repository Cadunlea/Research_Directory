r"""
compare_runs.py - is model B really better than model A, participant by
participant?

Reads two result JSONs written by the training scripts and compares them on
the participants both held out. The difference of two pooled F1 numbers says
nothing about whether that difference is larger than chance; this answers that
question with the unit of variation that actually matters here, the
participant.

    python compare_runs.py model_v1_outputs\model_v1_metrics_w8_loso.json ^
                           model_v2_outputs\model_v2_metrics_w8_loso.json

    REM an ablation: full Model B against Model B without the chew head
    python compare_runs.py model_v2_outputs\model_v2_metrics_w8_loso.json ^
                           model_v2_outputs\model_v2_metrics_w8_loso_noaux.json

Run B is compared against run A, so positive differences mean B is better.
Both runs should use the same CV scheme (LOSO for both, ideally). Comparing a
4-fold run with a LOSO run mixes a model difference with a training-set-size
difference, and the script warns when that happens.

Sweep files (model_v2_sweep_*.json) hold several runs; pass --run N to pick
one (0-based, in the order they were trained).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import ets_eval


def load(path: str, run: int) -> Tuple[Dict, Dict[str, Dict]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    entry = payload["runs"][run] if "runs" in payload else payload
    participants = entry.get("participants")
    if not participants:
        sys.exit(f"{path} has no per-participant results. It was written "
                 "before per-participant output was added; re-run it.")
    return entry, participants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("a", help="baseline run (JSON)")
    parser.add_argument("b", help="run compared against it (JSON)")
    parser.add_argument("--run", type=int, default=0,
                        help="which run inside a sweep file (both files)")
    parser.add_argument("--boot", type=int, default=10000,
                        help="bootstrap resamples over participants")
    parser.add_argument("--metric", default="f1",
                        choices=("f1", "balanced_accuracy", "accuracy",
                                 "precision", "recall"))
    args = parser.parse_args()

    entry_a, a = load(args.a, args.run)
    entry_b, b = load(args.b, args.run)

    if entry_a.get("cv") != entry_b.get("cv"):
        print(f"WARNING: different CV schemes ({entry_a.get('cv')} vs "
              f"{entry_b.get('cv')}). Training-set size differs between "
              "them, so the comparison is not purely between models.\n")
    only = sorted(set(a) ^ set(b))
    if only:
        print(f"WARNING: {len(only)} participant(s) scored in only one run, "
              f"left out: {', '.join(only)}\n")

    result = ets_eval.paired_comparison(a, b, key=args.metric,
                                        n_boot=args.boot)
    shared = sorted(set(a) & set(b))
    key = args.metric

    line = "=" * 74
    print(line)
    print(f"  A  {Path(args.a).name}")
    print(f"  B  {Path(args.b).name}")
    print(line)
    print(f"\n  {'participant':<16}{'n':>6}{'eat %':>7}"
          f"{'A ' + key:>12}{'B ' + key:>12}{'B - A':>9}")
    for participant in sorted(shared, key=lambda p: b[p][key] - a[p][key]):
        diff = b[participant][key] - a[participant][key]
        print(f"  {participant:<16}{int(a[participant]['n']):>6}"
              f"{100 * a[participant]['positive_rate']:>6.0f}%"
              f"{a[participant][key]:>12.3f}{b[participant][key]:>12.3f}"
              f"{diff:>+9.3f}")

    n = result["n_participants"]
    print(f"\n  per participant ({n} in common)")
    print(f"    mean {key} difference     {result[f'mean_{key}_difference']:+.4f}")
    print(f"    median {key} difference   {result[f'median_{key}_difference']:+.4f}")
    print(f"    B better / worse          {result['improved']} / "
          f"{result['worse']}")
    print(f"    Wilcoxon signed-rank p    {result['wilcoxon_p']:.4f}")

    print(f"\n  pooled F1 over those participants")
    print(f"    A {result['pooled_f1_a']:.4f}   B {result['pooled_f1_b']:.4f}   "
          f"B - A {result['pooled_f1_difference']:+.4f}")
    print(f"    95% bootstrap interval    "
          f"[{result['pooled_f1_difference_ci_low']:+.4f}, "
          f"{result['pooled_f1_difference_ci_high']:+.4f}]  "
          f"({args.boot:,} resamples of participants)")

    excludes_zero = (result["pooled_f1_difference_ci_low"] > 0 or
                     result["pooled_f1_difference_ci_high"] < 0)
    print(f"\n  reading: the interval {'EXCLUDES' if excludes_zero else 'includes'}"
          f" zero and p {'<' if result['wilcoxon_p'] < 0.05 else '>='} 0.05.")
    if excludes_zero and result["wilcoxon_p"] < 0.05:
        print("  Both say the difference is larger than participant-to-"
              "participant noise.")
    elif not excludes_zero and result["wilcoxon_p"] >= 0.05:
        print("  Neither separates the difference from noise. Report it as "
              "measured,\n  not as an effect.")
    else:
        print("  The two disagree. The Wilcoxon weights participants equally, "
              "the pooled\n  F1 weights them by window count; report both "
              "and do not lean on either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

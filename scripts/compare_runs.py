r"""
compare_runs.py - one table of every run, each tested against a reference.

Two runs on the same participants are PAIRED: every held-out participant was
scored by both. So the right question is not "is 0.85 bigger than 0.84?" but
"across participants, does the change help more of them than it hurts, and by
more than chance?". For each run this prints the pooled numbers, then against
the reference run: the change in pooled F1 with a 95% bootstrap interval
(resampling participants), how many participants improved or got worse, and a
Wilcoxon signed-rank p-value on the per-participant F1 differences.

A difference whose interval spans zero is not a result, however large the
headline gap. That is the line the paper has to draw.

USAGE
-----
    python compare_runs.py results\*.json
    python compare_runs.py results\*.json --reference results\time_se_attnpool_chew0.2_w8_128hz_loso_seed9.json
    python compare_runs.py results\*.json --csv results_summary.csv

The pattern is expanded here, so it works in Windows cmd, which does not
expand * itself. Without --reference, the default configuration is used when
present.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List

import ets_eval

DEFAULT_PREFIX = "time_se_attnpool_chew0.2_w8_128hz_"


def load(patterns: List[str]) -> List[Dict]:
    paths = sorted({Path(p) for pattern in patterns
                    for p in (glob.glob(pattern) or [pattern])})
    runs = []
    for path in paths:
        if path.suffix != ".json" or not path.is_file():
            continue
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        if "participants" not in record or "summary" not in record:
            print(f"  skipped {path.name}: not a train_food_intake.py result")
            continue
        record["_path"] = path
        runs.append(record)
    return runs


def pick_reference(runs: List[Dict], requested: str) -> Dict:
    if requested:
        wanted = Path(requested).name
        for run in runs:
            if run["_path"].name == wanted or run["tag"] == requested:
                return run
        raise SystemExit(f"reference {requested} is not among the runs given")
    defaults = [r for r in runs if r["tag"].startswith(DEFAULT_PREFIX)]
    loso = [r for r in defaults if r["cv"] == "loso"]
    return (loso or defaults or runs)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("runs", nargs="+", help="results JSON files or patterns")
    parser.add_argument("--reference", default="",
                        help="the run every other run is compared with")
    parser.add_argument("--csv", default="results_summary.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    runs = load(args.runs)
    if not runs:
        raise SystemExit("no result files found")
    reference = pick_reference(runs, args.reference)
    reference_participants = reference["participants"]

    rows = []
    for run in runs:
        summary = run["summary"]
        row = {
            "run": run["tag"],
            "description": run.get("description", ""),
            "cv": run.get("cv", ""),
            "participants": len(run["participants"]),
            "windows": summary["n"],
            "f1": summary["pooled_f1"],
            "balanced_accuracy": summary["pooled_balanced_accuracy"],
            "accuracy": summary["pooled_accuracy"],
            "precision": summary["pooled_precision"],
            "recall": summary["pooled_recall"],
            "false_alarms_per_hour": summary.get("false_alarms_per_hour", ""),
            "participant_f1_mean": summary.get("participant_f1_mean", ""),
            "participant_f1_sd": summary.get("participant_f1_std", ""),
        }
        if run is reference:
            row.update(reference="yes")
        else:
            shared = set(run["participants"]) & set(reference_participants)
            if len(shared) < 3:
                row.update(reference="no shared participants")
            else:
                paired = ets_eval.paired_comparison(
                    reference_participants, run["participants"],
                    n_boot=args.bootstrap)
                row.update(
                    reference="",
                    f1_change=paired["pooled_f1_difference"],
                    ci_low=paired["pooled_f1_difference_ci_low"],
                    ci_high=paired["pooled_f1_difference_ci_high"],
                    improved=paired["improved"], worse=paired["worse"],
                    wilcoxon_p=paired["wilcoxon_p"],
                    shared_participants=paired["participants"])
        rows.append(row)

    rows.sort(key=lambda r: -r["f1"])
    print(f"\nreference: {reference['tag']}\n")
    print(f"  {'F1':>6} {'bal acc':>7} {'prec':>6} {'recall':>6} {'FA/h':>6}"
          f" {'F1 change [95% CI]':>24} {'better/worse':>12} {'p':>7}  run")
    for row in rows:
        fa = row["false_alarms_per_hour"]
        fa = f"{fa:6.1f}" if fa != "" else f"{'':6}"
        if row["reference"] == "yes":
            change, counts, p = f"{'(reference)':>24}", f"{'':>12}", f"{'':>7}"
        elif "f1_change" in row:
            change = (f"{row['f1_change']:+.3f} [{row['ci_low']:+.3f}, "
                      f"{row['ci_high']:+.3f}]")
            change = f"{change:>24}"
            counts = f"{row['improved']:>5} / {row['worse']:<4}"
            counts = f"{counts:>12}"
            p = f"{row['wilcoxon_p']:7.3f}"
        else:
            change, counts, p = f"{row['reference']:>24}", f"{'':>12}", f"{'':>7}"
        print(f"  {row['f1']:6.3f} {row['balanced_accuracy']:7.3f} "
              f"{row['precision']:6.3f} {row['recall']:6.3f} {fa}"
              f" {change} {counts} {p}  {row['run']}")

    print("\n  F1 change: pooled F1 minus the reference's, on shared participants."
          "\n  An interval that spans 0 means the change is not distinguishable"
          "\n  from participant-to-participant variation.")
    cvs = {row["cv"] for row in rows}
    if len(cvs) > 1:
        print(f"\n  NOTE: these runs mix validation schemes ({', '.join(sorted(cvs))})."
              "\n  Compare like with like; report LOSO numbers.")

    fields = ["run", "description", "cv", "participants", "windows", "f1",
              "balanced_accuracy", "accuracy", "precision", "recall",
              "false_alarms_per_hour", "participant_f1_mean", "participant_f1_sd",
              "reference", "f1_change", "ci_low", "ci_high", "improved", "worse",
              "wilcoxon_p", "shared_participants"]
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

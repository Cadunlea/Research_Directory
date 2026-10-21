r"""
session_type_analysis.py - does performance depend on the kind of session?

Scores a finished run per session and per meal type, and tests whether one
group of sessions differs from the rest:

  1. pooled F1, precision, recall and bout structure per meal type
     (median length of annotated eating runs, in windows of 8 s);
  2. per-session F1 of the group against all other sessions
     (Mann-Whitney U, sessions independent);
  3. the fairer comparison: participants who have a session in BOTH groups.
     Each is their own control, which removes differences between people
     (paired Wilcoxon on per-participant F1 differences).

The group is a meal type (--type, default lunch) or, with --groups, the
sessions eaten with other people. Meal type is not a stand-in for group
setting: some lunches were eaten alone. --groups is a CSV with columns
participant, study, group (1 with others, 0 alone); sessions missing from it
are left out of the group comparison.

    python session_type_analysis.py results_fixed\<run>.json
    python session_type_analysis.py results_fixed\<run>.json --type ETS_dinner
    python session_type_analysis.py results_fixed\<run>.json --groups session_setting.csv

Reads the saved predictions (first seed); no database, no retraining.
Writes results_fixed\<run>_sessions.csv with one row per session.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import ets_eval


def runs(flags: np.ndarray) -> np.ndarray:
    d = np.diff(np.r_[0, flags.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def counts(y, p):
    return {"tp": int(np.sum(y & p)), "fp": int(np.sum(~y & p)),
            "fn": int(np.sum(y & ~p)), "tn": int(np.sum(~y & ~p))}


def compare_groups(rows, in_group, name: str, rest: str) -> None:
    """Per-session and within-participant F1 tests of one group of sessions."""
    from scipy.stats import mannwhitneyu, wilcoxon

    inside = [r for r in rows if in_group(r) and r["bouts"]]
    outside = [r for r in rows if not in_group(r) and r["bouts"]]
    if not inside or not outside:
        print(f"\n  no sessions on one side of {name} vs {rest}")
        return
    u = mannwhitneyu([r["f1"] for r in inside], [r["f1"] for r in outside])
    print(f"\n  {name} sessions vs {rest} sessions (per-session F1):")
    print(f"    median {np.median([r['f1'] for r in inside]):.3f} ({len(inside)} sessions) vs "
          f"{np.median([r['f1'] for r in outside]):.3f} ({len(outside)}); "
          f"Mann-Whitney p = {u.pvalue:.3f}")

    paired = []
    for person in sorted({r["participant"] for r in inside}):
        mine = [r for r in inside if r["participant"] == person]
        other = [r for r in outside if r["participant"] == person]
        if other:
            paired.append((person, float(np.mean([r["f1"] for r in mine])),
                           float(np.mean([r["f1"] for r in other])),
                           [r["study"] for r in mine], [r["study"] for r in other]))
    if len(paired) < 5:
        print(f"\n  only {len(paired)} participants have both {name} and {rest} sessions; "
              f"too few for a paired test")
        return
    diff = np.array([a - b for _, a, b, _, _ in paired])
    w = wilcoxon(diff)
    print(f"\n  same participants, {name} vs {rest} ({len(paired)} participants):")
    print(f"    median F1 difference {np.median(diff):+.3f}; {int(np.sum(diff < 0))} worse "
          f"{name}, {int(np.sum(diff > 0))} better; Wilcoxon p = {w.pvalue:.3f}")
    for person, a, b, mine, other in paired:
        print(f"      {person}  {'/'.join(mine)} {a:.3f}   {'/'.join(other)} {b:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run")
    parser.add_argument("--type", default="ETS_lunch", help="meal type to test against the rest")
    parser.add_argument("--groups", help="CSV of participant, study, group (1 with others, 0 alone)")
    args = parser.parse_args()

    path = Path(args.run)
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"), allow_pickle=False)
    seed = record["repeats"][0]["seed"]
    prob, thr = data[f"probability_seed{seed}"], data[f"threshold_seed{seed}"]
    keep = data["evaluated"] & np.isfinite(prob) & np.isfinite(thr)
    y = (data["y"] == 1) & keep
    pred = (prob >= thr) & keep
    people, studies, starts = data["participants"], data["studies"], data["starts"]

    rows = []
    for person, study in sorted({(p, s) for p, s in zip(people[keep], studies[keep])}):
        idx = np.flatnonzero(keep & (people == person) & (studies == study))
        idx = idx[np.argsort(starts[idx])]
        c = counts(y[idx], pred[idx])
        m = ets_eval.metrics_from_counts(c)
        bouts = runs(y[idx])
        rows.append({"participant": person, "study": study, "windows": len(idx),
                     "eating_share": round(float(y[idx].mean()), 3),
                     "bouts": len(bouts),
                     "median_bout_windows": float(np.median(bouts)) if len(bouts) else 0.0,
                     **{k: c[k] for k in ("tp", "fp", "fn", "tn")},
                     "f1": round(m["f1"], 4), "precision": round(m["precision"], 4),
                     "recall": round(m["recall"], 4)})

    print(f"\n{record['tag']}\n")
    print(f"  {'meal type':<15}{'sessions':>9}{'windows':>9}{'F1':>8}{'prec':>8}"
          f"{'recall':>8}{'eating':>8}{'median bout':>13}")
    for study in sorted({r["study"] for r in rows}):
        group = [r for r in rows if r["study"] == study]
        total = {k: sum(r[k] for r in group) for k in ("tp", "fp", "fn", "tn")}
        m = ets_eval.metrics_from_counts(total)
        bout = np.median([r["median_bout_windows"] for r in group if r["bouts"]])
        print(f"  {study:<15}{len(group):>9}{sum(r['windows'] for r in group):>9,}"
              f"{m['f1']:>8.3f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{sum(r['tp'] + r['fn'] for r in group) / sum(r['windows'] for r in group):>8.0%}"
              f"{bout * 8:>11.0f} s")

    if args.groups:
        with open(args.groups, newline="", encoding="utf-8-sig") as handle:
            setting = {(r["participant"].strip(), r["study"].strip()): int(r["group"])
                       for r in csv.DictReader(handle) if r["group"].strip()}
        for r in rows:
            r["group"] = setting.get((r["participant"], r["study"]), "")
        known = [r for r in rows if r["group"] != ""]
        print(f"\n  --groups: setting known for {len(known)} of {len(rows)} sessions")
        compare_groups(known, lambda r: r["group"] == 1, "with others", "alone")
    else:
        compare_groups(rows, lambda r: r["study"] == args.type, args.type, "other")

    out = path.with_name(f"{record['tag']}_sessions.csv")
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

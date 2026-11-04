r"""
participant_summary.py - per-participant F1 of a finished run, for the paper.

Prints, for each seed and averaged over seeds:
  * per-participant F1: mean and standard deviation, median, minimum, maximum
    (the mean +- SD is what Doulah et al. 2021 report, so it is the like-for-
    like comparison; pooled F1 counts every window together instead);
  * the range of the decision thresholds the folds chose.

Draws figures\participants_anon.png / .pdf: F1 per held-out participant
(mean over seeds), sorted, labeled P1 to P43 instead of study IDs, with the
pooled F1 dashed. Grayscale, Arial 11. The code-to-ID key is written to
results_fixed\participant_codes.csv: keep it private, out of the paper.

    python participant_summary.py results_fixed\<run>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

import ets_eval                                                 # noqa: E402

INK, MID = "#222222", "#777777"


def f1(y, p):
    m = ets_eval.metrics_from_counts({"tp": int(np.sum(y & p)), "fp": int(np.sum(~y & p)),
                                      "fn": int(np.sum(y & ~p)), "tn": int(np.sum(~y & ~p))})
    return m["f1"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run")
    parser.add_argument("--out", default=str(Path("figures") / "participants_anon"))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})

    path = Path(args.run)
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    data = np.load(path.with_name(f"{record['tag']}_predictions.npz"), allow_pickle=False)
    people = data["participants"]
    seeds = [r["seed"] for r in record["repeats"]]

    per_seed, pooled = {}, []
    print(f"\n{record['tag']}\n")
    print(f"  {'seed':<8}{'pooled F1':>10}{'mean':>8}{'SD':>7}{'median':>8}{'min':>7}"
          f"{'max':>7}   thresholds")
    for seed in seeds:
        prob, thr = data[f"probability_seed{seed}"], data[f"threshold_seed{seed}"]
        keep = data["evaluated"] & np.isfinite(prob) & np.isfinite(thr)
        y, pred = (data["y"] == 1) & keep, (prob >= thr) & keep
        scores = {}
        for person in np.unique(people[keep]):
            rows = keep & (people == person)
            if y[rows].any():                       # F1 needs eating to find
                scores[str(person)] = f1(y[rows], pred[rows])
        per_seed[seed] = scores
        pooled.append(f1(y, pred))
        v = np.array(list(scores.values()))
        fold_thr = np.unique(np.round(thr[keep], 4))
        print(f"  {seed:<8}{pooled[-1]:>10.3f}{v.mean():>8.3f}{v.std(ddof=1):>7.3f}"
              f"{np.median(v):>8.3f}{v.min():>7.3f}{v.max():>7.3f}   "
              f"{fold_thr.min():.2f} to {fold_thr.max():.2f}")

    names = sorted(set.intersection(*(set(s) for s in per_seed.values())))
    mean_f1 = {n: float(np.mean([per_seed[s][n] for s in seeds])) for n in names}
    v = np.array(list(mean_f1.values()))
    print(f"\n  over seeds: pooled F1 {np.mean(pooled):.3f}; per-participant F1 mean "
          f"{v.mean():.3f} +- {v.std(ddof=1):.3f} SD, median {np.median(v):.3f}, "
          f"range {v.min():.3f} to {v.max():.3f} ({len(v)} participants)")
    print(f"  below 0.70: {int(np.sum(v < 0.70))}; below 0.80: {int(np.sum(v < 0.80))}")

    order = sorted(names, key=lambda n: mean_f1[n])
    codes = {n: f"P{i + 1}" for i, n in enumerate(order)}
    key = Path("results_fixed") / "participant_codes.csv"
    key.parent.mkdir(parents=True, exist_ok=True)
    with open(key, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "participant", "f1_mean_over_seeds"])
        for n in order:
            writer.writerow([codes[n], n, f"{mean_f1[n]:.4f}"])

    figure, axis = plt.subplots(figsize=(10, 3.4))
    axis.bar(range(len(order)), [mean_f1[n] for n in order], color=MID, width=0.75)
    axis.axhline(np.mean(pooled), color=INK, linestyle="--", linewidth=1)
    # Bars are sorted ascending, so the left end has room above the line.
    axis.text(-0.4, np.mean(pooled) + 0.02, f"pooled F1 {np.mean(pooled):.2f}",
              ha="left", va="bottom", fontsize=10)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([codes[n] for n in order], rotation=90, fontsize=8)
    axis.set_xlim(-0.6, len(order) - 0.4)
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1")
    axis.set_xlabel("Held-out participant")
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=300)
    print(f"\nwrote {out.with_suffix('.png')}\n      {key} (private: codes to IDs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

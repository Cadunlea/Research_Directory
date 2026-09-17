r"""
ets_eval.py - metrics, threshold selection, temporal smoothing and reporting,
shared by both training scripts so their numbers are directly comparable.

WHY F1 AND BALANCED ACCURACY, NOT ACCURACY ALONE
------------------------------------------------
Accuracy moves with class balance. Per-session eating fraction in this dataset
runs from 17% to 74%, so a model that predicted 'not eating' for every window
in AIM122571's session would score 83% accuracy while being useless. Accuracy
is still reported - it is the number the previous model was quoted at - but F1
and balanced accuracy are what the comparison rests on.

THRESHOLD SELECTION
-------------------
The previous model fixed the decision threshold at 0.40. A threshold is a
parameter like any other: choosing it on the test set inflates the score.
Here it is chosen on the TRAINING folds only, then applied unchanged to the
held-out fold, so the reported number is what a new participant would get.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# Thresholds tried when selecting. Deliberately wide: with a class balance that
# varies this much between participants, the best operating point is not
# necessarily near 0.5.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def metrics(y_true: np.ndarray, y_probability: np.ndarray,
            threshold: float) -> Dict[str, float]:
    """Every headline number at one threshold, computed from counts directly.

    No sklearn dependency here: the four counts define all of these, and
    computing them in one place removes any doubt about which averaging
    convention a library used.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_probability, dtype=float) >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    total = max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(total),
        "positive_rate": float(np.mean(y_true == 1)),
    }


def choose_threshold(y_true: np.ndarray, y_probability: np.ndarray,
                     objective: str = "f1", tolerance: float = 0.01) -> float:
    """Threshold from the data given - which must never be the test fold.

    The CENTRE of the plateau, not the argmax. The objective-versus-threshold
    curve is nearly flat near its peak, so on a validation set of three
    participants the exact maximum sits wherever the noise happens to be
    highest. Measured: taking the argmax gave thresholds of 0.16, 0.17, 0.50 and
    0.50 across four folds - bimodal, with nothing in between - and the two
    folds pinned near the floor lost precision badly (0.593 and 0.767) while
    recall ran to 0.93-0.96.

    F1 makes this worse than it looks, because it ignores true negatives: once
    precision is cheap, lowering the threshold keeps buying recall at almost no
    F1 cost, so ties break downward.

    HONESTY ABOUT WHAT THIS BUYS: simulated against the argmax over 400 trials,
    with validation and test drawn as DIFFERENT participants with their own
    signal quality and eating fraction (as in the real data), the plateau centre
    won by +0.0009 F1 and cut the spread by nothing. It is NOT a performance
    improvement and must not be reported as one. It is kept because a reported
    threshold that jumps between 0.16 and 0.50 across folds, for no reason a
    reader can see, invites a question the argmax cannot answer - and the
    plateau centre gives the same score with a stable number beside it.

    The measured cause of the small drop that prompted this (Model A pooled F1
    0.777 -> 0.765) is more likely the third inner-validation participant,
    which removes a participant from the fit set; the difference is well inside
    the +-0.046 fold-to-fold spread either way.
    """
    scores = np.array([metrics(y_true, y_probability, t)[objective]
                       for t in THRESHOLD_GRID])
    plateau = THRESHOLD_GRID[scores >= scores.max() - tolerance]
    return float(np.median(plateau))


def smooth_predictions(probability: np.ndarray, starts: np.ndarray,
                       participants: np.ndarray, window_seconds: float,
                       hop_seconds: float, kernel: int = 3) -> np.ndarray:
    """Median-filter each participant's probabilities along time.

    Eating is temporally contiguous: a single 8 s window of chewing marooned
    between two minutes of nothing is far more likely to be an error than a
    real meal. A median filter removes those isolated flips without blurring
    the edges of a real bout, which is what a mean filter would do.

    Windows are sorted by time within each participant, and only consecutive
    windows are smoothed together, so a gap in the recording never causes two
    distant moments to be averaged.
    """
    if kernel <= 1:
        return probability
    half = kernel // 2
    smoothed = probability.copy()

    for participant in np.unique(participants):
        rows = np.flatnonzero(participants == participant)
        order = rows[np.argsort(starts[rows])]
        values = probability[order]
        times = starts[order]

        # Split where the gap between consecutive windows exceeds one hop, so
        # separate recordings are never mixed.
        breaks = np.flatnonzero(np.diff(times) > hop_seconds * 1.5 + 1e-6) + 1
        for segment in np.split(np.arange(len(order)), breaks):
            if len(segment) < kernel:
                continue
            block = values[segment]
            padded = np.pad(block, (half, half), mode="edge")
            filtered = np.array([np.median(padded[i:i + kernel])
                                 for i in range(len(block))])
            smoothed[order[segment]] = filtered
    return smoothed


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def aggregate(fold_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Mean and standard deviation across folds.

    The spread matters as much as the mean here. With 20 participants in 4
    folds, one unusual participant moves a fold's score several points, and a
    mean quoted without its spread would overstate how well the number is
    known.
    """
    summary: Dict[str, float] = {}
    for key in ("accuracy", "balanced_accuracy", "f1", "precision", "recall",
                "specificity"):
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    for key in ("tp", "tn", "fp", "fn", "n"):
        summary[key] = int(sum(m[key] for m in fold_metrics))
    pooled = metrics_from_counts(summary)
    summary.update({f"pooled_{k}": v for k, v in pooled.items()})
    return summary


def metrics_from_counts(counts: Dict[str, float]) -> Dict[str, float]:
    """Metrics over every held-out window at once (each appears exactly once
    across the folds), which is not the same as the mean of per-fold scores and
    is the fairer single number when folds differ in size."""
    tp, tn = counts["tp"], counts["tn"]
    fp, fn = counts["fp"], counts["fn"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    total = max(tp + tn + fp + fn, 1)
    return {"accuracy": (tp + tn) / total,
            "balanced_accuracy": (recall + specificity) / 2.0,
            "f1": f1, "precision": precision, "recall": recall,
            "specificity": specificity}


# The numbers to beat, quoted so every report states them rather than relying
# on anyone remembering. The previous model's figures come from a SINGLE
# 2-participant test split of 249 windows, WITH the phantom -8 s shift applied;
# they are a target, not a like-for-like comparison, and should be described
# that way.
BASELINE = {
    "name": "previous FFT-CNN (2-participant split, 249 windows, -8s shift)",
    "accuracy": 0.8032, "balanced_accuracy": 0.8252, "f1": 0.7879,
    "precision": 0.6842, "recall": 0.9286, "specificity": 0.7219,
}
RANDOM_FOREST_F1 = 0.81      # AIM-2 random forest paper


def report(title: str, fold_metrics: Sequence[Dict[str, float]],
           extra: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    summary = aggregate(fold_metrics)
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")

    print(f"\n  {'fold':<6}{'n':>7}{'acc':>8}{'bal acc':>9}{'F1':>8}"
          f"{'prec':>8}{'recall':>8}{'thr':>7}")
    for index, fold in enumerate(fold_metrics):
        print(f"  {index + 1:<6}{fold['n']:>7,}{fold['accuracy']:>8.3f}"
              f"{fold['balanced_accuracy']:>9.3f}{fold['f1']:>8.3f}"
              f"{fold['precision']:>8.3f}{fold['recall']:>8.3f}"
              f"{fold['threshold']:>7.2f}")

    print(f"\n  across folds (mean +- sd)")
    for key in ("accuracy", "balanced_accuracy", "f1", "precision", "recall",
                "specificity"):
        print(f"    {key:<20}{summary[f'{key}_mean']:.4f} "
              f"+- {summary[f'{key}_std']:.4f}")

    print(f"\n  pooled over all {summary['n']:,} held-out windows")
    for key in ("accuracy", "balanced_accuracy", "f1", "precision", "recall",
                "specificity"):
        print(f"    {key:<20}{summary[f'pooled_{key}']:.4f}")
    print(f"    confusion            tp={summary['tp']:,}  fp={summary['fp']:,}  "
          f"fn={summary['fn']:,}  tn={summary['tn']:,}")

    print(f"\n  versus the targets")
    for key, label in (("f1", "F1"), ("accuracy", "accuracy"),
                       ("balanced_accuracy", "balanced accuracy")):
        got = summary[f"pooled_{key}"]
        delta = got - BASELINE[key]
        print(f"    {label:<20}{got:.4f}  vs {BASELINE[key]:.4f} previous "
              f"({delta:+.4f})")
    delta_rf = summary["pooled_f1"] - RANDOM_FOREST_F1
    print(f"    {'F1 vs RF paper':<20}{summary['pooled_f1']:.4f}  vs "
          f"{RANDOM_FOREST_F1:.4f} ({delta_rf:+.4f})")
    print(f"\n  NOTE: the previous figures come from one 2-participant split of "
          f"249\n  windows with the phantom -8 s shift applied. These are "
          f"cross-validated\n  over every participant, which is a harder and "
          f"more honest test.")

    if extra:
        print()
        for key, value in extra.items():
            print(f"  {key}: {value}")
    return summary

r"""
ets_eval.py - metrics, threshold selection, temporal smoothing, reporting and
paired comparison between runs, shared by every training configuration so
their numbers are directly comparable.

WHY F1 AND BALANCED ACCURACY, NOT ACCURACY ALONE
------------------------------------------------
Accuracy moves with class balance. Per-session eating fraction in this dataset
runs from 17% to 74%, so a model that predicted 'not eating' for every window
in AIM122571's session would score 83% accuracy while being useless. Accuracy
is still reported, but F1 and balanced accuracy are what comparisons rest on.

THRESHOLD SELECTION
-------------------
A threshold is a parameter like any other: choosing it on the test set
inflates the score.
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
        # F-beta with beta = 0.5 weights precision twice as heavily as
        # recall: the objective when a false alarm costs more than a miss.
        "f0.5": (1.25 * precision * recall / (0.25 * precision + recall)
                 if (precision + recall) else 0.0),
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

    The measured cause of the small drop that prompted this (FFT baseline pooled F1
    0.777 -> 0.765) is more likely the third inner-validation participant,
    which removes a participant from the fit set; the difference is well inside
    the +-0.046 fold-to-fold spread either way.
    """
    if objective == "false-alarms":
        raise ValueError("use choose_threshold_for_false_alarms")
    scores = np.array([metrics(y_true, y_probability, t)[objective]
                       for t in THRESHOLD_GRID])
    plateau = THRESHOLD_GRID[scores >= scores.max() - tolerance]
    return float(np.median(plateau))


def choose_threshold_for_false_alarms(y_true: np.ndarray,
                                      y_probability: np.ndarray,
                                      max_per_hour: float,
                                      window_seconds: float) -> float:
    """The LOWEST threshold whose false-alarm rate stays within the budget.

    For free-living use a false alarm costs more than a missed window: every
    one is a wrong prompt or a wrong meal in the log. So instead of the best
    F1, this picks the operating point a deployment would: at most
    `max_per_hour` false alarms per hour of non-eating on the validation
    participants, and within that budget as much recall as possible (the
    lowest threshold that meets it). If no threshold meets the budget, the
    strictest one on the grid is returned.
    """
    y_true = np.asarray(y_true).astype(int)
    negatives = max(int(np.sum(y_true == 0)), 1)
    hours = negatives * window_seconds / 3600.0
    for threshold in THRESHOLD_GRID:
        predicted = np.asarray(y_probability) >= threshold
        false_alarms = int(np.sum(predicted & (y_true == 0)))
        if false_alarms / hours <= max_per_hour:
            return float(threshold)
    return float(THRESHOLD_GRID[-1])


def select_threshold(y_true, y_probability, objective: str,
                     window_seconds: float, max_false_alarms: float) -> float:
    if objective == "false-alarms":
        return choose_threshold_for_false_alarms(
            y_true, y_probability, max_false_alarms, window_seconds)
    return choose_threshold(y_true, y_probability, objective)


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


def false_alarms_per_hour(fp: float, tn: float, window_seconds: float) -> float:
    """False positives per hour of NON-eating time.

    The number that matters in free living, where most of the day is not a
    meal: it says how often the detector would fire wrongly, in units a
    reader can picture, independent of how much eating the test set held.
    """
    hours = (fp + tn) * window_seconds / 3600.0
    return float(fp / hours) if hours else 0.0


def report(title: str, fold_metrics: Sequence[Dict[str, float]],
           window_seconds: Optional[float] = None,
           participant_metrics: Optional[Dict[str, Dict[str, float]]] = None,
           extra: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    summary = aggregate(fold_metrics)
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")

    if len(fold_metrics) <= 12:
        print(f"\n  {'fold':<6}{'n':>7}{'acc':>8}{'bal acc':>9}{'F1':>8}"
              f"{'prec':>8}{'recall':>8}{'thr':>7}")
        for index, fold in enumerate(fold_metrics):
            print(f"  {index + 1:<6}{fold['n']:>7,}{fold['accuracy']:>8.3f}"
                  f"{fold['balanced_accuracy']:>9.3f}{fold['f1']:>8.3f}"
                  f"{fold['precision']:>8.3f}{fold['recall']:>8.3f}"
                  f"{fold['threshold']:>7.2f}")

    print(f"\n  across {len(fold_metrics)} folds (mean +- sd)")
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
    if window_seconds:
        summary["false_alarms_per_hour"] = false_alarms_per_hour(
            summary["fp"], summary["tn"], window_seconds)
        print(f"    false alarms / hour  {summary['false_alarms_per_hour']:.1f}"
              f"  (per hour of non-eating)")

    if participant_metrics:
        f1 = np.array([m["f1"] for m in participant_metrics.values()])
        summary["participant_f1_mean"] = float(f1.mean())
        summary["participant_f1_std"] = float(f1.std(ddof=0))
        summary["participant_f1_median"] = float(np.median(f1))
        summary["participants"] = int(len(f1))
        worst = sorted(participant_metrics.items(), key=lambda kv: kv[1]["f1"])[:3]
        print(f"\n  per participant ({len(f1)})")
        print(f"    F1 mean +- sd        {f1.mean():.4f} +- {f1.std(ddof=0):.4f}"
              f"   median {np.median(f1):.4f}")
        print("    lowest               " + ", ".join(
            f"{p} {m['f1']:.3f}" for p, m in worst))

    if extra:
        print()
        for key, value in extra.items():
            print(f"  {key}: {value}")
    return summary


# --------------------------------------------------------------------------- #
# per participant, and paired comparison between two runs
# --------------------------------------------------------------------------- #
def per_participant(y_true: np.ndarray, y_probability: np.ndarray,
                    participants: np.ndarray, threshold: float
                    ) -> Dict[str, Dict[str, float]]:
    """metrics() for each participant separately, at one threshold."""
    participants = np.asarray(participants)
    return {str(p): metrics(np.asarray(y_true)[participants == p],
                            np.asarray(y_probability)[participants == p],
                            threshold)
            for p in np.unique(participants)}


def _pooled_f1(runs: Sequence[Dict[str, Dict[str, float]]],
               keys: Sequence[str]) -> float:
    counts = {k: sum(runs[p][k] for p in keys) for k in ("tp", "fp", "fn", "tn")}
    return metrics_from_counts(counts)["f1"]


def paired_comparison(reference: Dict[str, Dict[str, float]],
                      candidate: Dict[str, Dict[str, float]],
                      n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Is `candidate` better than `reference` on the SAME participants?

    Two runs on the same participants are paired: each participant is scored
    by both, so the question is answered by the per-participant differences,
    not by comparing two fold means against their spread. This is what makes a
    difference of a few hundredths testable at n = 20 or more.

    Returns the mean per-participant F1 difference, how many participants
    improved or got worse, a Wilcoxon signed-rank p-value on those differences,
    and a bootstrap 95% interval for the difference in POOLED F1, resampling
    participants (not windows, which are not independent).
    """
    shared = sorted(set(reference) & set(candidate))
    if not shared:
        raise ValueError("the two runs share no participants")
    differences = np.array([candidate[p]["f1"] - reference[p]["f1"]
                            for p in shared])

    nonzero = differences[np.abs(differences) > 1e-12]
    if len(nonzero) == 0:
        p_value = 1.0
    else:
        from scipy.stats import wilcoxon
        p_value = float(wilcoxon(nonzero).pvalue)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for index in range(n_boot):
        sample = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
        boot[index] = (_pooled_f1(candidate, sample)
                       - _pooled_f1(reference, sample))

    return {
        "participants": len(shared),
        "mean_f1_difference": float(differences.mean()),
        "median_f1_difference": float(np.median(differences)),
        "improved": int(np.sum(differences > 1e-12)),
        "worse": int(np.sum(differences < -1e-12)),
        "unchanged": int(np.sum(np.abs(differences) <= 1e-12)),
        "wilcoxon_p": p_value,
        "pooled_f1_difference": _pooled_f1(candidate, shared)
                                - _pooled_f1(reference, shared),
        "pooled_f1_difference_ci_low": float(np.percentile(boot, 2.5)),
        "pooled_f1_difference_ci_high": float(np.percentile(boot, 97.5)),
    }

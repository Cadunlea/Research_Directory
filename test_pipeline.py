r"""
Tests for the data pipeline and evaluation, run without a database.

These exist because the failure modes they guard against are SILENT. A leak
between training and test participants, a window labelled from un-annotated
time, or a threshold chosen on the test fold all produce a higher score and no
error message - and a higher score is exactly what one is hoping for, so
nothing prompts a second look. Each of these is checked mechanically instead.

    python -m pytest test_pipeline.py -v
    python test_pipeline.py
"""

from __future__ import annotations

import pickle

import numpy as np

import ets_data
import ets_eval

FS = 128
GT_FS = 10
DAY_START = 1773100800
ANNOTATED_FROM = 12 * 3600
ANNOTATED_MINUTES = 16
EATING = [(60, 180), (300, 420), (600, 700)]


# --------------------------------------------------------------------------- #
# synthetic database
# --------------------------------------------------------------------------- #
def ground_truth(identifier: str) -> np.ndarray:
    day = np.full(86400 * GT_FS, ets_data.MISSING, dtype=np.int16)
    lo = ANNOTATED_FROM * GT_FS
    hi = (ANNOTATED_FROM + ANNOTATED_MINUTES * 60) * GT_FS
    day[lo:hi] = 0
    for start, end in EATING:
        begin = (ANNOTATED_FROM + start) * GT_FS
        finish = (ANNOTATED_FROM + end) * GT_FS
        if identifier == "BOGT":
            day[begin:finish] = 1
        elif identifier == "BIGT":
            day[begin:begin + GT_FS] = 1
        elif identifier == "CHGT":
            for moment in np.arange(start, end, 0.7):
                day[int((ANNOTATED_FROM + moment) * GT_FS)] = 1
    return day


def sensor_rows(participant: str) -> list:
    rng = np.random.default_rng(abs(hash(participant)) % 1000)
    rows = []
    for start in range(ANNOTATED_FROM - 8,
                       ANNOTATED_FROM + ANNOTATED_MINUTES * 60 + 8, 8):
        times = np.arange(1022) / FS
        envelope = np.zeros(1022)
        for begin, end in EATING:
            absolute = start + times
            envelope[(absolute >= ANNOTATED_FROM + begin) &
                     (absolute < ANNOTATED_FROM + end)] = 1
        optical = 3000 + 400 * envelope * np.sin(2 * np.pi * 1.4 * times)
        accelerometer = 400 + 50 * rng.standard_normal((3, 1022))
        packet = np.vstack([accelerometer, optical]).astype(np.int16)
        rows.append((DAY_START + start, pickle.dumps(packet)))
    return rows


PARTICIPANTS = [f"AIM20{index:04d}" for index in range(8)]
STUDY = "ETS_lunch"


def install_stub():
    packets = {p: sensor_rows(p) for p in PARTICIPANTS}

    def query(conn, sql, params=()):
        if "CHGT" in sql and ets_data.STUDY_DB in sql:
            return [(p, STUDY, identifier, DAY_START, 10.0,
                     pickle.dumps(ground_truth(identifier)))
                    for p in PARTICIPANTS
                    for identifier in ("CHGT", "BOGT", "BIGT")]
        if ets_data.RAW_DB in sql:
            participant, low, high = params[1], params[3], params[4]
            return [r for r in packets[participant] if low <= r[0] <= high]
        return []

    ets_data.query = query


def make_windows(window_seconds=8.0, hop_seconds=None):
    install_stub()
    sessions = ets_data.load_sessions(None)
    return ets_data.build_windows(None, sessions, window_seconds,
                                  hop_seconds or window_seconds, verbose=False)


def make_windows_with_context(window_seconds=8.0, hop_seconds=None,
                              context_seconds=0.0):
    install_stub()
    sessions = ets_data.load_sessions(None)
    return ets_data.build_windows(None, sessions, window_seconds,
                                  hop_seconds or window_seconds,
                                  context_seconds=context_seconds,
                                  verbose=False)


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def test_windows_have_the_expected_shape():
    windows = make_windows(8.0)
    assert windows.X.shape[1:] == (1024, 4)
    assert windows.X.dtype == np.float32
    assert set(np.unique(windows.y)) <= {0, 1}


def test_only_annotated_time_becomes_windows():
    """The day is 24 hours; only ~16 minutes are annotated. Windows must come
    from the annotated span alone - otherwise 98.9% of every record, which is
    -1, would arrive as fabricated 'not eating' and drown the real labels."""
    windows = make_windows(8.0)
    per_participant = len(windows) / len(PARTICIPANTS)
    expected = ANNOTATED_MINUTES * 60 / 8
    assert abs(per_participant - expected) <= 2, (
        f"{per_participant} windows per participant; annotated span allows "
        f"about {expected}. Un-annotated time is leaking in.")


def test_label_matches_the_eating_periods():
    windows = make_windows(8.0)
    eating_seconds = sum(end - start for start, end in EATING)
    expected = eating_seconds / (ANNOTATED_MINUTES * 60)
    actual = float(windows.y.mean())
    assert abs(actual - expected) < 0.06, (
        f"{actual:.3f} of windows labelled eating; the annotation says "
        f"{expected:.3f}")


def test_chew_counts_are_counts_not_states():
    """CHGT stores one flag per chew, so a window's value is a small count -
    not the number of samples the window spans."""
    windows = make_windows(8.0)
    eating = windows.chews[windows.y == 1]
    assert eating.max() < 8 * GT_FS, "chew counts look like durations"
    assert eating.mean() > 1.0, "eating windows should contain several chews"


def test_missing_sensor_data_is_excluded_from_normalisation():
    """-1 must not be treated as a reading. Left in, it is a large negative
    outlier against signals centred near 3000 counts."""
    window = np.full((1, 256, 4), 3000.0, dtype=np.float32)
    window[0, :64, 0] = ets_data.MISSING
    window[0, 64:, 0] = np.linspace(2900, 3100, 192)
    normalised = ets_data.normalise(window)
    assert np.all(np.abs(normalised[0, :64, 0]) < 1e-6), (
        "missing samples should sit at the channel mean (0), not at -1")
    assert abs(float(normalised[0, 64:, 0].mean())) < 1e-5


# --------------------------------------------------------------------------- #
# splitting - the leak-prone part
# --------------------------------------------------------------------------- #
def test_folds_are_disjoint_and_cover_everyone():
    folds = ets_data.participant_folds(np.array(PARTICIPANTS), 4, seed=9)
    combined = [p for fold in folds for p in fold]
    assert sorted(combined) == sorted(PARTICIPANTS)
    assert len(combined) == len(set(combined)), "a participant is in two folds"


def test_folds_are_deterministic_and_balanced():
    """The same participants must always give the same folds - no reliance on
    numpy's global random state, which any earlier call could have advanced -
    and the folds must be balanced, since at n=20 one participant is 5% of the
    data.

    Fold membership DOES change when participants are added; that is a
    deliberate trade for balance, documented in participant_folds. Results are
    therefore only comparable across runs with the same participant set, which
    is why the training scripts record fold membership in their output.
    """
    first = ets_data.participant_folds(np.array(PARTICIPANTS), 4, seed=9)
    np.random.random(100)                 # disturb the global random state
    second = ets_data.participant_folds(np.array(PARTICIPANTS), 4, seed=9)
    assert [list(f) for f in first] == [list(f) for f in second]

    sizes = [len(fold) for fold in first]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced folds: {sizes}"


def test_no_participant_appears_in_both_sides_of_a_split():
    windows = make_windows(8.0)
    for fold in ets_data.participant_folds(windows.participants, 4, seed=9):
        train, test = ets_data.fold_indices(windows, fold)
        overlap = (set(windows.participants[train]) &
                   set(windows.participants[test]))
        assert not overlap, f"leaked participants: {overlap}"
        assert len(train) + len(test) == len(windows)


def test_context_widens_the_signal_without_touching_the_labels():
    """--context-seconds must change ONLY what the model sees, never what it is
    scored against. If it shifted the labels or the window starts, every
    context run would be measured on a different target than the runs it is
    being compared with - and the comparison would be meaningless while still
    producing a plausible-looking number."""
    plain = make_windows(8.0)
    contextual = make_windows_with_context(8.0, context_seconds=8.0)

    assert contextual.X.shape[1] == plain.X.shape[1] * 3, (
        "8 s of context each side of an 8 s window should give 24 s of signal")
    assert len(contextual) == len(plain), "context changed the window count"
    assert np.array_equal(contextual.y, plain.y), "context changed the labels"
    assert np.allclose(contextual.starts, plain.starts), (
        "context moved the window starts, so smoothing and the "
        "non-overlapping grid would no longer line up")
    assert np.array_equal(contextual.chews, plain.chews)


def test_context_at_the_edges_is_missing_not_fabricated():
    """The first window has no real signal before it. That margin must arrive
    as -1 (missing), which normalise() neutralises - never as zeros, which the
    model would read as a genuine flat reading."""
    contextual = make_windows_with_context(8.0, context_seconds=8.0)
    first = contextual.X[np.argmin(contextual.starts)]
    lead_in = first[:1024]
    assert np.any(lead_in == ets_data.MISSING), (
        "the pre-roll of the first window should contain missing samples")


def test_overlapping_windows_are_halved_for_evaluation():
    """Training may use a half-window hop, but scoring must use the
    non-overlapping grid, or the same seconds are counted twice."""
    import ets_train
    windows = make_windows(8.0, 4.0)
    mask = ets_train.aligned_mask(windows)
    assert 0.4 < mask.mean() < 0.6, mask.mean()

    for participant in np.unique(windows.participants):
        rows = np.flatnonzero((windows.participants == participant) & mask)
        starts = np.sort(windows.starts[rows])
        gaps = np.diff(starts)
        assert np.all(gaps >= 8.0 - 1e-6), "aligned windows still overlap"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics_match_hand_computed_values():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    probability = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    result = ets_eval.metrics(y_true, probability, 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 4)
    assert abs(result["precision"] - 2 / 3) < 1e-9
    assert abs(result["recall"] - 2 / 3) < 1e-9
    assert abs(result["f1"] - 2 / 3) < 1e-9
    assert abs(result["accuracy"] - 6 / 8) < 1e-9


def test_accuracy_alone_would_hide_a_collapsed_model():
    """The reason F1 leads the report: a model that predicts one class scores
    well on accuracy when the classes are unbalanced."""
    y_true = np.zeros(100, dtype=int)
    y_true[:17] = 1                       # the least balanced real session
    always_negative = np.zeros(100)
    result = ets_eval.metrics(y_true, always_negative, 0.5)
    assert result["accuracy"] == 0.83
    assert result["f1"] == 0.0
    assert result["balanced_accuracy"] == 0.5


def test_smoothing_removes_an_isolated_spike():
    probability = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    starts = np.arange(6) * 8.0
    participants = np.array(["A"] * 6)
    smoothed = ets_eval.smooth_predictions(probability, starts, participants,
                                           8.0, 8.0, kernel=3)
    assert smoothed[2] < 0.5, "a lone positive window should be smoothed away"


def test_smoothing_keeps_a_real_bout():
    probability = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.1])
    starts = np.arange(6) * 8.0
    participants = np.array(["A"] * 6)
    smoothed = ets_eval.smooth_predictions(probability, starts, participants,
                                           8.0, 8.0, kernel=3)
    assert smoothed[2] > 0.5 and smoothed[3] > 0.5


def test_smoothing_does_not_bridge_a_recording_gap():
    """Two windows an hour apart must never be smoothed together."""
    probability = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    starts = np.array([0.0, 8.0, 16.0, 3600.0, 3608.0, 3616.0])
    participants = np.array(["A"] * 6)
    smoothed = ets_eval.smooth_predictions(probability, starts, participants,
                                           8.0, 8.0, kernel=3)
    assert smoothed[3] < 0.5, "the gap was bridged"


def test_threshold_is_chosen_to_maximise_the_objective():
    y_true = np.array([0] * 80 + [1] * 20)
    probability = np.concatenate([np.full(80, 0.2), np.full(20, 0.6)])
    threshold = ets_eval.choose_threshold(y_true, probability, "f1")
    assert 0.2 < threshold <= 0.6
    assert ets_eval.metrics(y_true, probability, threshold)["f1"] == 1.0


# --------------------------------------------------------------------------- #
# leave-one-subject-out and paired comparison
# --------------------------------------------------------------------------- #
def test_loso_folds_hold_out_exactly_one_participant_each():
    participants = np.array(PARTICIPANTS)
    folds = ets_data.participant_folds(participants, len(PARTICIPANTS), seed=9)
    assert all(len(fold) == 1 for fold in folds)
    assert sorted(p for fold in folds for p in fold) == sorted(PARTICIPANTS)


def test_inner_participants_never_include_the_test_participant():
    import ets_train
    for index, held_out in enumerate(
            ets_data.participant_folds(np.array(PARTICIPANTS), 8, seed=9)):
        train = [p for p in PARTICIPANTS if p not in held_out]
        for mode in ("first", "rotate"):
            inner = ets_train.choose_inner_participants(train, 2, index, 9, mode)
            assert len(inner) == 2
            assert not set(inner) & set(held_out), f"{mode} leaked the test fold"
            assert set(inner) <= set(train)


def test_first_selection_reproduces_the_original_behaviour():
    """Earlier 4-fold numbers came from np.unique(train)[:2]. 'first' must give
    exactly that, or those results stop being reproducible."""
    import ets_train
    train = np.array(PARTICIPANTS[::-1])
    got = ets_train.choose_inner_participants(train, 2, 0, 9, "first")
    assert list(got) == list(np.unique(train)[:2])


def test_rotating_selection_is_deterministic_and_actually_rotates():
    """Under LOSO, 'first' would give the same two people the threshold decision
    in nearly every fold. 'rotate' must spread that job around, and give the
    same answer on every run."""
    import ets_train
    chosen = []
    for index in range(len(PARTICIPANTS)):
        train = [p for i, p in enumerate(PARTICIPANTS) if i != index]
        a = ets_train.choose_inner_participants(train, 2, index, 9, "rotate")
        b = ets_train.choose_inner_participants(train, 2, index, 9, "rotate")
        assert list(a) == list(b), "rotation is not deterministic"
        chosen.append(tuple(a))
    assert len(set(chosen)) > 2, f"rotation barely varies: {set(chosen)}"


def test_per_participant_counts_add_up_to_the_fold():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    probability = np.array([0.9, 0.2, 0.4, 0.8, 0.6, 0.1, 0.7, 0.3])
    who = np.array(["A", "A", "A", "B", "B", "B", "C", "C"])
    parts = ets_eval.per_participant(y, probability, who, 0.5)
    whole = ets_eval.metrics(y, probability, 0.5)
    for key in ("tp", "tn", "fp", "fn", "n"):
        assert sum(m[key] for m in parts.values()) == whole[key], key


def _fake_run(f1_like):
    """Per-participant counts giving roughly the requested F1 each."""
    out = {}
    for index, quality in enumerate(f1_like):
        tp = int(round(40 * quality))
        out[f"P{index:02d}"] = dict(
            ets_eval.metrics_from_counts(
                {"tp": tp, "fn": 40 - tp, "fp": 40 - tp, "tn": 60}),
            tp=tp, fn=40 - tp, fp=40 - tp, tn=60, n=140, positive_rate=40 / 140)
    return out


def test_identical_runs_show_no_difference():
    run = _fake_run(np.linspace(0.6, 0.95, 20))
    result = ets_eval.paired_comparison(run, run, n_boot=500)
    assert result["mean_f1_difference"] == 0.0
    assert result["wilcoxon_p"] == 1.0
    assert result["pooled_f1_difference_ci_low"] <= 0.0 <= \
        result["pooled_f1_difference_ci_high"]


def test_a_consistent_improvement_is_detected():
    base = np.linspace(0.6, 0.9, 20)
    a, b = _fake_run(base), _fake_run(base + 0.05)
    result = ets_eval.paired_comparison(a, b, n_boot=500)
    assert result["improved"] == 20 and result["worse"] == 0
    assert result["wilcoxon_p"] < 0.01
    assert result["pooled_f1_difference_ci_low"] > 0.0


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"  PASS  {name}")
        except AssertionError as exc:                           # noqa: PERF203
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{'all tests pass' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)

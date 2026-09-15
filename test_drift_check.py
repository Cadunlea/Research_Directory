r"""
Tests for check_annotation_drift.py, run against synthetic data with a KNOWN
injected offset.

This matters more than a normal unit test. The drift check exists to answer
"is the 8 second sensor/annotation shift real?", and that answer will be
presented to the lab. A measurement instrument that has never been checked
against a known quantity cannot support a claim either way - so these tests
inject offsets whose true value is known and confirm the script recovers them.

They already caught one real bug: downsampling 128 Hz to 10 Hz by averaging
blocks of round(12.8) = 13 samples produced a 9.846 Hz series that the rest of
the code treated as 10 Hz. That error grows with elapsed time and reaches about
8 SECONDS by the middle of a 17 minute session - it would have manufactured
exactly the drift the script was written to measure.

    python -m pytest test_drift_check.py -v
    python test_drift_check.py              (runs without pytest too)
"""

from __future__ import annotations

import pickle

import numpy as np

import check_annotation_drift as drift

FS = int(drift.SENSOR_FS)
ANALYSIS_FS = int(drift.ANALYSIS_FS)
TOLERANCE_S = 1.0

DAY_START = 1773100800                      # midnight, as the real records store
ANNOTATED_FROM, ANNOTATED_TO = 12 * 3600, 12 * 3600 + 900       # 15 min of video
EATING_PERIODS = [(12 * 3600 + 120, 12 * 3600 + 300),
                  (12 * 3600 + 420, 12 * 3600 + 600)]


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def ground_truth_day(identifier: str) -> np.ndarray:
    """A full-day 10 Hz record: -1 outside the video, 0/1 inside it."""
    day = np.full(86400 * ANALYSIS_FS, drift.MISSING, dtype=np.int8)
    day[ANNOTATED_FROM * ANALYSIS_FS:ANNOTATED_TO * ANALYSIS_FS] = 0
    for start, end in EATING_PERIODS:
        if identifier in ("BOGT", "CHGT"):
            day[start * ANALYSIS_FS:end * ANALYSIS_FS] = 1
        elif identifier == "BIGT":
            day[start * ANALYSIS_FS:(start + 2) * ANALYSIS_FS] = 1
    return day


def sensor_packets(sensor_offset_s: float, stored_convention: str,
                   seed: int = 7) -> list:
    """8 s packets stored 4 x N, chewing at 1.5 Hz inside the eating periods,
    displaced by `sensor_offset_s`."""
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(ANNOTATED_FROM - 64, ANNOTATED_TO + 64, 8):
        times = np.arange(start, start + 8, 1.0 / FS)
        envelope = np.zeros(len(times))
        for begin, end in EATING_PERIODS:
            envelope[(times >= begin + sensor_offset_s) &
                     (times < end + sensor_offset_s)] = 1
        optical = (1000 + 300 * envelope * np.sin(2 * np.pi * 1.5 * times)
                   + 25 * rng.standard_normal(len(times)))
        accelerometer = 50 * rng.standard_normal((3, len(times)))
        packet = np.vstack([accelerometer, optical])
        timestamp = DAY_START + (start if stored_convention == "start"
                                 else start + 8)
        rows.append((timestamp, 8.0, pickle.dumps(packet)))
    return rows


def stub_query(sensor_offset_s: float, stored_convention: str):
    """Stands in for drift.query so the whole pipeline runs with no database."""
    packets = sensor_packets(sensor_offset_s, stored_convention)

    def query(conn, sql, params=()):
        if "CHGT" in sql and drift.STUDY_DB in sql:
            return [("AIM148647", "ETS_lunch", identifier, DAY_START, 10.0,
                     pickle.dumps(ground_truth_day(identifier)))
                    for identifier in ("BOGT", "BIGT", "CHGT")]
        if drift.RAW_DB in sql:
            low, high = params[2], params[3]
            return [row for row in packets if low <= row[0] <= high]
        return []

    return query


def measure(monkeypatch_target, sensor_offset_s: float, stored_convention: str,
            read_as: str):
    original = drift.query
    drift.query = stub_query(sensor_offset_s, stored_convention)
    try:
        sessions = drift.load_sessions(None, None)
        assert len(sessions) == 1
        return drift.measure_session(None, sessions[0], read_as, 30.0)
    finally:
        drift.query = original


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_recovers_zero_offset():
    """Aligned data must report ~0, or the script would invent a drift."""
    result = measure(None, 0.0, "start", "start")
    assert abs(result["best_lag_seconds"]) <= TOLERANCE_S
    assert result["best_correlation"] > 0.5


def test_recovers_known_offsets_in_both_directions():
    for offset in (-8.0, 8.0, 5.0, -12.0):
        result = measure(None, offset, "start", "start")
        assert abs(result["best_lag_seconds"] - offset) <= TOLERANCE_S, (
            f"injected {offset}, reported {result['best_lag_seconds']}")


def test_sign_convention_matches_the_old_hardcoded_shift():
    """Sensor activity arriving 8 s EARLY must report -8, the same number and
    sign as the old ANNOTATION_TIME_SHIFT_SECONDS = -8.0."""
    result = measure(None, -8.0, "start", "start")
    assert result["best_lag_seconds"] < 0
    assert abs(result["best_lag_seconds"] + 8.0) <= TOLERANCE_S


def test_packet_timestamp_convention_shifts_by_exactly_one_packet():
    """Reading correctly stored packets with the wrong convention must displace
    the result by one 8 s packet - the mechanism this script exists to expose."""
    correct = measure(None, 0.0, "start", "start")["best_lag_seconds"]
    wrong = measure(None, 0.0, "start", "end")["best_lag_seconds"]
    assert abs(correct) <= TOLERANCE_S
    assert abs((wrong - correct) + 8.0) <= TOLERANCE_S


def test_wrong_convention_can_masquerade_as_the_eight_second_drift():
    """Data stored END-stamped but read as START-stamped looks like a -8 s
    drift that is not in the data at all."""
    misread = measure(None, 0.0, "end", "start")["best_lag_seconds"]
    correct = measure(None, 0.0, "end", "end")["best_lag_seconds"]
    assert abs(correct) <= TOLERANCE_S
    assert abs(misread - 8.0) <= TOLERANCE_S


def test_downsampling_does_not_drift_over_a_long_session():
    """Guards the block-averaging bug directly: a constant chewing signal
    against a constant mask must not accumulate lag with session length."""
    minutes = 30
    samples = minutes * 60 * FS
    times = np.arange(samples) / FS
    optical = 1000 + 300 * np.sin(2 * np.pi * 1.5 * times)
    activity, valid = drift.chewing_activity(optical)
    expected = int(minutes * 60 * ANALYSIS_FS)
    assert abs(len(activity) - expected) <= 2, (
        f"{len(activity)} samples for {minutes} min; expected ~{expected}. "
        f"The 10 Hz grid is wrong, which would show up as a growing lag.")
    assert valid.all()


def test_missing_sensor_data_is_tolerated():
    """-1 samples must be excluded, not silently treated as a reading of -1."""
    rng = np.random.default_rng(3)
    minutes = 17
    samples = minutes * 60 * FS
    times = np.arange(samples) / FS
    envelope = ((times % 300) < 150).astype(float)
    optical = (1000 + 300 * envelope * np.sin(2 * np.pi * 1.5 * times)
               + 25 * rng.standard_normal(samples))
    optical[rng.random(samples) < 0.4] = drift.MISSING
    activity, valid = drift.chewing_activity(optical)
    assert activity.size > 0
    assert 0 < valid.mean() <= 1.0


def test_pure_noise_is_reported_as_unreliable():
    """No chewing means no lag to find; the script must say so rather than
    reporting whatever lag the noise happened to favour."""
    rng = np.random.default_rng(11)
    samples = 17 * 60 * FS
    optical = 1000 + 25 * rng.standard_normal(samples)
    activity, valid = drift.chewing_activity(optical)
    eating = np.zeros(17 * 60 * ANALYSIS_FS)
    eating[1200:3000] = 1.0
    lags = np.arange(-300, 301)
    correlations = drift.lagged_correlation(
        activity, valid, eating, np.ones(len(eating), bool), lags)
    peak = float(np.nanmax(correlations))
    assert peak < drift.MIN_TRUSTWORTHY_CORRELATION, (
        f"noise correlated at {peak:.3f}; the reliability threshold is too low")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"  PASS  {name}")
        except AssertionError as exc:                          # noqa: PERF203
            failures += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{'all tests pass' if not failures else f'{failures} test(s) failed'}")
    raise SystemExit(1 if failures else 0)

import numpy as np

import lumembrane


def test_version_is_pre_release():
    assert lumembrane.__version__ == "0.1.0a0"


def test_detect_peaks_finds_synthetic_events():
    trace = np.zeros(1_000)
    expected = np.array([100, 400, 800])
    trace[expected] = 10.0

    detected = lumembrane.detect_peaks(
        trace,
        fs=1_000,
        threshold_std=4.0,
        min_distance_ms=3.0,
    )

    np.testing.assert_array_equal(detected, expected)


def test_spike_triggered_average_uses_sample_aligned_time_axis():
    trace = np.zeros(200)
    peaks = np.array([50, 100, 150])
    trace[peaks] = 3.0

    time_ms, mean, sem, snippets = lumembrane.spike_triggered_average(
        trace,
        peaks,
        window_ms=(-2, 3),
        fs=1_000,
    )

    np.testing.assert_array_equal(time_ms, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert snippets.shape == (3, 5)
    assert mean[2] == 3.0
    np.testing.assert_allclose(sem, 0.0)

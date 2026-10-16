"""Synthetic Lumembrane quick start; no private or experimental data required."""

import numpy as np

from lumembrane import detect_peaks, estimate_firing_rate, spike_triggered_average


def main() -> None:
    sampling_rate = 1_000.0
    rng = np.random.default_rng(12)
    trace = rng.normal(0.0, 0.2, 4_000)
    expected = np.array([700, 1_600, 2_500, 3_300])
    trace[expected] += 5.0

    peaks = detect_peaks(trace, fs=sampling_rate, threshold_std=4.0)
    rate_hz, time_s = estimate_firing_rate(peaks, trace.size, fs=sampling_rate)
    time_ms, mean, sem, snippets = spike_triggered_average(
        trace,
        peaks,
        window_ms=(-20, 40),
        fs=sampling_rate,
    )

    print(f"detected {len(peaks)} spikes at samples {peaks.tolist()}")
    print(f"rate trace: {rate_hz.shape}, ending at {time_s[-1]:.3f} s")
    print(f"STA: {snippets.shape}, window {time_ms[0]:.1f} to {time_ms[-1]:.1f} ms")
    print(f"STA peak: {mean.max():.2f} ± {sem[mean.argmax()]:.2f}")


if __name__ == "__main__":
    main()

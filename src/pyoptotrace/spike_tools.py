"""Spike-train and spike-waveform analysis helpers for voltage imaging.

Consolidated 2026-08-26 from the F-IBMX comparison notebook
(260825_F-IBMX_two_cell_comparison.ipynb keeps its own inline copies for
reproducibility; new notebooks should import from here).

Conventions: traces are footprint-weighted MAD z-scores multiplied by -1
(Voltron2 inverted, spikes up); spike indices are sample indices at ``fs``.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def find_spike_bh(volt_hi, threshold, prom_th=0.0, min_interval=None):
    """Python translation of MATLAB find_spike_bh.m (template method, verbatim
    from the screening notebooks).

    Parameters
    ----------
    volt_hi : (T,) or (n_cells, T) array — voltage trace(s), spikes up
    threshold : float — minimum peak height
    prom_th : float — minimum local prominence (trace minus 7-sample moving
        median) at the peak
    min_interval : int or None — minimum peak spacing in samples

    Returns
    -------
    spike : binary spike array, same leading shape as input
    pks_all, prom_all : peak heights and local prominences
    """
    volt_hi = np.asarray(volt_hi, dtype=float)
    was_1d = volt_hi.ndim == 1
    if was_1d:
        volt_hi = volt_hi[np.newaxis, :]

    n_cells, T = volt_hi.shape
    spike = np.zeros((n_cells, T), dtype=np.uint8)
    pks_all = []
    prom_all = []

    for i in range(n_cells):
        trace = volt_hi[i]

        if min_interval is not None:
            peak_idx, props = find_peaks(trace, height=threshold, distance=min_interval)
            pks = props["peak_heights"]
        else:
            peak_idx, _ = find_peaks(trace)
            pks = trace[peak_idx]

        trace_hp = trace - median_filter(trace, size=7, mode="nearest")
        prom = trace_hp[peak_idx]

        keep = (pks > threshold) & (prom > prom_th)
        peak_idx = peak_idx[keep]
        pks = pks[keep]
        prom = prom[keep]

        spike[i, peak_idx] = 1
        pks_all.append(pks)
        prom_all.append(prom)

    if was_1d:
        return spike[0], pks_all[0], prom_all[0]
    return spike, pks_all, prom_all


def spike_metrics(spike_idx, n_frames, fs=1000.0, burst_isi_ms=15.0):
    """Rate, ISI, and burst statistics from spike sample indices.

    A burst is a run of consecutive ISIs < burst_isi_ms; a run of k short
    ISIs is one burst containing k+1 spikes. Also returns threshold-free
    burstiness indices (fraction of ISIs below 6 / 15 ms).
    """
    duration_s = n_frames / fs
    n = len(spike_idx)
    m = {
        "n_spikes": n,
        "duration_s": duration_s,
        "rate_hz": n / duration_s,
        "median_isi_ms": np.nan,
        "mean_isi_ms": np.nan,
        "cv_isi": np.nan,
        "frac_isi_lt6": np.nan,
        "frac_isi_lt15": np.nan,
        "n_bursts": 0,
        "burst_rate_hz": 0.0,
        "spikes_per_burst": np.nan,
        "frac_spikes_in_bursts": 0.0,
        "intraburst_rate_hz": np.nan,
    }
    if n < 2:
        return m
    isi_ms = np.diff(spike_idx) / fs * 1000.0
    m["median_isi_ms"] = float(np.median(isi_ms))
    m["mean_isi_ms"] = float(np.mean(isi_ms))
    m["cv_isi"] = float(np.std(isi_ms) / np.mean(isi_ms))
    m["frac_isi_lt6"] = float(np.mean(isi_ms < 6.0))
    m["frac_isi_lt15"] = float(np.mean(isi_ms < 15.0))

    short = isi_ms < burst_isi_ms
    if short.any():
        edges = np.diff(np.concatenate(([0], short.astype(int), [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        run_lens = stops - starts  # k short ISIs per burst
        m["n_bursts"] = int(len(run_lens))
        m["burst_rate_hz"] = len(run_lens) / duration_s
        m["spikes_per_burst"] = float(np.mean(run_lens + 1))
        m["frac_spikes_in_bursts"] = float(np.sum(run_lens + 1)) / n
        m["intraburst_rate_hz"] = float(1000.0 / np.mean(isi_ms[short]))
    return m


def classify_spikes(spike_idx, fs=1000.0, burst_isi_ms=15.0):
    """Split spikes into burst members vs isolated singles.

    A spike is part of a burst if the ISI to its previous OR next spike is
    < burst_isi_ms. Returns (burst_idx, single_idx, burst_first_idx); the
    first spike of each burst is the only burst spike with a clean pre-window.
    """
    spike_idx = np.asarray(spike_idx)
    if len(spike_idx) < 2:
        return spike_idx[:0], spike_idx.copy(), spike_idx[:0]
    short = (np.diff(spike_idx) / fs * 1000.0) < burst_isi_ms
    prev_short = np.concatenate(([False], short))
    next_short = np.concatenate((short, [False]))
    in_burst = prev_short | next_short
    burst_first = next_short & ~prev_short
    return spike_idx[in_burst], spike_idx[~in_burst], spike_idx[burst_first]


def burst_runs(spike_idx, fs=1000.0, burst_isi_ms=15.0):
    """List of spike-index arrays, one per burst (size >= 2)."""
    sp = np.asarray(spike_idx)
    if len(sp) < 2:
        return []
    short = (np.diff(sp) / fs * 1000.0) < burst_isi_ms
    runs, start = [], None
    for i, s in enumerate(short):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append(sp[start : i + 1])
            start = None
    if start is not None:
        runs.append(sp[start:])
    return runs


def isolated_singles(spike_idx, fs=1000.0, iso_ms=30.0):
    """Spikes with no neighbor within iso_ms on either side.

    >= 30 ms keeps the previous spike's peak AND its afterpotential out of a
    [-20, -10] ms pre-spike baseline window at 1 kHz.
    """
    spike_idx = np.asarray(spike_idx)
    if len(spike_idx) < 2:
        return spike_idx.copy()
    gap_ms = np.diff(spike_idx) / fs * 1000.0
    prev_ok = np.concatenate(([True], gap_ms >= iso_ms))
    next_ok = np.concatenate((gap_ms >= iso_ms, [True]))
    return spike_idx[prev_ok & next_ok]


def sta_from_spikes(trace, spike_idx, fs=1000.0, window_ms=(-20.0, 40.0)):
    """Spike-triggered snippets on an exact 1-sample grid (t=0 at the spike).

    Uses arange so the time axis is exactly fs-spaced (the linspace variant
    in utils/spike_analysis.py stretches the axis by one sample).
    Returns (t_ms, snippets) or (None, None) if no spike fits the window.
    """
    spike_idx = np.asarray(spike_idx)
    pre = int(round(abs(window_ms[0]) * fs / 1000.0))
    post = int(round(abs(window_ms[1]) * fs / 1000.0))
    idx = spike_idx[(spike_idx >= pre) & (spike_idx < len(trace) - post)]
    if len(idx) == 0:
        return None, None
    snippets = np.stack([trace[i - pre : i + post + 1] for i in idx])
    t_ms = np.arange(-pre, post + 1) / fs * 1000.0
    return t_ms, snippets


def fwhm_ms(t_ms, w, baseline_t_max=-10.0):
    """Full width at half max of a mean waveform (baseline: t < baseline_t_max).

    Feed a stats-restricted window (e.g. t <= +5 ms) when the snippet can
    contain neighboring spikes further out.
    """
    base = w[t_ms < baseline_t_max].mean()
    peak_i = int(np.argmax(w))
    half = base + (w[peak_i] - base) / 2.0
    above = w > half
    left = peak_i
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_i
    while right < len(w) - 1 and above[right + 1]:
        right += 1

    def _cross(i_above, i_below):
        if w[i_above] == w[i_below]:
            return t_ms[i_above]
        frac = (half - w[i_above]) / (w[i_below] - w[i_above])
        return t_ms[i_above] + frac * (t_ms[i_below] - t_ms[i_above])

    t_left = _cross(left, left - 1) if left > 0 else t_ms[0]
    t_right = _cross(right, right + 1) if right < len(w) - 1 else t_ms[-1]
    return float(abs(t_right - t_left))


def per_spike_after_potential(
    trace, spike_idx, fs=1000.0, iso_ms=30.0, sta_window_ms=(-20.0, 35.0), ap_window_ms=(2.0, 20.0)
):
    """Per-spike after-potential (window mean minus pre-spike baseline) for
    well-isolated single spikes.

    Negative values = afterhyperpolarization (AHP, interneuron-typical);
    positive = afterdepolarization (ADP, CA1-pyramidal-typical).
    Returns a 1-D array (possibly empty).
    """
    iso = isolated_singles(spike_idx, fs=fs, iso_ms=iso_ms)
    t_ms, snips = sta_from_spikes(trace, iso, fs=fs, window_ms=sta_window_ms)
    if snips is None:
        return np.empty(0, dtype=np.float32)
    base = snips[:, t_ms < -10].mean(axis=1)
    w0, w1 = ap_window_ms
    ap = snips[:, (t_ms >= w0) & (t_ms <= w1)].mean(axis=1) - base
    return ap.astype(np.float32)

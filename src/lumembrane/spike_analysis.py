"""
spike_analysis.py
-----------------
Spike detection, classification, firing rate estimation, and spike-triggered
averaging for voltage imaging traces and movies.

Public API
----------
detect_peaks(trace, fs, threshold_std, min_distance_ms, **kwargs)
    -> peak_indices

classify_spikes(trace, peak_idx, cs_max_isi_ms, cs_min_spikes, cs_min_width_ms, fs)
    -> ss_idx, cs_idx, labels

estimate_firing_rate(peak_idx, n_frames, sigma_ms, fs)
    -> rate_hz, time_s

spike_triggered_average(trace, peak_indices, window_ms, fs)
    -> times, mean_trace, sem_trace, snippets

spike_triggered_average_movie(movie, peak_indices, window_ms, fs)
    -> times, sta_movie

analyze_roi(trace, roi_label, threshold_std, fs, verbose)
    -> dict

plot_spike_average(trace, peak_indices, window_ms, fs, title, ax, ...)
    -> fig, ax

plot_spike_summary(traces, roi_labels, threshold_sd, xlim_s, figsize, fs)
    -> fig, results

plot_cs_zoom(trace, cs_idx, window_ms, n_events, figsize, fs)
    -> fig
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# ── constants ─────────────────────────────────────────────────────────────────
FS = 1000.0  # Hz — default sampling rate
CS_MAX_ISI_MS = 10  # ms: ISIs shorter than this → complex spike burst
CS_MIN_SPIKES = 2  # min components to call a complex spike
CS_MIN_WIDTH_MS = 3  # ms: isolated peak width above which it's flagged CS


# ── 1. peak detection ─────────────────────────────────────────────────────────


def detect_peaks(
    trace: np.ndarray,
    fs: float = FS,
    threshold_std: float = 4.0,
    min_distance_ms: float = 3.0,
    **find_peaks_kwargs,
) -> np.ndarray:
    """
    Detect peaks that exceed mean + threshold_std * std.

    Parameters
    ----------
    trace : 1-D array
    fs : float
    threshold_std : float
    min_distance_ms : float
    **find_peaks_kwargs : forwarded to scipy.signal.find_peaks

    Returns
    -------
    peak_indices : np.ndarray of int
    """
    trace = np.asarray(trace, dtype=float)
    threshold = trace.mean() + threshold_std * trace.std()
    min_distance_samples = int(np.round(min_distance_ms * fs / 1000.0))
    peak_indices, _ = find_peaks(
        trace,
        height=threshold,
        distance=max(1, min_distance_samples),
        **find_peaks_kwargs,
    )
    return peak_indices


# ── 2. complex spike classification ──────────────────────────────────────────


def classify_spikes(
    trace: np.ndarray,
    peak_idx: np.ndarray,
    cs_max_isi_ms: float = CS_MAX_ISI_MS,
    cs_min_spikes: int = CS_MIN_SPIKES,
    cs_min_width_ms: float = CS_MIN_WIDTH_MS,
    fs: float = FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify each detected peak as simple spike (SS) or complex spike (CS).

    Uses two passes:
      1. ISI — consecutive peaks with ISI < cs_max_isi_ms and burst length
               >= cs_min_spikes are labelled CS.
      2. Width — isolated peaks with half-prominence width > cs_min_width_ms
                 are also labelled CS.

    Returns
    -------
    ss_idx  : np.ndarray of int  — simple spike sample indices
    cs_idx  : np.ndarray of int  — first-component index per complex spike
    labels  : np.ndarray of str  — 'SS' or 'CS' per entry in peak_idx
    """
    if len(peak_idx) == 0:
        return np.array([], int), np.array([], int), np.array([], dtype="<U2")

    labels = np.array(["SS"] * len(peak_idx), dtype="<U2")
    cs_max_isi_samp = cs_max_isi_ms * fs / 1000
    cs_min_width_samp = cs_min_width_ms * fs / 1000

    # ISI pass
    isis = np.diff(peak_idx)
    in_burst = isis < cs_max_isi_samp
    i = 0
    while i < len(in_burst):
        if in_burst[i]:
            j = i
            while j < len(in_burst) and in_burst[j]:
                j += 1
            burst_indices = list(range(i, j + 1))
            if len(burst_indices) >= cs_min_spikes:
                for k in burst_indices:
                    labels[k] = "CS"
            i = j + 1
        else:
            i += 1

    # Width pass
    widths_samp, _, _, _ = peak_widths(trace, peak_idx, rel_height=0.5)
    labels[widths_samp > cs_min_width_samp] = "CS"

    ss_idx = peak_idx[labels == "SS"]
    cs_all = peak_idx[labels == "CS"]

    # Collapse burst components → first component only
    cs_first = []
    if len(cs_all) > 0:
        cs_first.append(cs_all[0])
        for k in range(1, len(cs_all)):
            if (cs_all[k] - cs_all[k - 1]) > cs_max_isi_samp * 2:
                cs_first.append(cs_all[k])
    cs_first = np.array(cs_first, dtype=int)

    return ss_idx, cs_first, labels


# ── 3. firing rate estimation ─────────────────────────────────────────────────


def estimate_firing_rate(
    peak_idx: np.ndarray,
    n_frames: int,
    sigma_ms: float = 100.0,
    fs: float = FS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian kernel-smoothed firing rate from spike times.

    Returns
    -------
    rate_hz : np.ndarray, shape (n_frames,)
    time_s  : np.ndarray, shape (n_frames,)
    """
    spike_train = np.zeros(n_frames)
    if len(peak_idx) > 0:
        spike_train[peak_idx] = 1.0
    sigma_samp = sigma_ms * fs / 1000
    rate_hz = gaussian_filter1d(spike_train, sigma=sigma_samp) * fs
    time_s = np.arange(n_frames) / fs
    return rate_hz, time_s


# ── 4. spike-triggered average (trace) ───────────────────────────────────────


def spike_triggered_average(
    trace: np.ndarray,
    peak_indices: np.ndarray,
    window_ms: tuple[float, float] = (-50.0, 100.0),
    fs: float = FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract trace snippets centred on each peak and return mean ± SEM.

    Parameters
    ----------
    trace        : 1-D array
    peak_indices : array-like of int
    window_ms    : (pre_ms, post_ms)  pre_ms ≤ 0, post_ms ≥ 0
    fs           : float

    Returns
    -------
    times      : np.ndarray — time axis in ms relative to peak
    mean_trace : np.ndarray
    sem_trace  : np.ndarray
    snippets   : np.ndarray, shape (n_events, window_samples)
    """
    trace = np.asarray(trace, dtype=float)
    peak_indices = np.asarray(peak_indices, dtype=int)

    pre_samples = int(np.round(abs(window_ms[0]) * fs / 1000.0))
    post_samples = int(np.round(abs(window_ms[1]) * fs / 1000.0))
    times = np.arange(-pre_samples, post_samples, dtype=float) * 1000.0 / fs

    snippets = []
    for idx in peak_indices:
        start, stop = idx - pre_samples, idx + post_samples
        if start < 0 or stop > len(trace):
            continue
        snippets.append(trace[start:stop])

    if len(snippets) == 0:
        raise ValueError("No valid snippets — check window bounds or peak indices.")

    snippets = np.vstack(snippets)
    mean_trace = snippets.mean(axis=0)
    sem_trace = snippets.std(axis=0) / np.sqrt(len(snippets))
    return times, mean_trace, sem_trace, snippets


# ── 5. spike-triggered average (movie) ───────────────────────────────────────


def spike_triggered_average_movie(
    movie: np.ndarray,
    peak_indices: np.ndarray,
    window_ms: tuple[float, float] = (-50.0, 100.0),
    fs: float = FS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spike-triggered average of a full (T, H, W) movie.

    Parameters
    ----------
    movie        : np.ndarray, shape (n_frames, height, width)
    peak_indices : array-like of int
    window_ms    : (pre_ms, post_ms)
    fs           : float

    Returns
    -------
    times     : np.ndarray — time axis in ms relative to peak
    sta_movie : np.ndarray, shape (window_frames, height, width)
    """
    movie = np.asarray(movie)
    peak_indices = np.asarray(peak_indices, dtype=int)

    pre_samples = int(np.round(abs(window_ms[0]) * fs / 1000.0))
    post_samples = int(np.round(abs(window_ms[1]) * fs / 1000.0))
    n_samples = pre_samples + post_samples
    times = np.linspace(window_ms[0], window_ms[1], n_samples)

    snippets = []
    n_frames = movie.shape[0]
    for idx in peak_indices:
        start, stop = idx - pre_samples, idx + post_samples
        if start < 0 or stop > n_frames:
            continue
        snippets.append(movie[start:stop, :, :])

    if len(snippets) == 0:
        raise ValueError("No valid snippets — check window bounds or peak indices.")

    snippets = np.stack(snippets, axis=0)  # (n_events, window_frames, H, W)
    sta_movie = snippets.mean(axis=0)  # (window_frames, H, W)
    return times, sta_movie


# ── 6. per-ROI analysis ───────────────────────────────────────────────────────


def analyze_roi(
    trace: np.ndarray,
    roi_label: str = "ROI",
    threshold_std: float = 4.0,
    fs: float = FS,
    verbose: bool = True,
) -> dict:
    """
    Run full detection → classification → firing rate for one ROI.

    Returns
    -------
    dict with keys: peak_idx, ss_idx, cs_idx, labels, rate_hz, time_s
    """
    peak_idx = detect_peaks(trace, fs=fs, threshold_std=threshold_std)
    ss_idx, cs_idx, labels = classify_spikes(trace, peak_idx, fs=fs)
    rate_hz, time_s = estimate_firing_rate(peak_idx, len(trace), fs=fs)

    if verbose:
        duration_s = len(trace) / fs
        print(
            f"{roi_label}: {len(ss_idx)} SS  |  {len(cs_idx)} CS  |  "
            f"mean rate {len(peak_idx) / duration_s:.2f} Hz  "
            f"({duration_s:.1f} s recording)"
        )

    return dict(
        peak_idx=peak_idx,
        ss_idx=ss_idx,
        cs_idx=cs_idx,
        labels=labels,
        rate_hz=rate_hz,
        time_s=time_s,
    )


# ── 7. plot: spike-triggered average (trace) ─────────────────────────────────


def plot_spike_average(
    trace: np.ndarray,
    peak_indices: np.ndarray,
    window_ms: tuple[float, float] = (-50.0, 100.0),
    fs: float = FS,
    title: str = "Spike-Triggered Average",
    ax: plt.Axes | None = None,
    color: str = "#2196F3",
    individual_alpha: float = 0.15,
    show_individuals: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot spike-triggered average with ± SEM shading and optional individual
    trace overlays.
    """
    import matplotlib.pyplot as plt

    times, mean_tr, sem_tr, snippets = spike_triggered_average(
        trace, peak_indices, window_ms=window_ms, fs=fs
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    if show_individuals and individual_alpha > 0:
        for snip in snippets:
            ax.plot(times, snip, color=color, alpha=individual_alpha, linewidth=0.8)

    ax.fill_between(
        times, mean_tr - sem_tr, mean_tr + sem_tr, color=color, alpha=0.25, label="± SEM"
    )
    ax.plot(times, mean_tr, color=color, linewidth=2.0, label=f"Mean (n={len(snippets)})")
    ax.axvline(0, color="0.4", linewidth=0.8, linestyle="--", label="Peak")

    ax.set_xlabel("Time relative to peak (ms)")
    ax.set_ylabel("ΔF/F")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ── 8. plot: multi-ROI summary ────────────────────────────────────────────────


def plot_spike_summary(
    traces: np.ndarray,
    roi_labels: list[str] | None = None,
    threshold_sd: float = 4.0,
    xlim_s: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    fs: float = FS,
) -> tuple[plt.Figure, list]:
    """
    One row per ROI: ΔF/F trace with SS/CS markers, smoothed firing rate on
    twin axis, and a spike-count bar chart.

    Parameters
    ----------
    traces     : (n_frames, n_rois) or (n_frames,)
    roi_labels : list of str
    threshold_sd : float
    xlim_s     : (t_start, t_end) in seconds
    figsize    : or None (auto)
    fs         : float

    Returns
    -------
    fig, results  — results is a list of dicts from analyze_roi
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    if traces.ndim == 1:
        traces = traces[:, np.newaxis]

    n_rois = traces.shape[1]
    roi_labels = roi_labels or [f"ROI {i + 1}" for i in range(n_rois)]
    figsize = figsize or (14, 2.8 * n_rois)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rois, 2, width_ratios=[4, 1], hspace=0.5, wspace=0.35)
    results = []

    for i, label in enumerate(roi_labels):
        trace = traces[:, i]
        res = analyze_roi(trace, roi_label=label, threshold_std=threshold_sd, fs=fs, verbose=True)
        results.append(res)

        ax_trace = fig.add_subplot(gs[i, 0])
        ax_rate = ax_trace.twinx()
        t = res["time_s"]

        # firing rate
        ax_rate.fill_between(t, res["rate_hz"], alpha=0.12, color="#888888")
        ax_rate.set_ylabel("rate (Hz)", fontsize=7, color="#888888")
        ax_rate.tick_params(axis="y", labelsize=7, colors="#888888")
        ax_rate.set_ylim(bottom=0)

        # trace + threshold
        ax_trace.plot(t, trace, lw=0.6, color="#2a2a2a", zorder=3)
        ax_trace.axhline(np.std(trace) * threshold_sd, color="#aaaaaa", lw=0.8, ls="--", zorder=2)

        # SS markers
        if len(res["ss_idx"]) > 0:
            ax_trace.scatter(
                t[res["ss_idx"]],
                trace[res["ss_idx"]],
                s=18,
                color="#1f77b4",
                zorder=5,
                label="SS",
                marker="|",
                linewidths=1.2,
            )

        # CS markers
        if len(res["cs_idx"]) > 0:
            ax_trace.scatter(
                t[res["cs_idx"]],
                trace[res["cs_idx"]] * 1.05,
                s=40,
                color="#d62728",
                zorder=6,
                label="CS",
                marker="v",
            )

        if xlim_s is not None:
            ax_trace.set_xlim(xlim_s)
            ax_rate.set_xlim(xlim_s)

        ax_trace.set_ylabel("ΔF/F", fontsize=8)
        ax_trace.set_title(label, fontsize=9, fontweight="bold", loc="left")
        if i == n_rois - 1:
            ax_trace.set_xlabel("time (s)", fontsize=8)
        ax_trace.legend(fontsize=7, loc="upper right", framealpha=0.7, markerscale=1.2)

        # count bar
        ax_bar = fig.add_subplot(gs[i, 1])
        counts = [len(res["ss_idx"]), len(res["cs_idx"])]
        colors = ["#1f77b4", "#d62728"]
        bars = ax_bar.bar(
            ["SS", "CS"], counts, color=colors, width=0.5, edgecolor="white", linewidth=0.8
        )
        for bar, count in zip(bars, counts):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax_bar.set_ylabel("count", fontsize=8)
        ax_bar.set_title(label, fontsize=9, fontweight="bold", loc="left")
        ax_bar.set_ylim(0, max(counts) * 1.3 + 1)
        ax_bar.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Spike & Complex Spike Summary", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, results


# ── 9. plot: complex spike zoom ───────────────────────────────────────────────


def plot_cs_zoom(
    trace: np.ndarray,
    cs_idx: np.ndarray,
    window_ms: float = 80.0,
    n_events: int = 6,
    figsize: tuple[float, float] | None = None,
    fs: float = FS,
) -> plt.Figure | None:
    """
    Overlay individual complex spike waveforms + mean for visual inspection.

    Parameters
    ----------
    trace     : 1-D array
    cs_idx    : np.ndarray of int
    window_ms : half-window around each event (ms)
    n_events  : max events to overlay
    """
    import matplotlib.pyplot as plt

    if len(cs_idx) == 0:
        print("No complex spikes detected.")
        return None

    half = int(window_ms * fs / 1000)
    t_ms = np.linspace(-window_ms, window_ms, 2 * half)
    events = [
        cs_idx[i]
        for i in range(min(n_events, len(cs_idx)))
        if cs_idx[i] - half >= 0 and cs_idx[i] + half < len(trace)
    ]

    figsize = figsize or (5, 3.5)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.plasma(np.linspace(0.2, 0.85, len(events)))

    for j, idx in enumerate(events):
        ax.plot(t_ms, trace[idx - half : idx + half], lw=1.0, alpha=0.75, color=cmap[j])

    snippets = np.array([trace[idx - half : idx + half] for idx in events])
    ax.plot(
        t_ms,
        snippets.mean(axis=0),
        lw=2.0,
        color="black",
        label=f"mean (n={len(events)})",
        zorder=5,
    )

    ax.axvline(0, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("time re. CS onset (ms)", fontsize=9)
    ax.set_ylabel("ΔF/F", fontsize=9)
    ax.set_title("Complex spike waveforms", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

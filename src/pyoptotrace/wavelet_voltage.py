"""
wavelet_voltage.py
Wavelet-based decomposition of voltage imaging traces into spike,
subthreshold, and residual (neuropil/noise) bands.

Usage:
    import wavelet_voltage as wv

    wd = wv.decompose(results[roi]['trace_z'], hz=1000.0)
    wv.inspect(wd, n_frames=10000)

    # Spike-triggered average on subthreshold band only:
    wv.spike_triggered_average(wd, threshold=3.0, window_ms=20, hz=1000.0)

Drop into your utils/ folder alongside signal_extraction.py, or use standalone.
"""

import numpy as np
import pywt
from scipy.ndimage import median_filter


def _mad_zscore(trace: np.ndarray, window: int = 3000) -> np.ndarray:
    """MAD z-score (duplicated here to keep module standalone)."""
    baseline = median_filter(trace, size=(window,))
    residual = trace - baseline
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = mad / 0.6745 + 1e-10
    return residual / sigma


def decompose(
    trace: np.ndarray,
    hz: float = 1000.0,
    wavelet: str = "sym4",
    spike_levels: list = None,
    subthresh_levels: list = None,
    max_level: int = None,
) -> dict:
    """
    Decompose voltage trace into spike, subthreshold, and residual bands
    using the Stationary Wavelet Transform (SWT).

    SWT is used instead of DWT because it is shift-invariant — spike
    timing is preserved exactly regardless of alignment.

    Parameters
    ----------
    trace : 1D array, typically the z-scored voltage trace
    hz : sampling rate in Hz
    wavelet : wavelet family ('sym4', 'db4', 'sym6' are good choices)
    spike_levels : detail levels for spike band (default: auto from hz)
    subthresh_levels : detail levels for subthreshold band (default: auto)
    max_level : deepest decomposition level (default: auto)

    Returns
    -------
    dict with keys:
        spike_trace, subthresh_trace, residual, coeffs,
        spike_levels, subthresh_levels, level_info, hz, wavelet
    """
    # ── Auto-assign max_level ─────────────────────────────────────────────
    if max_level is None:
        max_level = int(np.floor(np.log2(hz))) - 1  # ~9 for 1 kHz

    # ── Pad to multiple of 2**max_level for SWT ──────────────────────────
    n = len(trace)
    factor = 2**max_level
    pad_len = (factor - (n % factor)) % factor
    if pad_len > 0:
        trace_pad = np.pad(trace, (0, pad_len), mode="reflect")
    else:
        trace_pad = trace.copy()

    print(f"  Trace length: {n}, padded to {len(trace_pad)} (factor={factor}, pad={pad_len})")

    # ── Auto-assign frequency bands if not specified ──────────────────────
    if spike_levels is None or subthresh_levels is None:
        spike_levels = []
        subthresh_levels = []
        for lev in range(1, max_level + 1):
            fhi = hz / 2**lev
            flo = hz / 2 ** (lev + 1)
            if flo >= 30:
                spike_levels.append(lev)
            elif flo >= 2:
                subthresh_levels.append(lev)

    # ── SWT decomposition ─────────────────────────────────────────────────
    coeffs = pywt.swt(trace_pad, wavelet, level=max_level)

    # ── Reconstruct spike band ────────────────────────────────────────────
    spike_coeffs = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
    for lev in spike_levels:
        idx = lev - 1
        spike_coeffs[idx] = (spike_coeffs[idx][0], coeffs[idx][1])
    spike_trace = pywt.iswt(spike_coeffs, wavelet)[:n]

    # ── Reconstruct subthreshold band ─────────────────────────────────────
    sub_coeffs = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
    for lev in subthresh_levels:
        idx = lev - 1
        sub_coeffs[idx] = (sub_coeffs[idx][0], coeffs[idx][1])
    subthresh_trace = pywt.iswt(sub_coeffs, wavelet)[:n]

    # ── Residual = everything else ────────────────────────────────────────
    residual = trace - spike_trace - subthresh_trace

    # ── Build frequency info table ────────────────────────────────────────
    level_info = []
    for lev in range(1, max_level + 1):
        fhi = hz / 2**lev
        flo = hz / 2 ** (lev + 1)
        band = (
            "spike"
            if lev in spike_levels
            else "subthresh"
            if lev in subthresh_levels
            else "residual"
        )
        level_info.append({"level": lev, "f_lo": flo, "f_hi": fhi, "band": band})
        print(f"  Level {lev:2d}: {flo:7.1f} – {fhi:7.1f} Hz  [{band}]")

    return {
        "spike_trace": spike_trace,
        "subthresh_trace": subthresh_trace,
        "residual": residual,
        "coeffs": coeffs,
        "spike_levels": spike_levels,
        "subthresh_levels": subthresh_levels,
        "level_info": level_info,
        "hz": hz,
        "wavelet": wavelet,
        "original": trace,
    }


def inspect(wd: dict, n_frames: int = 10000, invert: bool = True):
    """
    Plot the decomposed bands for visual inspection.
    """
    import matplotlib.pyplot as plt

    sign = -1 if invert else 1
    n = min(n_frames, len(wd["original"]))

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(wd["original"][:n] * sign, lw=0.4, color="green")
    axes[0].set_title("Full trace (z-scored)", fontsize=10)
    axes[0].axhline(3, color="r", lw=0.5, ls="--", alpha=0.5)

    axes[1].plot(wd["spike_trace"][:n] * sign, lw=0.4, color="steelblue")
    spike_bands = [
        f"{d['f_lo']:.0f}–{d['f_hi']:.0f}" for d in wd["level_info"] if d["band"] == "spike"
    ]
    axes[1].set_title(f"Spike band ({', '.join(spike_bands)} Hz)", fontsize=10)

    axes[2].plot(wd["subthresh_trace"][:n] * sign, lw=0.4, color="purple")
    sub_bands = [
        f"{d['f_lo']:.0f}–{d['f_hi']:.0f}" for d in wd["level_info"] if d["band"] == "subthresh"
    ]
    axes[2].set_title(f"Subthreshold band ({', '.join(sub_bands)} Hz)", fontsize=10)

    axes[3].plot(wd["residual"][:n] * sign, lw=0.4, color="orange")
    axes[3].set_title("Residual (neuropil + slow drift)", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Frames")

    plt.tight_layout()
    plt.show()

    return fig


def inspect_detail_levels(wd: dict, n_frames: int = 10000, invert: bool = True):
    """
    Plot each individual wavelet detail level separately.
    Useful for tuning which levels go in spike vs subthreshold.
    """
    import matplotlib.pyplot as plt

    sign = -1 if invert else 1
    n = min(n_frames, len(wd["original"]))
    coeffs = wd["coeffs"]
    info = wd["level_info"]
    n_levels = len(info)

    fig, axes = plt.subplots(n_levels + 1, 1, figsize=(16, 2.5 * (n_levels + 1)), sharex=True)

    axes[0].plot(wd["original"][:n] * sign, lw=0.4, color="green")
    axes[0].set_title("Full trace", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = {"spike": "steelblue", "subthresh": "purple", "residual": "orange"}

    for i, d in enumerate(info):
        lev = d["level"]
        single = [(np.zeros_like(cA), np.zeros_like(cD)) for cA, cD in coeffs]
        single[lev - 1] = (single[lev - 1][0], coeffs[lev - 1][1])
        detail = pywt.iswt(single, wd["wavelet"])[: len(wd["original"])]

        ax = axes[i + 1]
        ax.plot(detail[:n] * sign, lw=0.4, color=colors[d["band"]])
        ax.set_title(
            f"Level {lev}: {d['f_lo']:.1f}–{d['f_hi']:.1f} Hz  [{d['band']}]",
            fontsize=9,
            color=colors[d["band"]],
        )
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frames")
    plt.tight_layout()
    plt.show()

    return fig


def spike_triggered_average(
    wd: dict,
    threshold: float = 3.0,
    window_ms: float = 20.0,
    hz: float = None,
    min_isi_ms: float = 5.0,
) -> dict:
    """
    Compute spike-triggered averages on each band separately.

    Detects spikes from the spike band, then averages windows from
    the spike trace, subthreshold trace, and full trace independently.
    """
    import matplotlib.pyplot as plt

    if hz is None:
        hz = wd["hz"]

    win = int(window_ms * hz / 1000)
    min_isi = int(min_isi_ms * hz / 1000)

    # Detect spikes on the spike band using MAD z-score
    spike_z = _mad_zscore(wd["spike_trace"], window=3000)

    # Peak detection: threshold crossing + local maximum
    above = spike_z > threshold
    peaks = np.where(
        above
        & np.roll(spike_z > np.roll(spike_z, 1), 0)
        & np.roll(spike_z > np.roll(spike_z, -1), 0)
    )[0]

    # Enforce minimum ISI
    if len(peaks) > 1:
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= min_isi:
                keep.append(p)
        peaks = np.array(keep)

    # Filter out edges
    peaks = peaks[(peaks > win) & (peaks < len(spike_z) - win)]
    print(f"Detected {len(peaks)} spikes (threshold={threshold}, min_isi={min_isi_ms} ms)")

    if len(peaks) == 0:
        print("No spikes found — try lowering threshold.")
        return {"spike_times": peaks, "n_spikes": 0}

    # Extract windows and average
    t_ms = np.arange(-win, win + 1) / hz * 1000

    def _sta(trace, times, w):
        snippets = np.array([trace[t - w : t + w + 1] for t in times])
        return snippets.mean(axis=0), snippets.std(axis=0), snippets

    sta_full, std_full, snip_full = _sta(wd["original"], peaks, win)
    sta_spike, std_spike, snip_spike = _sta(wd["spike_trace"], peaks, win)
    sta_sub, std_sub, snip_sub = _sta(wd["subthresh_trace"], peaks, win)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)

    for ax, sta, std, title, color in [
        (axes[0], sta_full * -1, std_full, "Full trace STA", "green"),
        (axes[1], sta_spike * -1, std_spike, "Spike band STA", "steelblue"),
        (axes[2], sta_sub * -1, std_sub, "Subthreshold band STA", "purple"),
    ]:
        ax.fill_between(t_ms, (sta - std), (sta + std), alpha=0.2, color=color)
        ax.plot(t_ms, sta, lw=1.5, color=color)
        ax.axvline(0, color="r", lw=0.5, ls="--")
        ax.set_title(f"{title} (n={len(peaks)})", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

    return {
        "spike_times": peaks,
        "n_spikes": len(peaks),
        "t_ms": t_ms,
        "sta_full": sta_full,
        "sta_spike": sta_spike,
        "sta_subthresh": sta_sub,
        "snippets_full": snip_full,
        "snippets_spike": snip_spike,
        "snippets_sub": snip_sub,
    }


def decompose_clean(
    trace,
    hz=1000.0,
    wavelet="sym4",
    spike_levels=[1, 2, 3, 4],
    subthresh_levels=[5, 6, 7],
    spike_threshold=3.5,
    blank_pre_ms=1.0,
    blank_post_ms=8.0,
):
    from scipy.signal import find_peaks

    wd = decompose(
        trace, hz=hz, wavelet=wavelet, spike_levels=spike_levels, subthresh_levels=subthresh_levels
    )

    spike_frames, _ = find_peaks(trace * -1, height=spike_threshold, distance=3)

    pre_samples = int(blank_pre_ms * hz / 1000)
    post_samples = int(blank_post_ms * hz / 1000)

    spike_mask = np.zeros(len(trace), dtype=bool)
    for idx in spike_frames:
        lo = max(0, idx - pre_samples)
        hi = min(len(trace), idx + post_samples + 1)
        spike_mask[lo:hi] = True

    sub_clean = wd["subthresh_trace"].copy()
    if spike_mask.any():
        good = ~spike_mask
        t = np.arange(len(trace))
        sub_clean[spike_mask] = np.interp(t[spike_mask], t[good], sub_clean[good])

    wd["subthresh_clean"] = sub_clean
    wd["spike_mask"] = spike_mask
    wd["spike_frames"] = spike_frames
    print(
        f"  Found {len(spike_frames)} spikes, blanked {spike_mask.sum()} samples "
        f"(pre={blank_pre_ms}ms, post={blank_post_ms}ms)"
    )

    return wd

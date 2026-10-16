import numpy as np
from scipy.signal import find_peaks


def detect_spikes_for_sta(trace_z, threshold=3.5, min_distance=5, voltron_sign=-1):
    """
    Find spike frames for STA computation.

    Parameters
    ----------
    trace_z : np.ndarray
        MAD z-scored trace.
    threshold : float
        Detection threshold in z units. 3.5 is conservative.
    min_distance : int
        Minimum frames between spikes (refractory).
    voltron_sign : {-1, +1}
        Sign of spikes in ``trace_z``. Use -1 for a raw negative-going Voltron2
        trace and +1 when the trace has already been inverted so spikes point up.

    Returns
    -------
    spike_frames : np.ndarray, int
    """
    if voltron_sign not in {-1, 1}:
        raise ValueError("voltron_sign must be -1 or +1")
    signal = trace_z if voltron_sign == 1 else -trace_z
    peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
    return peaks


def robust_dfof(
    mov_raw: np.ndarray,  # (T, H, W) — the corrected_mov (pre-residual), real F values
    roi_mask: np.ndarray,  # (H, W) bool
    fp_norm: np.ndarray,  # (n_px,) footprint weights (sums to 1)
    spike_frames: np.ndarray,  # frames containing spike peaks
    sta_window: tuple = (-2, 3),  # frames around spike for STA (e.g. -2 to +3 = 5 frames centered)
    quiet_frames: np.ndarray = None,  # frames with no spikes for F0; if None, computed
    quiet_margin: int = 50,  # min frames from any spike to count as quiet
) -> dict:
    """
    Compute robust ΔF/F0 via pixel-by-pixel ΔF vs F0 regression slope.

    Returns
    -------
    dict with:
        slope : float            — robust dF/F per spike (the regression slope)
        raw_dfof_sta : float     — raw dF/F at STA peak (footprint trace, for ratio)
        scale_factor : float     — slope / raw_dfof_sta (multiply your dF/F by this)
        dF_img : (H, W)          — spike-triggered ΔF image (cell footprint signal)
        F0_img : (H, W)          — F0 image (silent-period mean)
        scatter_x : ΔF per pixel — for plotting/QC
        scatter_y : F0 per pixel — for plotting/QC
    """
    T, H, W = mov_raw.shape

    # ── 1. Identify quiet frames ──────────────────────────────────────────
    if quiet_frames is None:
        all_frames = np.arange(T)
        # Exclude any frame within quiet_margin of a spike
        spike_zone = np.zeros(T, dtype=bool)
        for sf in spike_frames:
            lo = max(0, sf - quiet_margin)
            hi = min(T, sf + quiet_margin + 1)
            spike_zone[lo:hi] = True
        quiet_frames = all_frames[~spike_zone]

    if len(quiet_frames) < 100:
        raise ValueError(f"Only {len(quiet_frames)} quiet frames available — STA unreliable")

    # ── 2. Compute F0 image: mean of all quiet frames ─────────────────────
    # We need this for ALL pixels in the ROI (and we'll mask later)
    # Chunked read if mov_raw is a memmap:
    F0_img = np.zeros((H, W), dtype=np.float64)
    chunk_size = 1000
    for i in range(0, len(quiet_frames), chunk_size):
        batch = quiet_frames[i : i + chunk_size]
        F0_img += mov_raw[batch].astype(np.float64).sum(axis=0)
    F0_img /= len(quiet_frames)

    # ── 3. Compute ΔF image: STA over spike windows minus F0 ──────────────
    sta_offsets = np.arange(sta_window[0], sta_window[1] + 1)
    valid_spikes = spike_frames[
        (spike_frames + sta_offsets[0] >= 0) & (spike_frames + sta_offsets[-1] < T)
    ]
    n_valid = len(valid_spikes)
    if n_valid < 5:
        raise ValueError(f"Only {n_valid} usable spike events — STA unreliable")

    sta_sum = np.zeros((H, W), dtype=np.float64)
    for sf in valid_spikes:
        win = mov_raw[sf + sta_offsets[0] : sf + sta_offsets[-1] + 1]
        sta_sum += win.astype(np.float64).mean(axis=0)
    sta_mean = sta_sum / n_valid

    dF_img = sta_mean - F0_img  # spike-triggered change per pixel
    # For Voltron2, dF_img will be NEGATIVE (dimmer at spike) — keep sign honest

    # ── 4. Regression on ROI pixels: dF vs F0 ─────────────────────────────
    scatter_x = F0_img[roi_mask].astype(np.float64)
    scatter_y = dF_img[roi_mask].astype(np.float64)

    # Linear regression with intercept: dF = slope * F0 + intercept
    # slope is dF/F0 for signal; intercept absorbs the spatially-uniform background
    A = np.vstack([scatter_x, np.ones_like(scatter_x)]).T
    slope, intercept = np.linalg.lstsq(A, scatter_y, rcond=None)[0]

    # ── 5. Compute raw dF/F from footprint-weighted trace at STA peak ─────
    # Trace-level: average F at quiet frames, average F at STA peak frames
    fp_trace_full = mov_raw[:, roi_mask].astype(np.float32) @ fp_norm
    F0_trace = fp_trace_full[quiet_frames].mean()
    sta_peak_frames = np.concatenate([sf + sta_offsets for sf in valid_spikes])
    sta_peak_frames = sta_peak_frames[(sta_peak_frames >= 0) & (sta_peak_frames < T)]
    F_sta = fp_trace_full[sta_peak_frames].mean()
    raw_dfof_sta = (F_sta - F0_trace) / F0_trace

    scale_factor = slope / raw_dfof_sta if raw_dfof_sta != 0 else np.nan

    return {
        "slope": slope,
        "intercept": intercept,
        "raw_dfof_sta": raw_dfof_sta,
        "scale_factor": scale_factor,
        "dF_img": dF_img.astype(np.float32),
        "F0_img": F0_img.astype(np.float32),
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
        "n_spikes_used": n_valid,
        "n_quiet": len(quiet_frames),
    }


def apply_robust_scaling(raw_dfof_trace: np.ndarray, scale_factor: float) -> np.ndarray:
    """Multiply raw dF/F trace by per-ROI scaling factor → comparable across ROIs."""
    return raw_dfof_trace * scale_factor

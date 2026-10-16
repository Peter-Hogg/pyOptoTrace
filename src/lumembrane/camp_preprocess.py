"""
camp_preprocess.py
Bleach correction and ΔF/F₀ for slow cAMP indicators (cAMPinG1, Pink Flamindo, etc.)
in spontaneous-activity recordings.

Designed as a sibling to pyneurotrace.filters.deltaFOverF0. The Jia et al. 2011
running-minimum F₀ is appropriate for fast, sparse calcium transients but will
absorb slow sustained cAMP events. This module instead:

    1. Optionally subtracts a background ROI trace (per-frame)
    2. Fits a bi-exponential to the raw trace, masking putative events
    3. Normalizes by the fit (F(t)/F_fit(t)) or subtracts it, yielding a
       bleach-flat trace
    4. Computes ΔF/F₀ against a low-percentile F₀ from the detrended trace

Usage
-----
    import camp_preprocess as cp

    dff, info = cp.deltaFOverF0_camp(
        raw_trace,
        hz=10.0,
        bg_trace=bg_trace,          # optional
        event_mask_pct=75,          # mask top 25% during bleach fit
        f0_percentile=10,
        f0_window_s=90,
    )

    # quick look at the fit quality
    cp.plot_bleach_fit(raw_trace, info, hz=10.0)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────────────────────────────────────
# Bleach models
# ──────────────────────────────────────────────────────────────────────────────
def _mono_exp(t, A, tau, C):
    """Single-exponential decay to an offset."""
    return A * np.exp(-t / tau) + C


def _bi_exp(t, A1, tau1, A2, tau2, C):
    """Bi-exponential decay to an offset."""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C


# ──────────────────────────────────────────────────────────────────────────────
# Bleach fitting with event masking
# ──────────────────────────────────────────────────────────────────────────────
def fit_bleach(
    trace: np.ndarray,
    hz: float,
    model: str = "biexp",
    event_mask_pct: float | None = 75.0,
    robust_iters: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Fit a (bi-)exponential decay to the trace, iteratively masking out samples
    that lie above the fit (candidate cAMP events) so the fit tracks the
    bleaching envelope rather than being pulled up by transients.

    Parameters
    ----------
    trace : 1D array of raw fluorescence
    hz : sampling rate (Hz)
    model : 'biexp' (recommended for cAMPinG1) or 'monoexp'
    event_mask_pct : percentile threshold for initial event rejection (0-100).
        Samples above this percentile of the *raw* trace are excluded from the
        first fit pass. Set to None to disable initial masking.
    robust_iters : number of refit passes. Each pass re-masks samples that
        exceed fit + 2*MAD of residuals. 2 is usually enough; 0 disables.

    Returns
    -------
    fit : 1D array, same length as trace — the bleach model evaluated at every
          time point (including originally masked samples)
    info : dict with keys 'params', 'model', 'mask', 'residual_mad', 'success'
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    t = np.arange(n) / hz

    # --- initial mask ---------------------------------------------------------
    if event_mask_pct is not None:
        thresh = np.percentile(trace, event_mask_pct)
        mask = trace <= thresh
    else:
        mask = np.ones(n, dtype=bool)

    # Always keep the first ~2 s — important for anchoring the fast component
    early = int(min(2.0 * hz, 0.05 * n))
    mask[:early] = True

    # --- initial guesses ------------------------------------------------------
    F0_guess = np.median(trace[-max(int(0.1 * n), 1) :])  # tail as asymptote
    A_total = trace[: max(int(0.01 * n), 1)].mean() - F0_guess
    T = t[-1]

    if model == "biexp":
        p0 = [0.6 * A_total, T / 20, 0.4 * A_total, T / 2, F0_guess]
        bounds = (
            [-np.inf, 1e-3, -np.inf, 1e-3, -np.inf],
            [np.inf, T, np.inf, 10 * T, np.inf],
        )
        func = _bi_exp
    elif model == "monoexp":
        p0 = [A_total, T / 3, F0_guess]
        bounds = ([-np.inf, 1e-3, -np.inf], [np.inf, 10 * T, np.inf])
        func = _mono_exp
    else:
        raise ValueError(f"Unknown model {model!r}")

    # --- iterative fit with residual-based re-masking -------------------------
    success = True
    params = None
    residual_mad = np.nan

    try:
        for it in range(robust_iters + 1):
            popt, _ = curve_fit(func, t[mask], trace[mask], p0=p0, bounds=bounds, maxfev=20000)
            fit_all = func(t, *popt)

            # residual-based re-masking for next iteration
            resid = trace - fit_all
            residual_mad = np.median(np.abs(resid - np.median(resid))) / 0.6745
            # keep samples not strongly above the fit (asymmetric: events are up)
            new_mask = resid < 2.0 * residual_mad
            new_mask[:early] = True
            # also keep tail samples for asymptote anchoring
            new_mask[-early:] = True

            if np.array_equal(new_mask, mask) and it > 0:
                break
            mask = new_mask
            p0 = popt  # warm-start next iter

        params = popt
        fit = func(t, *popt)

    except (RuntimeError, ValueError):
        # Fall back to monoexp if biexp fails, or return a flat trace
        if model == "biexp":
            return fit_bleach(
                trace,
                hz,
                model="monoexp",
                event_mask_pct=event_mask_pct,
                robust_iters=robust_iters,
            )
        success = False
        fit = np.full(n, np.median(trace))
        params = None

    info = {
        "params": params,
        "model": model,
        "mask": mask,
        "residual_mad": residual_mad,
        "success": success,
    }
    return fit, info


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point: ΔF/F₀ for cAMP
# ──────────────────────────────────────────────────────────────────────────────
def deltaFOverF0_camp(
    trace: np.ndarray,
    hz: float,
    bg_trace: np.ndarray | None = None,
    bg_scale: float = 1.0,
    model: str = "biexp",
    event_mask_pct: float | None = 75.0,
    detrend_mode: str = "divide",
    f0_percentile: float = 10.0,
    f0_window_s: float = 90.0,
    trim_tail: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    ΔF/F₀ pipeline for slow cAMP indicators.

    Parameters
    ----------
    trace : 1D array, raw ROI fluorescence
    hz : sampling rate (Hz)
    bg_trace : optional 1D background-ROI trace (same length). Subtracted per
        frame before detrending.
    bg_scale : scalar multiplier for bg_trace (e.g. 0.7 for partial subtraction)
    model : 'biexp' or 'monoexp' — bleach model
    event_mask_pct : percentile above which to mask samples for bleach fit.
        75 = treat top 25% as candidate events. None disables initial masking.
    detrend_mode : 'divide' normalizes by the fit (removes multiplicative
        bleaching, good when F scales with illumination). 'subtract' keeps
        absolute units. 'divide' is the standard choice.
    f0_percentile : low percentile of the detrended trace used as F₀ baseline
    f0_window_s : window length (seconds) for the rolling percentile. Use a
        value several times longer than your expected event duration. Set to
        0 or None for a scalar F₀ (whole-trace percentile).
    trim_tail : drop this many samples from the end before processing
        (useful for the acquisition-end cliff you saw).

    Returns
    -------
    dff : 1D array, ΔF/F₀
    info : dict with 'bleach_fit', 'bleach_info', 'F0', 'detrended',
        'bg_subtracted', 'trace_used'
    """
    trace = np.asarray(trace, dtype=np.float64)

    if trim_tail > 0:
        trace = trace[:-trim_tail]

    # 1. background subtraction (optional) ------------------------------------
    if bg_trace is not None:
        bg_trace = np.asarray(bg_trace, dtype=np.float64)
        if trim_tail > 0:
            bg_trace = bg_trace[:-trim_tail]
        if len(bg_trace) != len(trace):
            raise ValueError(f"bg_trace length {len(bg_trace)} != trace length {len(trace)}")
        bg_subtracted = trace - bg_scale * bg_trace
    else:
        bg_subtracted = trace.copy()

    # 2. bleach fit ------------------------------------------------------------
    if model is not None:
        fit, bleach_info = fit_bleach(
            bg_subtracted,
            hz,
            model=model,
            event_mask_pct=event_mask_pct,
        )
    else:
        fit, bleach_info = None, None
    # 3. detrend ---------------------------------------------------------------
    if detrend_mode == "divide":
        # Normalize fit to start at its own initial value — the detrended trace
        # then has the same amplitude units as bg_subtracted.
        scale = fit / fit[0]
        # Guard against zero crossings (shouldn't happen for fluorescence)
        scale = np.where(np.abs(scale) < 1e-6, 1e-6, scale)
        detrended = bg_subtracted / scale
    elif detrend_mode == "subtract":
        detrended = bg_subtracted - (fit - fit[0])
    elif detrend_mode is None:
        detrended = bg_subtracted

    else:
        raise ValueError(f"Unknown detrend_mode {detrend_mode!r}")

    # 4. F₀ from low percentile of detrended trace -----------------------------
    if f0_window_s is None or f0_window_s <= 0:
        F0 = np.full_like(detrended, np.percentile(detrended, f0_percentile))
    else:
        window = max(int(f0_window_s * hz), 3)
        # Ensure odd for symmetric footprint
        if window % 2 == 0:
            window += 1
        F0 = percentile_filter(
            detrended,
            percentile=f0_percentile,
            size=window,
            mode="reflect",
        )

    # Guard against F0 ≈ 0
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (detrended - F0) / F0_safe

    info = {
        "bleach_fit": fit,
        "bleach_info": bleach_info,
        "F0": F0,
        "detrended": detrended,
        "bg_subtracted": bg_subtracted,
        "trace_used": trace,
        "hz": hz,
    }
    return dff, info


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ──────────────────────────────────────────────────────────────────────────────
def plot_bleach_fit(raw_trace, info, hz: float = None, ax=None):
    """
    Four-panel diagnostic: raw + fit + mask, detrended + F0, dFF, and the
    bleach residual. Pass the `info` dict returned by deltaFOverF0_camp.
    """
    import matplotlib.pyplot as plt

    hz = hz or info.get("hz", 1.0)
    fit = info["bleach_fit"]
    detrended = info["detrended"]
    F0 = info["F0"]
    bg_sub = info["bg_subtracted"]

    n = len(bg_sub)
    t = np.arange(n) / hz

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # 1. raw + fit + mask
    axes[0].plot(t, bg_sub, lw=0.6, label="bg-subtracted")
    axes[0].plot(t, fit, lw=1.5, color="C3", label="bleach fit")
    # mark masked (= presumed event) samples

    axes[0].set_ylabel("F (raw)")
    axes[0].legend(loc="upper right", fontsize=8)
    params = info["bleach_info"]["params"]
    model = info["bleach_info"]["model"]
    if params is not None and model == "biexp":
        A1, tau1, A2, tau2, C = params
        axes[0].set_title(
            f"bi-exp: τ₁={tau1:.1f}s (A={A1:.1f}), τ₂={tau2:.1f}s (A={A2:.1f}), C={C:.1f}",
            fontsize=10,
        )
    elif params is not None and model == "monoexp":
        A, tau, C = params
        axes[0].set_title(
            f"mono-exp: τ={tau:.1f}s (A={A:.1f}), C={C:.1f}",
            fontsize=10,
        )

    # 2. detrended + F0
    axes[1].plot(t, detrended, lw=0.6, label="detrended")
    axes[1].plot(t, F0, lw=1.2, color="C3", label="F₀ (rolling pct)")
    axes[1].set_ylabel("F (detrended)")
    axes[1].legend(loc="upper right", fontsize=8)

    # 3. dFF
    F0_safe = np.where(np.abs(F0) < 1e-6, 1e-6, F0)
    dff = (detrended - F0) / F0_safe
    axes[2].plot(t, dff, lw=0.6, color="C2")
    axes[2].axhline(0, color="k", lw=0.4, alpha=0.5)
    axes[2].set_ylabel("ΔF/F₀")

    # 4. bleach-corrected residual (sanity check — should look stationary)
    axes[3].plot(t, bg_sub - fit, lw=0.6, color="C4")
    axes[3].axhline(0, color="k", lw=0.4, alpha=0.5)
    axes[3].set_ylabel("raw − fit")
    axes[3].set_xlabel("time (s)")

    plt.tight_layout()
    return fig, axes

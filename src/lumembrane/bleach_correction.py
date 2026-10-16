import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────
# Exponential model(s)
# ──────────────────────────────────────────────────────────────────────────


def _biexp(t, a1, tau1, a2, tau2, c):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def _monoexp(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_bleach_curve(trace, hz=1000.0, model="monoexp", p0=None, maxfev=20000):
    """
    Fit an exponential decay to a 1D trace (e.g. global_F).

    Parameters
    ----------
    trace : array-like, shape (T,)
        Trace to fit (typically global_F, since it should be dominated by
        bleaching + background, not single-cell voltage dynamics).
    hz : float
        Sampling rate, used only to give `tau` physically meaningful units
        (seconds) in the returned params. Fitting itself is done in samples
        for numerical stability, then tau is converted.
    model : {"monoexp", "biexp"}
        Single or double exponential decay model.
    p0 : sequence or None
        Initial guess. If None, a reasonable guess is constructed from the
        data (start/end levels, ~1/3 of recording length as initial tau).
    maxfev : int
        Max function evaluations for curve_fit.

    Returns
    -------
    fit_curve : np.ndarray, shape (T,)
        The fitted bleach curve, same length as `trace`.
    params : dict
        Fitted parameters, with tau values converted to seconds.
    """
    trace = np.asarray(trace, dtype=np.float64)
    T = trace.shape[0]
    t = np.arange(T, dtype=np.float64)

    if model == "monoexp":
        if p0 is None:
            a0 = trace[0] - trace[-1]
            c0 = trace[-1]
            tau0 = T / 3.0
            p0 = [a0, tau0, c0]
        popt, _ = curve_fit(_monoexp, t, trace, p0=p0, maxfev=maxfev)
        fit_curve = _monoexp(t, *popt)
        params = {"a": popt[0], "tau_sec": popt[1] / hz, "c": popt[2]}

    elif model == "biexp":
        if p0 is None:
            a0 = (trace[0] - trace[-1]) * 0.7
            a1 = (trace[0] - trace[-1]) * 0.3
            c0 = trace[-1]
            p0 = [a0, T / 10.0, a1, T / 2.0, c0]
        popt, _ = curve_fit(_biexp, t, trace, p0=p0, maxfev=maxfev)
        fit_curve = _biexp(t, *popt)
        params = {
            "a1": popt[0],
            "tau1_sec": popt[1] / hz,
            "a2": popt[2],
            "tau2_sec": popt[3] / hz,
            "c": popt[4],
        }
    else:
        raise ValueError("model must be 'monoexp' or 'biexp'")

    return fit_curve, params


# ──────────────────────────────────────────────────────────────────────────
# Percentile-window envelope (kept as an option, but see caveat in docstring)
# ──────────────────────────────────────────────────────────────────────────


def rolling_percentile_envelope(trace, window, percentile=80):
    """
    Track a rolling percentile of `trace` using a sliding window, implemented
    via min/max morphological filters for speed when percentile is 0 or 100,
    otherwise falls back to a strided percentile computation.

    Parameters
    ----------
    trace : array-like, shape (T,)
    window : int
        Window size in samples (should span multiple spikes so the envelope
        doesn't just track individual peaks).
    percentile : float
        Percentile to track, 0-100. 100 = rolling max, 0 = rolling min.

    Returns
    -------
    envelope : np.ndarray, shape (T,)
    """
    trace = np.asarray(trace, dtype=np.float64)
    T = trace.shape[0]
    window = int(window)
    if window % 2 == 0:
        window += 1

    if percentile >= 99.999:
        return maximum_filter1d(trace, size=window, mode="nearest")
    if percentile <= 0.001:
        return minimum_filter1d(trace, size=window, mode="nearest")

    half = window // 2
    envelope = np.empty(T, dtype=np.float64)
    # Strided percentile -- fine for offline/QC use; vectorize further only
    # if this becomes a bottleneck on very long movies.
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        envelope[i] = np.percentile(trace[lo:hi], percentile)
    return envelope


# ──────────────────────────────────────────────────────────────────────────
# Main correction routine
# ──────────────────────────────────────────────────────────────────────────


def correct_bleaching(
    cell_trace,
    global_F,
    hz=1000.0,
    method="global_ratio",
    model="monoexp",
    percentile=80,
    percentile_window_sec=5.0,
    return_diagnostics=False,
):
    """
    Correct a single-cell voltage trace for photobleaching using the global
    fluorescence (whole-FOV) trace as a bleach reference, rather than fitting
    the cell's own percentile envelope.

    Why reference off global_F instead of the cell trace's own top
    percentile: for an indicator like Voltron2 where F decreases on
    depolarization, the upper envelope of the cell's own trace confounds two
    things that look identical in raw F -- photobleaching, and slow
    depolarization block. Fitting an exponential to the cell's own top
    percentile and dividing it out will silently remove real depolarization
    along with bleaching. global_F is dominated by bulk/neuropil signal and
    bleaching photophysics, and is far less sensitive to one cell's spiking
    history, so it gives you a bleach estimate that's mostly orthogonal to
    single-cell voltage dynamics.

    Parameters
    ----------
    cell_trace : array-like, shape (T,)
        Raw (not yet inverted) fluorescence trace for one ROI.
    global_F : array-like, shape (T,)
        Whole-FOV mean fluorescence trace (same units/scale family as
        cell_trace, i.e. raw F, not already normalized).
    hz : float
        Sampling rate in Hz.
    method : {"global_ratio", "global_subtract", "percentile_fallback"}
        "global_ratio": fit exponential to global_F, divide both global_F's
            fit and cell_trace by (fit / fit[0]) to remove proportional
            bleaching. Use when bleaching is expected to be multiplicative
            (typical for photobleaching).
        "global_subtract": fit exponential to global_F, subtract
            (fit - fit[-1]) from cell_trace directly. Use if you believe
            bleaching is closer to additive in your pipeline (e.g. if
            cell_trace is already a background-subtracted dF rather than raw
            F).
        "percentile_fallback": ignores global_F, fits exponential to a
            rolling top-percentile envelope of cell_trace itself. This is
            the risky option discussed above -- only use for comparison /
            sanity-checking against the global_F-based methods, since it can
            remove real depolarization-driven decay.
    model : {"monoexp", "biexp"}
        Passed to fit_bleach_curve.
    percentile : float
        Only used if method == "percentile_fallback".
    percentile_window_sec : float
        Window (seconds) for the rolling percentile, only used if
        method == "percentile_fallback". Should span several spikes.
    return_diagnostics : bool
        If True, also return the bleach curve(s) and fit params for
        plotting/QC.

    Returns
    -------
    cell_corrected : np.ndarray, shape (T,)
        Bleach-corrected cell trace, same scale as input cell_trace.
    diagnostics : dict (only if return_diagnostics=True)
        Contains 'bleach_curve', 'params', and method used.
    """
    cell_trace = np.asarray(cell_trace, dtype=np.float64)
    global_F = np.asarray(global_F, dtype=np.float64)
    T = cell_trace.shape[0]

    if global_F.shape[0] != T:
        raise ValueError(
            f"cell_trace and global_F must have the same length, got {T} and {global_F.shape[0]}"
        )

    diagnostics = {"method": method}

    if method == "global_ratio":
        bleach_curve, params = fit_bleach_curve(global_F, hz=hz, model=model)
        # Normalize bleach curve to start at 1 so cell_trace[0] is preserved
        norm_curve = bleach_curve / bleach_curve[0]
        cell_corrected = cell_trace / norm_curve
        diagnostics.update(
            {"bleach_curve": bleach_curve, "norm_curve": norm_curve, "params": params}
        )

    elif method == "global_subtract":
        bleach_curve, params = fit_bleach_curve(global_F, hz=hz, model=model)
        decay_component = bleach_curve - bleach_curve[-1]
        cell_corrected = cell_trace - decay_component
        diagnostics.update(
            {"bleach_curve": bleach_curve, "decay_component": decay_component, "params": params}
        )

    elif method == "percentile_fallback":
        window = int(percentile_window_sec * hz)
        envelope = rolling_percentile_envelope(cell_trace, window=window, percentile=percentile)
        bleach_curve, params = fit_bleach_curve(envelope, hz=hz, model=model)
        norm_curve = bleach_curve / bleach_curve[0]
        cell_corrected = cell_trace / norm_curve
        diagnostics.update(
            {
                "envelope": envelope,
                "bleach_curve": bleach_curve,
                "norm_curve": norm_curve,
                "params": params,
            }
        )

    else:
        raise ValueError(
            "method must be 'global_ratio', 'global_subtract', or 'percentile_fallback'"
        )

    if return_diagnostics:
        return cell_corrected, diagnostics
    return cell_corrected


# ──────────────────────────────────────────────────────────────────────────
# QC plotting helper
# ──────────────────────────────────────────────────────────────────────────


def plot_bleach_correction_qc(cell_trace, global_F, hz=1000.0, model="monoexp", invert=True):
    """
    Quick side-by-side QC plot: raw cell trace, global_F with its bleach fit,
    and the corrected cell trace for both global_ratio and percentile_fallback
    methods, so you can visually compare whether the percentile method is
    eating real signal that the global_F method preserves.
    """
    import matplotlib.pyplot as plt

    sign = -1 if invert else 1
    t_sec = np.arange(len(cell_trace)) / hz

    corrected_global, diag_global = correct_bleaching(
        cell_trace, global_F, hz=hz, method="global_ratio", model=model, return_diagnostics=True
    )
    corrected_pct, diag_pct = correct_bleaching(
        cell_trace,
        global_F,
        hz=hz,
        method="percentile_fallback",
        model=model,
        return_diagnostics=True,
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t_sec, sign * cell_trace, lw=0.5)
    axes[0].set_title("Raw cell trace (inverted)" if invert else "Raw cell trace")

    axes[1].plot(t_sec, global_F, lw=0.5, label="global_F")
    axes[1].plot(t_sec, diag_global["bleach_curve"], lw=1.5, label="exp fit")
    axes[1].set_title("global_F with bleach fit")
    axes[1].legend(loc="upper right")

    axes[2].plot(t_sec, sign * corrected_global, lw=0.5)
    axes[2].set_title("Corrected via global_F ratio")

    axes[3].plot(t_sec, sign * corrected_pct, lw=0.5, color="tab:orange")
    axes[3].plot(t_sec, sign * cell_trace * 0, lw=0)  # keep axis scale sane
    axes[3].set_title(
        "Corrected via percentile_fallback (compare carefully -- may remove real depolarization)"
    )
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig, {"global_ratio": diag_global, "percentile_fallback": diag_pct}

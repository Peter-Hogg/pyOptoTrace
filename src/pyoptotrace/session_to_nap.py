"""
session_to_nap.py
Wrap the outputs of the voltage and cAMP pipelines into a unified pynapple
session object with automatic bad-epoch detection.

Assumes:
    - Voltage: dict of {cell_id: {'trace_z': 1D array, ...}} at voltage_hz (1 kHz)
      produced by signal_extraction.extract_all_rois
    - cAMP: dict of {cell_id: 1D array} at camp_hz (10 Hz) — the ΔF/F₀ output
      of camp_preprocess.deltaFOverF0_camp

Produces:
    Session dataclass with:
        .voltage    : TsdFrame (T_v, N_cells) at 1 kHz
        .camp       : TsdFrame (T_c, N_cells) at 10 Hz
        .recording  : IntervalSet spanning the full session
        .bad_epochs : IntervalSet of z-artifact windows to exclude
        .good_epochs: recording ∖ bad_epochs
        + convenience methods for common operations
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pynapple as nap


# ──────────────────────────────────────────────────────────────────────────────
# Z-artifact detection
# ──────────────────────────────────────────────────────────────────────────────
def detect_z_artifacts(
    camp: nap.TsdFrame,
    mad_threshold: float = 4.0,
    pad_s: float = 0.5,
    merge_s: float = 1.0,
) -> nap.IntervalSet:
    """
    Detect z-drift artifacts from the common-mode cAMP signal.

    Rationale: cAMP dynamics are slow (off-τ several s). Sharp negative
    deflections shared across cells are almost always z-shifts pulling the
    cells out of focus. We detect these as large negative excursions of the
    across-cell mean and return intervals to exclude.

    Parameters
    ----------
    camp : TsdFrame with one column per cell
    mad_threshold : k × MAD below the median counts as artifact. 4 is
        conservative; drop to 3 for more aggressive rejection.
    pad_s : seconds of padding added on each side of detected events
    merge_s : intervals closer than this are merged into one

    Returns
    -------
    IntervalSet of bad epochs in seconds.
    """
    # Across-cell mean (robust to single-cell biology)
    common_mode = np.mean(camp.values, axis=1)
    t = camp.index.values

    med = np.median(common_mode)
    mad = np.median(np.abs(common_mode - med)) / 0.6745
    bad_mask = (common_mode - med) < -mad_threshold * mad

    if not bad_mask.any():
        return nap.IntervalSet(start=np.array([]), end=np.array([]))

    # Turn boolean mask into contiguous runs
    edges = np.diff(bad_mask.astype(int))
    starts_idx = np.where(edges == 1)[0] + 1
    ends_idx = np.where(edges == -1)[0] + 1
    if bad_mask[0]:
        starts_idx = np.concatenate([[0], starts_idx])
    if bad_mask[-1]:
        ends_idx = np.concatenate([ends_idx, [len(bad_mask)]])

    starts = t[starts_idx] - pad_s
    # end index is exclusive; use end_idx-1 for last bad sample then add pad
    ends = t[np.clip(ends_idx - 1, 0, len(t) - 1)] + pad_s

    # Merge intervals that are closer than merge_s
    if len(starts) > 1:
        keep_start = [starts[0]]
        keep_end = [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s - keep_end[-1] < merge_s:
                keep_end[-1] = e
            else:
                keep_start.append(s)
                keep_end.append(e)
        starts = np.array(keep_start)
        ends = np.array(keep_end)

    # Clamp to recording bounds
    starts = np.clip(starts, t[0], t[-1])
    ends = np.clip(ends, t[0], t[-1])

    return nap.IntervalSet(start=starts, end=ends)


# ──────────────────────────────────────────────────────────────────────────────
# Session container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Session:
    voltage: nap.TsdFrame | None = None
    camp: nap.TsdFrame | None = None
    recording: nap.IntervalSet | None = None
    bad_epochs: nap.IntervalSet | None = None
    good_epochs: nap.IntervalSet | None = None
    metadata: dict = field(default_factory=dict)

    # ── Restriction helpers ──────────────────────────────────────────────
    def restrict_to_good(self) -> Session:
        """Return a copy with voltage and cAMP restricted to good epochs."""
        return Session(
            voltage=self.voltage.restrict(self.good_epochs) if self.voltage is not None else None,
            camp=self.camp.restrict(self.good_epochs) if self.camp is not None else None,
            recording=self.good_epochs,
            bad_epochs=nap.IntervalSet(start=np.array([]), end=np.array([])),
            good_epochs=self.good_epochs,
            metadata={**self.metadata, "restricted": True},
        )

    # ── Resampling helpers ───────────────────────────────────────────────
    def camp_at_voltage_rate(self, method: str = "interpolate") -> nap.TsdFrame:
        """
        Upsample cAMP to the voltage timebase for joint plotting / analysis.

        method='interpolate' uses linear interpolation (smooth, good for display).
        method='bin' uses nearest-sample binning (faster, stairstep).
        """
        if self.voltage is None or self.camp is None:
            raise ValueError("Session needs both voltage and camp set")
        target_t = self.voltage.index.values
        if method == "interpolate":
            # pynapple >=0.6 has .interpolate(ts_target)
            if hasattr(self.camp, "interpolate"):
                return self.camp.interpolate(nap.Ts(t=target_t))
            # fallback to scipy
            from scipy.interpolate import interp1d

            f = interp1d(
                self.camp.index.values,
                self.camp.values,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return nap.TsdFrame(t=target_t, d=f(target_t), columns=list(self.camp.columns))
        elif method == "bin":
            bin_size = 1.0 / self._estimate_rate(self.voltage)
            return self.camp.bin_average(bin_size=bin_size)
        else:
            raise ValueError(f"Unknown method {method!r}")

    def voltage_at_camp_rate(self, bin_size_s: float | None = None) -> nap.TsdFrame:
        """Downsample voltage to cAMP timebase by averaging within bins."""
        if self.voltage is None or self.camp is None:
            raise ValueError("Session needs both voltage and camp set")
        if bin_size_s is None:
            bin_size_s = 1.0 / self._estimate_rate(self.camp)
        return self.voltage.bin_average(bin_size=bin_size_s)

    # ── Utility ──────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_rate(tsd) -> float:
        t = tsd.index.values
        return 1.0 / np.median(np.diff(t))

    def summary(self) -> str:
        lines = ["── Session ──"]
        if self.voltage is not None:
            hz = self._estimate_rate(self.voltage)
            lines.append(
                f"  voltage : {self.voltage.shape} @ {hz:.1f} Hz ({self.voltage.shape[1]} cells)"
            )
        if self.camp is not None:
            hz = self._estimate_rate(self.camp)
            lines.append(
                f"  camp    : {self.camp.shape} @ {hz:.1f} Hz ({self.camp.shape[1]} cells)"
            )
        if self.recording is not None:
            dur = float(self.recording.end[0] - self.recording.start[0])
            lines.append(f"  duration: {dur:.1f} s")
        if self.bad_epochs is not None and len(self.bad_epochs) > 0:
            bad_dur = float(np.sum(self.bad_epochs.end - self.bad_epochs.start))
            lines.append(f"  bad_epochs: {len(self.bad_epochs)} ({bad_dur:.2f} s excluded)")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Constructor from pipeline outputs
# ──────────────────────────────────────────────────────────────────────────────
def build_session(
    voltage_results: dict | None = None,
    camp_dff: dict | None = None,
    voltage_hz: float = 1000.0,
    camp_hz: float = 10.0,
    voltage_key: str = "trace_z",
    voltage_n_samples: int | None = None,
    camp_n_samples: int | None = None,
    detect_artifacts: bool = True,
    artifact_mad_threshold: float = 4.0,
    t0: float = 0.0,
    metadata: dict | None = None,
) -> Session:
    """
    Wrap pipeline outputs into a pynapple Session.

    Parameters
    ----------
    voltage_results : dict like {cell_id: {voltage_key: 1D array, ...}}
        Output of signal_extraction.extract_all_rois. Keys are np.int8 cell IDs.
    camp_dff : dict like {cell_id: 1D array}
        Output of camp_preprocess.deltaFOverF0_camp, collected per cell.
    voltage_hz : sampling rate of voltage traces (default 1 kHz)
    camp_hz : sampling rate of cAMP traces (default 10 Hz)
    voltage_key : which key to pull from voltage_results[cid] dict
    voltage_n_samples, camp_n_samples : trim to these lengths (useful for
        aligning channels to a common end-time). None = use full length.
    detect_artifacts : run z-artifact detection from the common-mode cAMP
    artifact_mad_threshold : passed to detect_z_artifacts
    t0 : offset added to all timestamps (for stitching across sessions)
    metadata : arbitrary extra info stored on the Session

    Returns
    -------
    Session object.
    """
    session = Session(metadata=metadata or {})

    # ── Voltage ──────────────────────────────────────────────────────────
    if voltage_results is not None and len(voltage_results) > 0:
        # Normalize cell IDs to int for cleaner downstream use
        cell_ids = sorted(voltage_results.keys(), key=int)
        traces = [np.asarray(voltage_results[cid][voltage_key] * -1) for cid in cell_ids]
        # Common length
        n = min(len(tr) for tr in traces)
        if voltage_n_samples is not None:
            n = min(n, voltage_n_samples)
        data = np.column_stack([tr[:n] for tr in traces]).astype(np.float32)
        t = t0 + np.arange(n) / voltage_hz
        session.voltage = nap.TsdFrame(
            t=t, d=data, columns=[f"v_cell_{int(cid)}" for cid in cell_ids]
        )

    # ── cAMP ─────────────────────────────────────────────────────────────
    if camp_dff is not None and len(camp_dff) > 0:
        cell_ids = sorted(camp_dff.keys(), key=int)
        traces = [np.asarray(camp_dff[cid]) for cid in cell_ids]
        n = min(len(tr) for tr in traces)
        if camp_n_samples is not None:
            n = min(n, camp_n_samples)
        data = np.column_stack([tr[:n] for tr in traces]).astype(np.float32)
        t = t0 + np.arange(n) / camp_hz
        session.camp = nap.TsdFrame(t=t, d=data, columns=[f"c_cell_{int(cid)}" for cid in cell_ids])

    # ── Recording interval (union of modalities' time extents) ───────────
    t_ends = []
    if session.voltage is not None:
        t_ends.append(session.voltage.index.values[-1])
    if session.camp is not None:
        t_ends.append(session.camp.index.values[-1])
    if not t_ends:
        raise ValueError("No data provided — both voltage_results and camp_dff are None")
    session.recording = nap.IntervalSet(start=t0, end=min(t_ends))

    # ── Bad epochs ───────────────────────────────────────────────────────
    if detect_artifacts and session.camp is not None and session.camp.shape[1] >= 2:
        session.bad_epochs = detect_z_artifacts(
            session.camp,
            mad_threshold=artifact_mad_threshold,
        )
    else:
        session.bad_epochs = nap.IntervalSet(start=np.array([]), end=np.array([]))

    # good = recording ∖ bad
    if len(session.bad_epochs) > 0:
        session.good_epochs = session.recording.set_diff(session.bad_epochs)
    else:
        session.good_epochs = session.recording

    return session

"""
luminos_io.py
-------------
Utility functions for loading Luminos acquisition data directly,
without going through the full NWB conversion pipeline.

Usage:
    from luminos_io import load_acquisition, load_bin, load_camera_meta

    # Load everything from a folder
    mov1, mov2, meta = load_acquisition("./baseline_001")

    # Load just one bin file with known dimensions
    mov = load_bin("frames1.bin", nrow=512, ncol=512)

    # Load camera metadata only
    cam = load_camera_meta("metadata.mat", camera_index=1)
    mov = load_bin("frames1.bin", cam=cam)
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Camera metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CameraParams:
    """Camera acquisition parameters parsed from metadata.mat."""

    name: str = "unknown"
    cam_id: str = "unknown"
    roi: list = field(default_factory=lambda: [0, 2048, 0, 2048])
    nrow: int = 2048
    ncol: int = 2048
    binning: int = 1
    exposuretime: float = float("nan")
    frames_requested: int = 0
    dropped_frames: int = 0
    hsync_rate: float = float("nan")
    daqtrig_period_ms: float = float("nan")
    microns_per_pixel: float = float("nan")

    @property
    def fps(self) -> float:
        return 1000.0 / self.daqtrig_period_ms if self.daqtrig_period_ms > 0 else float("nan")

    def __repr__(self):
        return (
            f"CameraParams({self.name!r}, {self.ncol}x{self.nrow}, "
            f"bin={self.binning}, {self.fps:.2f} fps, "
            f"exposure={self.exposuretime * 1000:.2f} ms)"
        )


def _h5_scalar(f, ds) -> float:
    """Dereference an HDF5 dataset or object-ref scalar to a Python float/int."""
    v = ds[()]
    if hasattr(v, "dtype") and v.dtype.kind == "O":
        # It's an object reference — follow it
        v = f[v.ravel()[0]][()]
    v = np.asarray(v).squeeze()
    return v.flat[0] if v.ndim > 0 else float(v)


def _h5_str(f, ds) -> str:
    """Dereference an HDF5 dataset to a string (handles char arrays)."""
    v = ds[()]
    if hasattr(v, "dtype") and v.dtype.kind == "O":
        v = f[v.ravel()[0]][()]
    v = np.asarray(v).squeeze()
    if v.dtype.kind in ("U", "S"):
        return str(v)
    # MATLAB char array stored as uint16 code points
    return "".join(chr(int(c)) for c in v.flat)


def load_camera_meta(
    mat_path: str | Path,
    camera_index: int = 1,
) -> CameraParams:
    """
    Parse camera parameters from a Luminos metadata.mat (HDF5 / v7.3).

    Parameters
    ----------
    mat_path : path to metadata.mat
    camera_index : 1-based index into Device_Data (1 = red/frames1, 2 = green/frames2)

    Returns
    -------
    CameraParams dataclass
    """
    mat_path = Path(mat_path)
    params = CameraParams()

    with h5py.File(mat_path, "r") as f:
        if "Device_Data" not in f:
            raise KeyError("No 'Device_Data' in metadata.mat")

        refs = f["Device_Data"][()].ravel()

        # Find the nth camera device (camera_index is 1-based across all devices,
        # but we want the nth camera specifically)
        cam_count = 0
        entry = None
        for ref in refs:
            try:
                dev = f[ref]
                dtype_str = _h5_str(f, dev["deviceType"]).lower() if "deviceType" in dev else ""
                if "cam" in dtype_str:
                    cam_count += 1
                    if cam_count == camera_index:
                        entry = dev
                        break
            except Exception:
                continue

        if entry is None:
            raise IndexError(
                f"Camera {camera_index} not found in Device_Data (found {cam_count} cameras total)"
            )

        def get(key, default=float("nan")):
            try:
                return _h5_scalar(f, entry[key])
            except Exception:
                return default

        def get_str(key, default="unknown"):
            try:
                return _h5_str(f, entry[key])
            except Exception:
                return default

        roi_raw = entry["ROI"][()] if "ROI" in entry else np.array([0, 2048, 0, 2048])
        roi = np.asarray(roi_raw).squeeze().tolist()

        binning = int(get("bin", 1))
        nrow = int(roi[3]) // binning
        ncol = int(roi[1]) // binning

        params = CameraParams(
            name=get_str("name"),
            cam_id=get_str("cam_id"),
            roi=roi,
            nrow=nrow,
            ncol=ncol,
            binning=binning,
            exposuretime=float(get("exposuretime")),
            frames_requested=int(get("frames_requested", 0)),
            dropped_frames=int(get("dropped_frames", 0)),
            hsync_rate=float(get("hsync_rate")),
            daqtrig_period_ms=float(get("daqtrig_period_ms")),
            microns_per_pixel=float(get("microns_per_pixel")),
        )

    print(params)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Binary frame loading
# ─────────────────────────────────────────────────────────────────────────────


def load_bin(
    bin_path: str | Path,
    nrow: int = None,
    ncol: int = None,
    cam: CameraParams = None,
    dtype: str = "uint16",
    endian: str = "<",
    n_frames: int = None,
    start_frame: int = 0,
    memmap: bool = False,
) -> np.ndarray:
    """
    Load a Luminos raw binary file (flat uint16, no header).

    Dimensions can be provided either directly (nrow, ncol) or via a
    CameraParams object. CameraParams takes priority if both are given.

    Parameters
    ----------
    bin_path    : path to .bin file
    nrow        : number of rows per frame
    ncol        : number of columns per frame
    cam         : CameraParams (overrides nrow/ncol if provided)
    dtype       : numpy dtype string (default: "uint16")
    endian      : byte order "<" little (default) or ">" big
    n_frames    : number of frames to load (default: all)
    start_frame : first frame to load (default: 0)
    memmap      : if True return a memory-mapped array instead of loading to RAM

    Returns
    -------
    np.ndarray, shape (n_frames, nrow, ncol)
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    if cam is not None:
        nrow, ncol = cam.nrow, cam.ncol
    if nrow is None or ncol is None:
        raise ValueError("Provide nrow/ncol or a CameraParams object")

    dt = np.dtype(dtype).newbyteorder(endian)
    pixels_per_frame = nrow * ncol
    total_bytes = bin_path.stat().st_size
    total_pixels = total_bytes // dt.itemsize
    total_frames = total_pixels // pixels_per_frame

    if total_pixels % pixels_per_frame != 0:
        raise ValueError(
            f"{bin_path.name}: {total_pixels} pixels not divisible by "
            f"{pixels_per_frame} (nrow={nrow}, ncol={ncol})"
        )

    # Slice selection
    end_frame = total_frames if n_frames is None else start_frame + n_frames
    end_frame = min(end_frame, total_frames)
    n_load = end_frame - start_frame

    print(
        f"{bin_path.name}: {total_frames} frames ({nrow}x{ncol}), "
        f"loading [{start_frame}:{end_frame}]"
    )

    mm = np.memmap(bin_path, mode="r", dtype=dt)
    sliced = mm[start_frame * pixels_per_frame : end_frame * pixels_per_frame]
    shaped = sliced.reshape(n_load, nrow, ncol)

    return shaped if memmap else np.asarray(shaped)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────────────────────────────────────────


def load_acquisition(
    folder: str | Path,
    mat_name: str = "output_data.mat",
    bin1_name: str = "frames1.bin",
    bin2_name: str = "frames2.bin",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray | None, CameraParams]:
    """
    Load a complete Luminos acquisition folder.

    Returns both frame stacks and the camera metadata for camera 1.
    frames2 (green channel) is None if the file is absent or empty
    (i.e. opto acquisition).

    Parameters
    ----------
    folder   : acquisition folder path
    mat_name : metadata filename (default: "metadata.mat")
    bin1_name: red camera binary (default: "frames1.bin")
    bin2_name: green camera binary (default: "frames2.bin")
    **kwargs : forwarded to load_bin (n_frames, start_frame, memmap, etc.)

    Returns
    -------
    (mov1, mov2, cam)
        mov1 : (n, nrow, ncol) uint16 — Voltron2 red channel
        mov2 : (n, nrow, ncol) uint16 — cAMPinG1 green channel, or None
        cam  : CameraParams for camera 1
    """
    folder = Path(folder)
    mat_path = folder / mat_name
    bin1_path = folder / bin1_name
    bin2_path = folder / bin2_name

    cam1 = load_camera_meta(mat_path, camera_index=1)
    mov1 = load_bin(bin1_path, cam=cam1, **kwargs)

    mov2 = None
    if bin2_path.exists() and bin2_path.stat().st_size > 0:
        try:
            cam2 = load_camera_meta(mat_path, camera_index=2)
        except IndexError:
            cam2 = cam1  # fall back if only one camera in metadata
        mov2 = load_bin(bin2_path, cam=cam2, **kwargs)
    else:
        print(f"{bin2_name} absent or empty — opto acquisition.")

    return mov1, mov2, cam1


# ─────────────────────────────────────────────────────────────────────────────
# Path utility
# ─────────────────────────────────────────────────────────────────────────────


def get_acq_paths(folder):
    """
    Resolve standard Luminos acquisition file paths from a folder and
    parse camera metadata for both channels.

    Returns a SimpleNamespace with attributes:
        .folder  : Path
        .mat     : Path  — output_data.mat
        .bin1    : Path  — frames1.bin  (Voltron2, red)
        .bin2    : Path  — frames2.bin  (cAMPinG1, green) or None if absent
        .cam1    : CameraParams for frames1
        .cam2    : CameraParams for frames2, or None if absent

    Each bin is paired with its own CameraParams so dimensions are always
    correct even when the two cameras use different ROIs.

    Example
    -------
    p = get_acq_paths("/data/Data/Optopatch/20260302/120108cAMPinG1-Test")
    mov1 = load_bin(p.bin1, cam=p.cam1)
    mov2 = load_bin(p.bin2, cam=p.cam2)   # uses cam2 ROI, not cam1
    mov1, mov2, _ = load_acquisition(p.folder)
    """
    from types import SimpleNamespace

    folder = Path(folder)
    mat = folder / "output_data.mat"
    bin1 = folder / "frames1.bin"
    bin2 = folder / "frames2.bin"

    cam1 = load_camera_meta(mat, camera_index=1)

    cam2 = None
    bin2_out = None
    if bin2.exists() and bin2.stat().st_size > 0:
        bin2_out = bin2
        try:
            cam2 = load_camera_meta(mat, camera_index=2)
        except IndexError:
            # Only one camera in metadata (opto acq) — fall back to cam1 dims
            cam2 = cam1

    return SimpleNamespace(
        folder=folder,
        mat=mat,
        bin1=bin1,
        bin2=bin2_out,
        cam1=cam1,
        cam2=cam2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Waveform reconstruction
# ─────────────────────────────────────────────────────────────────────────────


def _awfm_constant(t: np.ndarray, value: float = 1.0) -> np.ndarray:
    return np.full(len(t), value)


def _dwfm_constant(t: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Digital constant waveform; same shape semantics as awfm_constant."""
    return np.full(len(t), value, dtype=float)


def _awfm_pulse_single(t: np.ndarray, high_v=1, delay=0, pulse_duration=1) -> np.ndarray:
    """Port of awfm_pulse_single.m: one pulse of high_v for
    delay < t < delay + pulse_duration (strict on both edges), 0 elsewhere."""
    y = np.zeros(len(t))
    y[(t > delay) & (t < delay + pulse_duration)] = high_v
    return y


def _awfm_pulses_delay(
    t: np.ndarray, low_v=0, high_v=1, dlta=0.1, up_t=0.1, down_t=0.1, nstps=5, delay=0, repeat=0
) -> np.ndarray:
    rate = 1.0 / (t[1] - t[0])
    downsamps = round(down_t * rate)
    upsamps = round(up_t * rate)
    delay_samps = round(delay * rate)

    y = [low_v] * delay_samps
    for i in range(int(nstps)):
        y += [low_v] * downsamps
        y += [high_v + i * dlta] * upsamps

    y = np.array(y, dtype=float)

    if len(y) < len(t):
        if repeat:
            y = np.tile(y, len(t) // len(y))
        y = np.concatenate([y, np.full(len(t) - len(y), low_v)])
    if len(y) > len(t):
        y = y[: len(t)]

    y[-1] = low_v
    return y


def _normalize_awfm_pulses_params(params) -> np.ndarray:
    """
    Normalize saved awfm_pulses parameters to the MATLAB 8-argument form:

        low_v, high_v, dlta, up_t, down_t, nstps, load_cal, phase

    Luminos metadata from some acquisitions contains one extra legacy value
    after dlta:

        low_v, high_v, dlta, legacy, up_t, down_t, nstps, load_cal, phase

    That legacy value is ignored because it is not present in the MATLAB
    awfm_pulses function supplied with this library.
    """
    p = np.asarray(params, dtype=float).flatten()

    defaults = np.array([0, 1, 0.1, 0.1, 0.1, 5, -1, 0], dtype=float)

    if len(p) == 9:
        # Drop the legacy fourth saved value.
        p = p[[0, 1, 2, 4, 5, 6, 7, 8]]

    if len(p) > 8:
        raise TypeError(
            f"awfm_pulses expected at most 8 MATLAB parameters "
            f"(or the supported 9-value legacy layout), got {len(p)}"
        )

    out = defaults.copy()
    out[: len(p)] = p
    return out


def _awfm_pulses(t: np.ndarray, *params) -> np.ndarray:
    """
    Python port of MATLAB awfm_pulses.

    Supports both the current 8-parameter MATLAB signature and the 9-value
    legacy parameter layout stored in some Luminos metadata files.
    """
    if len(t) < 2:
        raise ValueError("awfm_pulses requires at least two time samples")

    low_v, high_v, dlta, up_t, down_t, nstps, load_cal, phase = _normalize_awfm_pulses_params(
        params
    )

    rate = 1.0 / (t[1] - t[0])

    downsamps = int(np.floor(down_t * rate + 0.5))
    upsamps = int(np.floor(up_t * rate + 0.5))
    nstps = int(round(nstps))

    parts = []
    for i in range(nstps):
        if downsamps:
            parts.append(np.full(downsamps, low_v, dtype=float))
        if upsamps:
            parts.append(np.full(upsamps, high_v + i * dlta, dtype=float))

    y = np.concatenate(parts) if parts else np.empty(0, dtype=float)

    # MATLAB:
    # y = circshift(y, floor((upsamps + downsamps) * phase));
    if y.size:
        shift = int(np.floor((upsamps + downsamps) * phase))
        y = np.roll(y, shift)

    # Exact behavior of the MATLAB function when called on a full t vector.
    if y.size < len(t):
        if y.size == 0:
            y = np.full(len(t), low_v, dtype=float)
        else:
            y = np.tile(y, len(t) // len(y))
            y = np.concatenate([y, np.full(len(t) - len(y), low_v, dtype=float)])

    if y.size > len(t):
        y = y[: len(t)]

    if y.size:
        y[-1] = low_v

    load_cal = int(round(load_cal))
    if load_cal in (1, 2, 3, 4, 5, 6, 7):
        raise RuntimeError(
            "awfm_pulses MATLAB calibration fits (load_cal=1..7) are not yet ported to Python."
        )

    return y


def _awfm_upRamp(t: np.ndarray, rest_time=0.5, ramp_time=0.5, minval=0, maxval=1) -> np.ndarray:
    """
    Python port of MATLAB awfm_upRamp.

    Builds one waveform block of length rest_time + ramp_time:
      - a linear ramp from minval to maxval over ramp_time
      - followed by minval for rest_time

    The block is then tiled to fill t, with any final partial remainder
    padded using minval, matching MATLAB tile_and_pad(y, t, minval).
    """
    if len(t) < 2:
        raise ValueError("awfm_upRamp requires at least two time samples")

    rate = 1.0 / (t[1] - t[0])

    # MATLAB round() for non-negative sample counts.
    rest_samps = int(np.floor(rest_time * rate + 0.5))
    ramp_samps = int(np.floor(ramp_time * rate + 0.5))

    block_len = rest_samps + ramp_samps
    if block_len <= 0:
        return np.full(len(t), minval, dtype=float)

    y = np.full(block_len, minval, dtype=float)

    # MATLAB: y(1:ramp_samps) = linspace(minval, maxval, ramp_samps)
    if ramp_samps > 0:
        y[:ramp_samps] = np.linspace(minval, maxval, ramp_samps)

    # MATLAB: y = tile_and_pad(y, t, minval)
    if len(y) < len(t):
        copies = len(t) // len(y)
        y = np.tile(y, copies)
        if len(y) < len(t):
            y = np.concatenate([y, np.full(len(t) - len(y), minval, dtype=float)])
    elif len(y) > len(t):
        y = y[: len(t)]

    return y


def _awfm_pulses_manual_PP(
    t: np.ndarray,
    low_v=0,
    high_v=1,
    dlta=0.1,
    up_t=0.1,
    down_t=0.1,
    nstps=5,
    phase=0,
    t_before=1,
    rmphght=1,
    rampt=np.nan,
) -> np.ndarray:
    """
    Python port of MATLAB awfm_pulses_manual_PP (staircase + end ramp).

    Layout (all preceded by t_before * 1 s of low):
      - nstps pulses of up_t s at levels high_v + i*dlta (i = 0..nstps-1),
        separated by (down_t - up_t) s at low_v
      - 1 s at zero
      - a rampt-s linear ramp from low_v to rmphght
    Zero-padded / truncated to len(t), then circularly shifted by
    floor((upsamps + downsamps) * phase) samples, matching MATLAB.
    """
    if len(t) < 2:
        raise ValueError("awfm_pulses_manual_PP requires at least two samples")
    rate = 1.0 / t[1]
    if rampt is None or (np.isscalar(rampt) and np.isnan(rampt)):
        rampt = 0.3 * (t[-1] + t[1])

    empty = int(round(rate))  # 1 s of zeros
    downsamps = int(round(down_t * rate))
    upsamps = int(round(up_t * rate))
    rampsamps = int(round(rampt * rate))
    nstps = int(round(nstps))

    parts = [np.zeros(int(empty * t_before))]
    for i in range(nstps - 1):
        parts.append(np.full(upsamps, high_v + i * dlta, dtype=float))
        parts.append(np.full(max(downsamps - upsamps, 0), low_v, dtype=float))
    parts.append(np.full(upsamps, high_v + (nstps - 1) * dlta, dtype=float))
    parts.append(np.zeros(empty))
    parts.append(np.linspace(low_v, rmphght, rampsamps))
    y = np.concatenate(parts)

    y = np.roll(y, int(np.floor((upsamps + downsamps) * phase)))

    if len(y) < len(t):
        y = np.concatenate([y, np.zeros(len(t) - len(y))])
    return y[: len(t)]


_WAVEFORM_REGISTRY: dict[str, callable] = {
    "awfm_constant": _awfm_constant,
    "dwfm_constant": _dwfm_constant,
    "awfm_pulses_delay": _awfm_pulses_delay,
    "awfm_pulses": _awfm_pulses,
    "awfm_upRamp": _awfm_upRamp,
    "awfm_pulses_manual_PP": _awfm_pulses_manual_PP,
    "awfm_pulse_single": _awfm_pulse_single,
    # Add more ports here as needed
}


def _native_waveform_samples(wavefile: str, params: np.ndarray, sample_rate: float) -> int | None:
    """Length of one un-tiled block for a known waveform function."""
    p = np.asarray(params, dtype=float).flatten()

    def time_to_samples(seconds):
        return int(np.floor(float(seconds) * sample_rate + 0.5))

    if wavefile == "awfm_pulses":
        q = _normalize_awfm_pulses_params(p)
        up_t, down_t, nstps = q[3], q[4], int(round(q[5]))

        # One pulse train is:
        #
        #   low -> pulse -> low -> pulse -> ... -> low -> pulse
        #
        # The MATLAB function itself ends immediately after the final pulse
        # (apart from forcing the last sample low).  When this waveform is
        # used as a segment before another waveform, preserve one final
        # down_t low interval so the next segment does not begin immediately
        # at the falling edge of the last pulse.
        return nstps * (time_to_samples(up_t) + time_to_samples(down_t)) + time_to_samples(down_t)

    if wavefile == "awfm_upRamp":
        defaults = np.array([0.5, 0.5, 0, 1], dtype=float)
        q = defaults.copy()
        q[: min(len(p), 4)] = p[:4]
        rest_time, ramp_time = q[0], q[1]
        return time_to_samples(rest_time) + time_to_samples(ramp_time)

    if wavefile == "awfm_pulses_delay":
        defaults = np.array([0, 1, 0.1, 0.1, 0.1, 5, 0, 0], dtype=float)
        q = defaults.copy()
        q[: min(len(p), 8)] = p[:8]
        up_t, down_t = q[3], q[4]
        nstps, delay = int(round(q[5])), q[6]
        return time_to_samples(delay) + nstps * (time_to_samples(up_t) + time_to_samples(down_t))

    # Constants do not have a finite native block.
    if wavefile == "awfm_constant":
        return None

    return None


def _waveform_baseline(wavefile: str, params: np.ndarray) -> float:
    """Return the resting/final value of a waveform."""
    p = np.asarray(params, dtype=float).flatten()

    if wavefile == "awfm_pulses":
        return float(_normalize_awfm_pulses_params(p)[0])

    if wavefile == "awfm_pulses_delay":
        return float(p[0]) if len(p) else 0.0

    if wavefile == "awfm_upRamp":
        return float(p[2]) if len(p) >= 3 else 0.0

    if wavefile == "awfm_constant":
        return float(p[0]) if len(p) else 1.0

    return 0.0


def _reconstruct_native(wavefile: str, params: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Reconstruct exactly one native block, without allowing the waveform's
    internal tile-and-pad logic to repeat it.
    """
    n = _native_waveform_samples(wavefile, params, sample_rate)
    if n is None:
        raise ValueError(f"Waveform {wavefile!r} has no finite native block length")
    if n <= 0:
        return np.empty(0, dtype=float)

    return _reconstruct(wavefile, params, sample_rate, n)


def _compose_named_waveforms(specs, sample_rate: float, n_samples: int) -> np.ndarray:
    """
    Compose repeated metadata entries for one physical DAQ output.

    Earlier entries are placed as one native segment each, in metadata order.
    The LAST entry receives the entire remaining acquisition window and is
    reconstructed normally over that window.

    This is important for MATLAB waveform functions such as awfm_upRamp,
    which call tile_and_pad internally: if enough time remains, the ramp will
    repeat; if less time remains, only one ramp (plus baseline padding) will
    appear.
    """
    pieces = []
    used = 0

    for idx, (_, _, wavefile, params) in enumerate(specs):
        remaining = n_samples - used
        if remaining <= 0:
            break

        is_last = idx == len(specs) - 1

        if is_last:
            # Give the final waveform all remaining time so its own MATLAB
            # tile/pad behavior determines whether it appears once, twice,
            # or more.
            piece = _reconstruct(
                wavefile,
                params,
                sample_rate,
                remaining,
            )
        else:
            # Earlier waveform definitions represent one segment only.
            piece = _reconstruct_native(
                wavefile,
                params,
                sample_rate,
            )
            piece = piece[:remaining]

        pieces.append(piece)
        used += len(piece)

    y = np.concatenate(pieces) if pieces else np.empty(0, dtype=float)

    if len(y) < n_samples:
        final_wavefile = specs[-1][2]
        final_params = specs[-1][3]
        baseline = _waveform_baseline(final_wavefile, final_params)
        y = np.concatenate([y, np.full(n_samples - len(y), baseline, dtype=float)])

    return y[:n_samples]


@dataclass
class Waveform:
    """A single reconstructed DAQ output waveform."""

    name: str
    channel: str  # 'ao' or 'do'
    port: str
    wavefile: str
    params: np.ndarray
    sample_rate: float
    data: np.ndarray  # shape (n_samples,)

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.sample_rate

    def __repr__(self):
        duration = len(self.data) / self.sample_rate
        return (
            f"Waveform({self.name!r}, {self.wavefile}({self.params}), "
            f"{self.sample_rate / 1000:.0f} kHz, {duration:.1f} s)"
        )


def _reconstruct(
    wavefile: str, params: np.ndarray, sample_rate: float, n_samples: int
) -> np.ndarray:
    """Evaluate a named waveform function at the given sample rate."""
    if wavefile not in _WAVEFORM_REGISTRY:
        raise NotImplementedError(
            f"Waveform function '{wavefile}' not yet ported. Known: {list(_WAVEFORM_REGISTRY)}"
        )
    t = np.arange(n_samples) / sample_rate
    return _WAVEFORM_REGISTRY[wavefile](t, *params.flatten())


def _bt_scalar(f: h5py.File, device: h5py.Group, field: str, idx: int = 0) -> float:
    """Read a scalar from buffered_tasks (handles nested object refs)."""
    bt_refs = device["buffered_tasks"][()]
    bt = f[bt_refs.ravel()[idx]]
    raw = bt[field][()]
    if raw.dtype.kind == "O":
        raw = f[raw.ravel()[0]][()]
    return float(np.asarray(raw).flat[0])


def _daq_device(f: h5py.File) -> h5py.Group:
    """Return the DAQ device group from #refs#/s."""
    return f["#refs#"]["s"]


def _gp_scalar(daq: h5py.Group, field: str) -> float:
    """Read a float scalar from global_props."""
    return float(daq["global_props"][field][()].flat[0])


def _gp_str(daq: h5py.Group, field: str) -> str:
    """Read a string from global_props (stored as uint16 char array)."""
    return "".join(chr(int(c)) for c in daq["global_props"][field][()].flat)


def _deref_matlab_string(f: h5py.File, refs_arr: np.ndarray, idx: int) -> str:
    """Dereference a MATLAB HDF5 string stored through an object-ref array."""
    ref = refs_arr.ravel()[idx]
    raw = f[ref][()]
    return "".join(chr(int(c)) for c in np.asarray(raw).flat).strip()


def _deref_matlab_value(f: h5py.File, refs_arr: np.ndarray, idx: int):
    """Dereference one numeric MATLAB value from an object-ref array."""
    ref = refs_arr.ravel()[idx]
    return np.asarray(f[ref][()]).squeeze()


def _concat_time_seconds(value) -> float:
    """
    Convert a saved concatTime value to seconds.

    Luminos uses [0, 0] uint64 placeholders when concatTime is not relevant.
    """
    a = np.asarray(value).squeeze()
    if a.size == 0:
        return 0.0
    if a.size == 1:
        return float(a.flat[0])
    if np.all(a == 0):
        return 0.0
    return float(a.flat[0])


def _apply_waveform_operation(
    existing: np.ndarray | None,
    incoming: np.ndarray,
    operation: str,
    concat_time: float,
    sample_rate: float,
) -> np.ndarray:
    """Apply the operation stored in Luminos wfm_data."""
    op = operation.strip().lower()

    # The first definition for a named output establishes its waveform.
    # Its saved operation describes how the GUI/task builder created it, but
    # there is no prior same-name waveform to combine with here.
    if existing is None:
        return incoming.copy()

    if op == "addition":
        return existing + incoming

    if op == "multiplication":
        return existing * incoming

    if op == "concatenation":
        cut = int(np.floor(concat_time * sample_rate + 0.5))
        cut = max(0, min(cut, len(existing)))

        # Concatenation starts the NEW waveform at local time zero.
        #
        # MATLAB-style interpretation:
        #   out = [existing before concatTime,
        #          beginning of incoming waveform]
        #
        # Do NOT use incoming[cut:], because that jumps concat_time seconds
        # into the incoming waveform and can make the first ramp look cut off.
        n_remaining = len(existing) - cut

        out = existing.copy()
        out[cut:] = incoming[:n_remaining]
        return out

    # Be explicit rather than silently guessing a combination rule.
    raise ValueError(f"Unknown waveform operation {operation!r}")


def load_waveforms(
    mat_path: str | Path,
    channels: tuple[str, ...] = ("ao", "do"),
) -> dict[str, Waveform]:
    """
    Reconstruct DAQ outputs using the operation metadata saved by Luminos.

    Each waveform function is first evaluated over the full acquisition time
    vector, matching MATLAB. Repeated definitions for the same physical output
    are then combined using wfm_data.operation and wfm_data.concatTime.

    Supported operations:
        Addition
        Multiplication
        Concatenation
    """
    mat_path = Path(mat_path)
    result: dict[str, Waveform] = {}

    with h5py.File(mat_path, "r") as f:
        daq = _daq_device(f)
        gp = daq["global_props"]

        sample_rate = float(gp["rate"][()].flat[0])
        total_time = float(gp["total_time"][()].flat[0])
        n_samples = int(round(sample_rate * total_time))

        print(f"DAQ: {sample_rate / 1000:.0f} kHz, {n_samples} samples, {total_time:.1f} s")

        wfm_data = daq["wfm_data"]

        # Accumulate data and metadata by physical output name while walking
        # the saved entries in their original order.
        accumulated: dict[str, np.ndarray] = {}
        metadata: dict[str, dict] = {}

        for ch in channels:
            if ch not in wfm_data:
                continue

            grp = wfm_data[ch]
            if not isinstance(grp, h5py.Group) or "wavefile" not in grp:
                continue

            wavefile_refs = grp["wavefile"][()]
            params_refs = grp["params"][()]
            name_refs = grp["name"][()]
            port_refs = grp["port"][()]

            operation_refs = grp["operation"][()] if "operation" in grp else None
            concat_refs = grp["concatTime"][()] if "concatTime" in grp else None

            n_entries = wavefile_refs.size

            for i in range(n_entries):
                name_str = _deref_matlab_string(f, name_refs, i)
                port_str = _deref_matlab_string(f, port_refs, i)
                wavefile_str = _deref_matlab_string(f, wavefile_refs, i)

                params = np.asarray(
                    _deref_matlab_value(f, params_refs, i),
                    dtype=float,
                ).reshape(-1)

                operation = (
                    _deref_matlab_string(f, operation_refs, i)
                    if operation_refs is not None
                    else "Addition"
                )

                concat_time = (
                    _concat_time_seconds(_deref_matlab_value(f, concat_refs, i))
                    if concat_refs is not None
                    else 0.0
                )

                try:
                    # Reconstruct a full waveform template. For
                    # Concatenation, _apply_waveform_operation takes the
                    # BEGINNING of this template at concatTime, so the new
                    # waveform starts at local t=0.
                    incoming = _reconstruct(
                        wavefile_str,
                        params,
                        sample_rate,
                        n_samples,
                    )

                    combined = _apply_waveform_operation(
                        accumulated.get(name_str),
                        incoming,
                        operation,
                        concat_time,
                        sample_rate,
                    )
                    accumulated[name_str] = combined

                    if name_str not in metadata:
                        metadata[name_str] = {
                            "channel": ch,
                            "port": port_str,
                            "wavefiles": [],
                            "params": [],
                            "operations": [],
                            "concat_times": [],
                        }

                    m = metadata[name_str]
                    m["wavefiles"].append(wavefile_str)
                    m["params"].append(params)
                    m["operations"].append(operation)
                    m["concat_times"].append(concat_time)

                    extra = (
                        f", concat={concat_time:g}s"
                        if operation.strip().lower() == "concatenation"
                        else ""
                    )
                    print(
                        f"  {name_str!r}: {operation} {wavefile_str}({np.round(params, 3)}){extra}"
                    )

                except (NotImplementedError, RuntimeError, ValueError, TypeError) as e:
                    print(f"  Warning: {name_str!r} / {wavefile_str}: {e}")
                    if name_str not in accumulated:
                        accumulated[name_str] = np.full(n_samples, np.nan)

        for name_str, data in accumulated.items():
            m = metadata.get(name_str, {})
            wavefiles = m.get("wavefiles", ["unknown"])
            param_sets = m.get("params", [])

            # Preserve numeric params so existing plotting code using
            # np.round(wfm.params, 3) still works. NaN separates definitions.
            joined = []
            for j, p in enumerate(param_sets):
                if j:
                    joined.append(np.array([np.nan]))
                joined.append(np.asarray(p, dtype=float).reshape(-1))
            display_params = np.concatenate(joined) if joined else np.array([], dtype=float)

            display_wavefile = " -> ".join(wavefiles)

            wfm = Waveform(
                name=name_str,
                channel=m.get("channel", ""),
                port=m.get("port", ""),
                wavefile=display_wavefile,
                params=display_params,
                sample_rate=sample_rate,
                data=data,
            )
            print(f"  {wfm}")
            result[name_str] = wfm

    return result

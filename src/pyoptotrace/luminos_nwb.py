"""
bin2nwb.py
----------
Converts a single Luminos acquisition folder to an NWB file.

Luminos folder structure expected:
    <acq_folder>/
        metadata.mat      (MATLAB v7.3 / HDF5)
        frames1.bin       (Voltron2, red camera, always present)
        frames2.bin       (cAMPinG1 green camera, absent during bPAC stim)

Usage:
    python bin2nwb.py <acquisition_folder> [options]

    python bin2nwb.py ./stim_001
    python bin2nwb.py ./baseline_001 --subject mouse_01 --session 2025-10-10
    python bin2nwb.py ./stim_001 --acq-type opto
    python bin2nwb.py ./baseline_001 --rig-config /path/to/rig_config.yaml

Options:
    --subject       Subject/animal ID (default: "unknown")
    --session       Session date string YYYY-MM-DD (default: parsed from metadata)
    --acq-type      One of: imaging, opto, mixed (default: auto-detected)
    --rig-config    Path to rig_config.yaml (default: rig_config.yaml next to this script)
    --output        Output .nwb path (default: <acq_folder>/<folder_name>.nwb)
    --description   Free-text experiment description

Requirements:
    pip install pynwb h5py numpy pyyaml
"""

import argparse
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Optional imports are lazy so importing this module does not initialize PyNWB
# caches or fail when only the core package is installed.
try:
    import yaml
except ImportError:  # optional I/O dependency
    yaml = None

try:
    import h5py
except ImportError:  # optional I/O dependency
    h5py = None

DynamicTable = None
NWBHDF5IO = None
NWBFile = None
OnePhotonSeries = None
OpticalChannel = None
OptogeneticSeries = None
OptogeneticStimulusSite = None


def _require_h5py():
    if h5py is None:
        raise ModuleNotFoundError(
            "Luminos metadata parsing requires the 'io' extra: pip install 'pyoptotrace[io]'"
        )


def _require_nwb():
    global DynamicTable, NWBHDF5IO, NWBFile, OnePhotonSeries, OpticalChannel
    global OptogeneticSeries, OptogeneticStimulusSite

    _require_h5py()
    if NWBFile is None:
        try:
            from hdmf.common import DynamicTable as _DynamicTable
            from pynwb import NWBHDF5IO as _NWBHDF5IO
            from pynwb import NWBFile as _NWBFile
            from pynwb.ogen import OptogeneticSeries as _OptogeneticSeries
            from pynwb.ogen import OptogeneticStimulusSite as _OptogeneticStimulusSite
            from pynwb.ophys import OnePhotonSeries as _OnePhotonSeries
            from pynwb.ophys import OpticalChannel as _OpticalChannel
        except ImportError as exc:
            raise ModuleNotFoundError(
                "NWB export requires the 'io' extra: pip install 'pyoptotrace[io]'"
            ) from exc

        DynamicTable = _DynamicTable
        NWBHDF5IO = _NWBHDF5IO
        NWBFile = _NWBFile
        OnePhotonSeries = _OnePhotonSeries
        OpticalChannel = _OpticalChannel
        OptogeneticSeries = _OptogeneticSeries
        OptogeneticStimulusSite = _OptogeneticStimulusSite


def _require_yaml():
    if yaml is None:
        raise ModuleNotFoundError(
            "Rig configuration requires the 'io' extra: pip install 'pyoptotrace[io]'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ─────────────────────────────────────────────────────────────────────────────


def load_metadata(mat_path: Path) -> dict:
    """
    Load Luminos metadata.mat (MATLAB v7.3 / HDF5) and return a flat dict
    of the fields we care about.
    """
    _require_h5py()
    meta = {}

    with h5py.File(mat_path, "r") as f:

        def _scalar(ds):
            v = ds[()]
            if isinstance(v, np.ndarray):
                v = v.flat[0]
            return v

        def _str(ds):
            """Decode HDF5 string dataset."""
            v = ds[()]
            if isinstance(v, bytes):
                return v.decode()
            if isinstance(v, np.ndarray):
                return "".join(chr(int(c)) for c in v.flat)
            return str(v)

        # ── rig name ──────────────────────────────────────────────────────────
        if "rigName" in f:
            meta["rig_name"] = _str(f["rigName"])

        # ── git info ──────────────────────────────────────────────────────────
        if "gitInfo" in f:
            meta["git_info"] = {}
            gi = f["gitInfo"]
            for k in gi.keys():
                try:
                    meta["git_info"][k] = _str(gi[k])
                except Exception:
                    pass

        # ── Device_Data ───────────────────────────────────────────────────────
        # Device_Data is a cell array of device structs; iterate until we find
        # the camera entries (deviceType == 'camera') and extract per-camera params.
        cameras = []
        if "Device_Data" in f:
            dd = f["Device_Data"]
            # HDF5 cell arrays are stored as object references
            refs = dd[()]
            if refs.ndim == 1:
                refs = refs.reshape(1, -1)
            for ref in refs.flat:
                try:
                    dev = f[ref]
                    dtype = _str(dev["deviceType"]).lower() if "deviceType" in dev else ""
                    if "cam" in dtype or "camera" in dtype:
                        cam = {}
                        for field in [
                            "cam_id",
                            "exposuretime",
                            "ROI",
                            "bin",
                            "frames_requested",
                            "acq_complete",
                            "dropped_frames",
                            "hsync_rate",
                            "daqtrig_period_ms",
                            "microns_per_pixel",
                            "readout_mode",
                            "frametrigger_source",
                        ]:
                            if field in dev:
                                try:
                                    cam[field] = _scalar(dev[field])
                                except Exception:
                                    pass
                        if "ROI" in dev:
                            cam["ROI"] = dev["ROI"][()].flatten().tolist()
                        if "name" in dev:
                            cam["name"] = _str(dev["name"])
                        cameras.append(cam)
                except Exception:
                    continue

        meta["cameras"] = cameras

        # ── DAQ timing ────────────────────────────────────────────────────────
        # daqtrig_period_ms from camera gives frame interval
        # Pull from first camera that has it
        meta["daqtrig_period_ms"] = None
        for cam in cameras:
            if "daqtrig_period_ms" in cam and cam["daqtrig_period_ms"]:
                meta["daqtrig_period_ms"] = float(cam["daqtrig_period_ms"])
                break

        # ── DMD / patterning ──────────────────────────────────────────────────
        meta["has_patterning"] = "Patterning_ROI_Stack" in f or "pattern_stack" in f

        # ── Archive index (folder/session index) ──────────────────────────────
        if "Archive_Index" in f:
            try:
                meta["archive_index"] = int(_scalar(f["Archive_Index"]))
            except Exception:
                pass

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Binary frame reading
# ─────────────────────────────────────────────────────────────────────────────


def read_bin(bin_path: Path, height: int, width: int) -> np.ndarray:
    """
    Read a Luminos raw binary file.

    Frames are stored as a flat sequence of uint16 values, no header.
    Shape returned: (n_frames, height, width)
    """
    data = np.fromfile(bin_path, dtype=np.uint16)
    pixels_per_frame = height * width
    n_frames = len(data) // pixels_per_frame
    if n_frames == 0:
        raise ValueError(f"No complete frames found in {bin_path}")
    data = data[: n_frames * pixels_per_frame]
    # MATLAB stores in column-major (Fortran) order; frames are rows×cols
    frames = data.reshape((n_frames, height, width))
    return frames


def roi_to_hw(roi, binning: int = 1):
    """
    Convert Luminos ROI [x_start, width, y_start, height] to (height, width)
    accounting for binning.
    """
    # ROI format from metadata: [x_offset, roi_width, y_offset, roi_height]
    roi_width = int(roi[1]) // binning
    roi_height = int(roi[3]) // binning
    return roi_height, roi_width


# ─────────────────────────────────────────────────────────────────────────────
# NWB construction
# ─────────────────────────────────────────────────────────────────────────────


def build_nwb(
    acq_folder: Path,
    rig_cfg: dict,
    subject_id: str,
    session_str: str,
    acq_type: str,
    description: str,
    output_path: Path,
):
    _require_nwb()
    meta_path = acq_folder / "metadata.mat"
    frames1_path = acq_folder / "frames1.bin"
    frames2_path = acq_folder / "frames2.bin"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.mat not found in {acq_folder}")
    if not frames1_path.exists():
        raise FileNotFoundError(f"frames1.bin not found in {acq_folder}")

    print(f"Loading metadata from {meta_path} ...")
    meta = load_metadata(meta_path)

    # ── Session start time ────────────────────────────────────────────────────
    try:
        session_start = datetime.strptime(session_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        warnings.warn(f"Could not parse session date '{session_str}', using now()")
        session_start = datetime.now(timezone.utc)

    # ── Frame rate ────────────────────────────────────────────────────────────
    fps = None
    if meta.get("daqtrig_period_ms"):
        fps = 1000.0 / meta["daqtrig_period_ms"]
        print(f"Frame rate from metadata: {fps:.2f} fps")
    else:
        warnings.warn(
            "daqtrig_period_ms not found in metadata. "
            "Frame rate will be set to NaN — edit the NWB file to correct this."
        )
        fps = float("nan")

    # ── Camera dimensions ─────────────────────────────────────────────────────
    # Use first camera in metadata for dimensions; both cameras should match
    # since the ROI is set on the rig jointly.
    cam_meta = meta["cameras"][0] if meta["cameras"] else {}
    binning = int(cam_meta.get("bin", 1))
    roi = cam_meta.get("ROI", [0, 2048, 0, 2048])  # fallback full frame
    height, width = roi_to_hw(roi, binning)
    print(f"Frame dimensions: {height}×{width} px (binning={binning})")

    # ── Auto-detect acquisition type ──────────────────────────────────────────
    has_frames2 = frames2_path.exists() and frames2_path.stat().st_size > 0
    if acq_type == "auto":
        if meta.get("has_patterning") and not has_frames2:
            acq_type = "opto"
        elif has_frames2:
            acq_type = "imaging"
        else:
            acq_type = "imaging"
        print(f"Auto-detected acquisition type: {acq_type}")
    else:
        print(f"Acquisition type: {acq_type}")

    # ── NWBFile ───────────────────────────────────────────────────────────────
    nwbfile = NWBFile(
        session_description=description or f"Luminos acquisition — {acq_type}",
        identifier=f"{subject_id}_{session_str}_{acq_folder.name}",
        session_start_time=session_start,
        lab="",  # fill in
        institution="",  # fill in
        experiment_description=(
            f"Simultaneous Voltron2 voltage imaging (frames1) and "
            f"{'cAMPinG1 cAMP imaging' if acq_type != 'opto' else 'bPAC optogenetic stimulation'} "
            f"(frames2/DMD). Rig: {meta.get('rig_name', rig_cfg.get('rig_name', 'unknown'))}."
        ),
        keywords=["voltage imaging", "Voltron2", "cAMPinG1", "bPAC", "optogenetics"],
    )

    # ── Devices ───────────────────────────────────────────────────────────────
    red_camera_device = nwbfile.create_device(
        name="red_camera",
        description="Hamamatsu sCMOS — Voltron2 channel (frames1.bin)",
    )
    green_camera_device = nwbfile.create_device(
        name="green_camera",
        description="Hamamatsu sCMOS — cAMPinG1 / bPAC channel (frames2.bin)",
    )
    dmd_device = nwbfile.create_device(
        name="DMD",
        description=(
            "Digital Micromirror Device. Shared path with green camera. "
            "Used for structured illumination (cAMPinG1) and bPAC optogenetic activation."
        ),
    )

    # ── Optical channels ──────────────────────────────────────────────────────
    rc = rig_cfg["red_camera"]
    gc = rig_cfg["green_camera"]

    red_optical_channel = OpticalChannel(
        name="Voltron2_red",
        description=(
            f"{rc['indicator']} ({rc['indicator_type']}). "
            f"Excitation: {rc['excitation_nm']} nm. "
            f"Emission filter: {rc['emission_filter']}."
        ),
        emission_lambda=float(rc["emission_lambda_nm"]),
    )

    green_optical_channel = OpticalChannel(
        name="cAMPinG1_green",
        description=(
            f"{gc['indicator']} ({gc['indicator_type']}). "
            f"Excitation: {gc['excitation_nm']} nm. "
            f"Emission filter: {gc['emission_filter']}."
        ),
        emission_lambda=float(gc["emission_lambda_nm"]),
    )

    # ── Imaging planes ────────────────────────────────────────────────────────
    microns_per_pixel = float(cam_meta.get("microns_per_pixel", float("nan")))

    red_imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane_Voltron2",
        optical_channel=red_optical_channel,
        imaging_rate=fps,
        description="Voltron2 voltage imaging plane",
        device=red_camera_device,
        excitation_lambda=float(rc["excitation_nm"]),
        indicator=rc["indicator"],
        location="unknown",  # update with brain region
        grid_spacing=[microns_per_pixel, microns_per_pixel],
        grid_spacing_unit="micrometers",
    )

    green_imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane_cAMPinG1",
        optical_channel=green_optical_channel,
        imaging_rate=fps if acq_type != "opto" else float("nan"),
        description="cAMPinG1 cAMP imaging plane (inactive during bPAC opto acquisitions)",
        device=green_camera_device,
        excitation_lambda=float(gc["excitation_nm"]),
        indicator=gc["indicator"],
        location="unknown",  # update with brain region
        grid_spacing=[microns_per_pixel, microns_per_pixel],
        grid_spacing_unit="micrometers",
    )

    # ── frames1 → Voltron2 OnePhotonSeries ───────────────────────────────────
    print("Reading frames1.bin ...")
    frames1 = read_bin(frames1_path, height, width)
    print(f"  frames1 shape: {frames1.shape}")

    voltron_series = OnePhotonSeries(
        name="Voltron2",
        data=frames1,
        imaging_plane=red_imaging_plane,
        rate=fps,
        starting_time=0.0,
        unit="arbitrary units",
        description="Raw Voltron2 voltage imaging frames (uint16)",
        comments=(
            f"Source: frames1.bin. "
            f"Dropped frames reported in metadata: {cam_meta.get('dropped_frames', 'unknown')}. "
            f"Exposure time: {cam_meta.get('exposuretime', 'unknown')} s."
        ),
    )
    nwbfile.add_acquisition(voltron_series)

    # ── frames2 → cAMPinG1 OnePhotonSeries (imaging only) ────────────────────
    if acq_type != "opto" and has_frames2:
        print("Reading frames2.bin ...")
        frames2 = read_bin(frames2_path, height, width)
        print(f"  frames2 shape: {frames2.shape}")

        camping_series = OnePhotonSeries(
            name="cAMPinG1",
            data=frames2,
            imaging_plane=green_imaging_plane,
            rate=fps,
            starting_time=0.0,
            unit="arbitrary units",
            description="Raw cAMPinG1 cAMP imaging frames (uint16)",
            comments="Source: frames2.bin.",
        )
        nwbfile.add_acquisition(camping_series)
    elif acq_type == "opto":
        print("Opto acquisition — skipping frames2 (bPAC stimulation, no camera frames)")

    # ── Optogenetic stimulus (opto acquisitions) ──────────────────────────────
    if acq_type == "opto" or (acq_type == "mixed" and meta.get("has_patterning")):
        og = rig_cfg["optogenetics"]

        ogen_site = OptogeneticStimulusSite(
            name="bPAC_site",
            device=dmd_device,
            description=(
                f"{og['opsin']} photoactivatable adenylyl cyclase. "
                f"Blue DMD pattern ({og['stimulation_nm']} nm). "
                f"Shared illumination path with cAMPinG1 — mutually exclusive per acquisition."
            ),
            excitation_lambda=float(og["stimulation_nm"]),
            location="unknown",  # update with target region
        )
        nwbfile.add_ogen_site(ogen_site)

        # Stimulus waveform placeholder — replace with actual DAQ waveform when
        # parsing of DQ_CO_Ticks / DQ_Buffered_Task is implemented
        # For now: single pulse covering the whole acquisition
        n_voltron_frames = frames1.shape[0]
        stim_duration_s = n_voltron_frames / fps if not np.isnan(fps) else 0.0

        ogen_series = OptogeneticSeries(
            name="bPAC_stimulus",
            data=np.ones(2, dtype=np.float32),  # placeholder: on for full acq
            timestamps=np.array([0.0, stim_duration_s]),
            site=ogen_site,
            unit="watts",
            description=(
                "bPAC DMD stimulation. "
                "NOTE: this is a placeholder waveform (constant on). "
                "Replace with actual DAQ waveform from DQ_CO_Ticks once parsing is implemented."
            ),
        )
        nwbfile.add_stimulus(ogen_series)

    # ── Acquisition metadata as lab_meta_data ─────────────────────────────────
    # Store key acquisition parameters as a simple table for traceability
    meta_table = DynamicTable(
        name="luminos_acquisition_metadata",
        description="Key acquisition parameters extracted from Luminos metadata.mat",
    )
    fields = {
        "rig_name": meta.get("rig_name", "unknown"),
        "acq_type": acq_type,
        "binning": str(binning),
        "roi": str(roi),
        "frame_height_px": str(height),
        "frame_width_px": str(width),
        "fps": str(round(fps, 4)) if not np.isnan(fps) else "nan",
        "daqtrig_period_ms": str(meta.get("daqtrig_period_ms", "unknown")),
        "exposure_s": str(cam_meta.get("exposuretime", "unknown")),
        "frames_requested": str(cam_meta.get("frames_requested", "unknown")),
        "dropped_frames": str(cam_meta.get("dropped_frames", "unknown")),
        "microns_per_pixel": str(microns_per_pixel),
        "has_dmd_pattern": str(meta.get("has_patterning", False)),
        "red_emission_filter": rc["emission_filter"],
        "green_emission_filter": gc["emission_filter"],
        "source_folder": str(acq_folder.resolve()),
    }
    meta_table.add_column(name="key", description="Parameter name", data=list(fields.keys()))
    meta_table.add_column(name="value", description="Parameter value", data=list(fields.values()))
    meta_table.add_row(id=0)  # required by NWB DynamicTable
    # NWB doesn't have a standard slot for free metadata tables at top level,
    # so we attach it to the processing module
    proc = nwbfile.create_processing_module(
        name="acquisition_metadata",
        description="Luminos acquisition parameters from metadata.mat",
    )
    proc.add(meta_table)

    # ── Write ─────────────────────────────────────────────────────────────────
    print(f"Writing NWB file to {output_path} ...")
    with NWBHDF5IO(str(output_path), "w") as io:
        io.write(nwbfile)
    print(f"Done. File size: {output_path.stat().st_size / 1e6:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Convert a Luminos acquisition folder to NWB.")
    p.add_argument("folder", help="Path to acquisition folder")
    p.add_argument("--subject", default="unknown", help="Subject/animal ID")
    p.add_argument("--session", default=None, help="Session date YYYY-MM-DD")
    p.add_argument(
        "--acq-type",
        default="auto",
        choices=["auto", "imaging", "opto", "mixed"],
        help="Acquisition type (default: auto-detect)",
    )
    p.add_argument(
        "--rig-config",
        default=None,
        help="Path to rig_config.yaml (default: alongside this script)",
    )
    p.add_argument("--output", default=None, help="Output .nwb path")
    p.add_argument("--description", default="", help="Experiment description")
    return p.parse_args()


def main():
    _require_yaml()
    args = parse_args()
    acq_folder = Path(args.folder).resolve()

    if not acq_folder.is_dir():
        raise SystemExit(f"Not a directory: {acq_folder}")

    # ── Load rig config ───────────────────────────────────────────────────────
    if args.rig_config:
        cfg_path = Path(args.rig_config)
    else:
        cfg_path = Path(__file__).parent / "rig_config.yaml"

    if not cfg_path.exists():
        raise SystemExit(
            f"rig_config.yaml not found at {cfg_path}\n"
            "Create one from the template or pass --rig-config <path>"
        )
    with open(cfg_path) as f:
        rig_cfg = yaml.safe_load(f)

    # ── Session date ──────────────────────────────────────────────────────────
    session_str = args.session or datetime.now().strftime("%Y-%m-%d")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = acq_folder / f"{acq_folder.name}.nwb"

    build_nwb(
        acq_folder=acq_folder,
        rig_cfg=rig_cfg,
        subject_id=args.subject,
        session_str=session_str,
        acq_type=args.acq_type,
        description=args.description,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

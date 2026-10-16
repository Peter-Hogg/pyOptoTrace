"""Batch spinning-disk + motion-correction preprocessing for voltage imaging sessions.

Applies spinning-disk artifact correction and rigid motion correction to every
acquisition under a parent directory.

Caching policy (single-cache): the ONLY large file kept on scratch per movie
is the final corrected movie (spinning-disk artifact removed + rigid motion
corrected), plus tiny sidecars (motion template, per-frame shifts).

RAM discipline: raw movies are memmapped from the RAID archive and streamed;
the spinning-disk intermediate streams to a TRANSIENT memmap on scratch
(NVMe) that is deleted as soon as motion correction has consumed it. Peak
disk usage during processing is therefore ~2x one movie, steady state 1x,
and RAM stays at chunk size throughout.

Cache layout per acquisition (unchanged names, so previously registered
sessions keep working):

    <scratch>/<acq>/mov1_motion_correction_cache/
        mov1_corrected_mov.npy               <- the single big cache file
        mov1_corrected_mov.mov1_shifts.npy   <- (n_frames, 2) float32
        mov1_motion_template.npy
    <scratch>/<acq>/mov2_motion_correction_cache/   (only if mov2 is a time series)
        mov2_corrected_mov.npy, ...
    <scratch>/<acq>/tmp/                     <- ephemeral working memmaps
        (see ephemeral_dir(); safe to delete in its entirety at any time)

The old caches (preprocess_cache/mov1_preprocessed.npy, mov2_cropped.npy,
mov1_preprocessed_light.npy) are no longer produced and can be deleted once
mov1_corrected_mov.npy exists for a session.
"""

import os

# Must be set before jnormcorre imports JAX/XLA.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import gc
import glob
from pathlib import Path

import numpy as np
import torch
from jnormcorre.motion_correction import MotionCorrect
from tqdm.auto import tqdm

from . import helpers, luminos_preprocess
from . import luminos_io as lio

# ── Storage configuration ───────────────────────────────────────────────────
ARCHIVE_ROOT = Path(os.environ.get("PYOPTOTRACE_ARCHIVE", "/data")).expanduser()
SCRATCH_ROOT = Path(
    os.environ.get("PYOPTOTRACE_SCRATCH", Path.home() / ".cache" / "pyoptotrace")
).expanduser()


def to_scratch(archive_path: Path) -> Path:
    """Map an archive path to the corresponding scratch path."""
    try:
        relative = Path(archive_path).resolve().relative_to(ARCHIVE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{archive_path} is outside PYOPTOTRACE_ARCHIVE={ARCHIVE_ROOT}") from exc
    return SCRATCH_ROOT / relative


def ephemeral_dir(acq_folder) -> Path:
    """Directory on scratch (NVMe) for working memmaps: <scratch>/<acq>/tmp/.

    Use this for anything that needs disk backing but is NOT part of the
    durable cache (e.g. the regressed movie from preprocess_movie_light).
    Safe to delete in its entirety at any time.
    """
    d = to_scratch(Path(acq_folder)) / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_end_frame(mov) -> int:
    """Index of the last good frame (upper bound for correct_spinning_disk_artifact).

    Threshold at the midpoint between the good cluster (median brightness) and
    the blank floor (min); take the last frame above it. Robust to a sharp end
    cliff.
    """
    brightness = mov.mean(axis=(1, 2))
    hi = np.median(brightness)  # good cluster
    lo = brightness.min()  # blank floor (cliff is sharp)
    threshold = 0.5 * (hi + lo)
    good_frames = np.flatnonzero(brightness > threshold)
    if good_frames.size == 0:
        raise ValueError("movie contains no frames above the inferred blank-frame threshold")
    return int(good_frames[-1] + 1)


def run_rigid_motion_correction(
    mov,
    cache_dir: Path,
    prefix: str,
    *,
    max_shifts=(20, 20),
    frames_per_split=500,
    num_splits_to_process_rig=None,
    niter_rig=4,
    pw_rigid=False,
    strides=(100, 100),
    overlaps=(50, 50),
    max_deviation_rigid=8,
    gSig_filt=(7, 7),
    apply_batch_size=2500,
    force_reregister=False,
):
    """Rigid motion-correct ``mov`` and stream the result to a memmap on scratch.

    Returns (corrected_mov, shifts) where ``shifts`` is always (n_frames, 2).
    Cached template / corrected movie / shifts are reused unless force_reregister.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    template_path = cache_dir / f"{prefix}_motion_template.npy"
    corrected_path = cache_dir / f"{prefix}_corrected_mov.npy"
    # Match the notebook's filename exactly so caches are shared both ways:
    #   corrected_path.with_suffix('').with_suffix(f'.{prefix}_shifts.npy')
    #   -> "{prefix}_corrected_mov.{prefix}_shifts.npy"
    shifts_path = cache_dir / f"{prefix}_corrected_mov.{prefix}_shifts.npy"

    n_frames, H, W = mov.shape

    # ── Reuse cached corrected movie + shifts ───────────────────────────────
    if corrected_path.exists() and shifts_path.exists() and not force_reregister:
        corrected_mov = np.load(corrected_path, mmap_mode="r")
        shifts = np.load(shifts_path)
        print(f"[{prefix}] loaded cached corrected movie and shifts {shifts.shape}")
        return corrected_mov, shifts

    # ── Template ────────────────────────────────────────────────────────────
    if template_path.exists() and not force_reregister:
        print(f"[{prefix}] loading cached template...")
        template = np.load(template_path)
    else:
        print(f"[{prefix}] computing template from median of 1000-frame sample...")
        template = helpers.median_proj(mov, 1000).astype(np.float32)
        np.save(template_path, template)

    # ── Estimate shifts ─────────────────────────────────────────────────────
    print(f"[{prefix}] running motion correction...")
    corrector = MotionCorrect(
        mov,
        max_shifts=max_shifts,
        frames_per_split=frames_per_split,
        num_splits_to_process_rig=num_splits_to_process_rig,
        strides=strides,
        overlaps=overlaps,
        max_deviation_rigid=max_deviation_rigid,
        niter_rig=niter_rig,
        pw_rigid=pw_rigid,
        gSig_filt=gSig_filt,
    )
    frame_corrector, _ = corrector.motion_correct(template=template, save_movie=False)

    # ── Interpolate shifts to per-frame if needed ───────────────────────────
    shifts_raw = np.array(corrector.shifts_rig, dtype=np.float32)
    print(f"[{prefix}] shifts_rig length: {len(shifts_raw)} (movie: {n_frames} frames)")

    if len(shifts_raw) == n_frames:
        shifts = shifts_raw
        print(f"[{prefix}] shifts_rig is per-frame, no interpolation needed")
    else:
        print(f"[{prefix}] interpolating {len(shifts_raw)} shift estimates -> {n_frames} frames...")
        split_centers = np.linspace(0, n_frames - 1, len(shifts_raw))
        shifts = np.stack(
            [
                np.interp(np.arange(n_frames), split_centers, shifts_raw[:, 0]),
                np.interp(np.arange(n_frames), split_centers, shifts_raw[:, 1]),
            ],
            axis=1,
        ).astype(np.float32)

    assert shifts.shape == (n_frames, 2), (
        f"shifts shape {shifts.shape} doesn't match movie length {n_frames}"
    )

    # ── Apply shifts into a memmap ──────────────────────────────────────────
    corrected_mov = np.lib.format.open_memmap(
        corrected_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_frames, H, W),
    )
    for i in tqdm(
        range(0, n_frames, apply_batch_size), desc=f"{prefix}: applying shifts", unit="batch"
    ):
        chunk = mov[i : i + apply_batch_size]
        output = frame_corrector.register_frames(chunk, pw_rigid=pw_rigid)
        corrected_mov[i : i + apply_batch_size] = np.asarray(output, dtype=np.float32)
        del output, chunk
        gc.collect()

    corrected_mov.flush()

    # Saved LAST on purpose: the cache is only treated as valid when movie AND
    # shifts both exist, so a crash mid-apply leaves no shifts file and the
    # partial movie is recomputed instead of silently loaded.
    np.save(shifts_path, shifts)
    print(f"[{prefix}] corrected movie cached -> {corrected_path}")
    return corrected_mov, shifts


def sd_mc_single_cache(
    folder=None,
    *,
    p=None,
    mov_raw=None,
    prefix="mov1",
    period=12,
    discard_frames=2000,
    sd_chunk_size=5000,
    force=False,
    **mc_kwargs,
):
    """Load-or-compute the single cached movie: spinning-disk + motion corrected.

    Fast path: if the final corrected movie + shifts exist on scratch, they are
    returned immediately (movie as a read-only memmap) without touching raw data.

    Slow path: raw is streamed from the RAID memmap; spinning-disk correction
    streams to a transient memmap on scratch (NVMe), motion correction consumes
    it into the single cache file, and the transient is deleted. RAM stays at
    chunk size; peak scratch usage is ~2x one movie while processing.

    Parameters
    ----------
    folder : path-like, optional
        Acquisition folder on the archive. Ignored if ``p`` is given.
    p : acquisition path bundle from lio.get_acq_paths, optional
    mov_raw : array-like, optional
        Already-loaded raw movie (e.g. from a notebook; a memmap is ideal).
        If None, memmapped lazily from p.bin1 only when recomputation is needed.
    force : bool
        Recompute even if the cache exists.
    **mc_kwargs
        Passed through to run_rigid_motion_correction.

    Returns
    -------
    (corrected_mov, shifts) : read-only memmap (n_frames, H-2, W) float32,
                              and (n_frames, 2) float32 shifts.
    """
    if p is None:
        p = lio.get_acq_paths(folder)

    cache_dir = to_scratch(p.folder) / f"{prefix}_motion_correction_cache"
    corrected_path = cache_dir / f"{prefix}_corrected_mov.npy"
    shifts_path = cache_dir / f"{prefix}_corrected_mov.{prefix}_shifts.npy"

    if corrected_path.exists() and shifts_path.exists() and not force:
        corrected = np.load(corrected_path, mmap_mode="r")
        shifts = np.load(shifts_path)
        print(
            f"[{prefix}] loaded cached SD+MC movie {corrected.shape} "
            f"({corrected.nbytes / 1e9:.1f} GB), shifts {shifts.shape}"
        )
        return corrected, shifts

    if mov_raw is None:
        cam = lio.load_camera_meta(p.mat)
        mov_raw = lio.load_bin(p.bin1, cam=cam, memmap=True)  # stream from RAID

    end_frame = detect_end_frame(mov_raw)
    if end_frame <= discard_frames:
        raise ValueError(
            f"trailing-blank detection gave end_frame={end_frame}, "
            f"which leaves no frames after discard_frames={discard_frames}. "
            f"Check the brightness trace for {p.folder} (all-blank or dim FOV?)."
        )

    # Transient spinning-disk output on scratch (NVMe) — deleted after MC.
    cache_dir.mkdir(parents=True, exist_ok=True)
    sd_temp = cache_dir / f"{prefix}_sd_temp.npy"
    try:
        print(
            f"[{prefix}] spinning-disk correction {mov_raw.shape} "
            f"(end_frame={end_frame}) -> transient {sd_temp.name}..."
        )
        sd_mov, _gt = luminos_preprocess.correct_spinning_disk_artifact(
            mov_raw,
            end_frame=end_frame,
            period=period,
            discard_frames=discard_frames,
            output_path=sd_temp,
            chunk_size=sd_chunk_size,
        )
        del mov_raw

        run_rigid_motion_correction(
            sd_mov,
            cache_dir,
            prefix,
            force_reregister=force,
            **mc_kwargs,
        )
    finally:
        if "sd_mov" in locals():
            del sd_mov
        gc.collect()
        sd_temp.unlink(missing_ok=True)

    # Reopen read-only so callers never hold the writable memmap.
    corrected = np.load(corrected_path, mmap_mode="r")
    shifts = np.load(shifts_path)
    return corrected, shifts


def crop_mov2(mov2, mov2_fs=10):
    """Trim mov2 start/end and convert time series to float32 (what motion
    correction has always consumed).

    The float32 copy lands in RAM, but mov2 is the slow camera (~4 GB max),
    unlike mov1 which must stay memmap-backed. None (opto acquisition) and
    single-frame snapshots pass through untouched.
    """
    if mov2 is None or mov2.shape[0] <= 1:
        return mov2
    if mov2.shape[0] > 2 * mov2_fs + 50:
        mov2 = mov2[2 * mov2_fs : -50, :, :]
    return np.asarray(mov2, dtype=np.float32)


def batch_SD_preprocessing(
    parent_dir_path,
    force=False,
    force_repreprocess=False,
    force_reregister=False,
    exclude_patterns=(),
):
    """Preprocess + motion-correct every acquisition under ``parent_dir_path``.

    With the single-cache scheme there is one force flag; the two legacy
    kwargs are kept as aliases. Sessions whose folder name contains any of
    ``exclude_patterns`` are skipped (e.g. a different sensor's dataset
    sharing the same recording day).
    """
    force = force or force_repreprocess or force_reregister
    dirs = sorted(glob.glob(os.path.join(parent_dir_path, "*")))
    failed_dirs = []

    for folder in dirs:
        if not os.path.isdir(folder):
            continue
        if not os.path.isfile(os.path.join(folder, "output_data.mat")):
            continue
        name = os.path.basename(folder)
        if any(pat in name for pat in exclude_patterns):
            print(f"excluded by pattern: {name}")
            continue

        try:
            p = lio.get_acq_paths(folder)

            def _cached(prefix):
                d = to_scratch(p.folder) / f"{prefix}_motion_correction_cache"
                return (d / f"{prefix}_corrected_mov.npy").exists() and (
                    d / f"{prefix}_corrected_mov.{prefix}_shifts.npy"
                ).exists()

            # Skip the raw load entirely when both movies are already done.
            # (Sessions whose mov2 is a snapshot never get a mov2 cache, so
            # they reload raw here just to confirm there is nothing to do.)
            if _cached("mov1") and _cached("mov2") and not force:
                print(f"skipping (cached): {folder}")
                continue

            # memmap=True: raw streams from the RAID archive, no RAM copy
            mov1_raw, mov2_raw, cam = lio.load_acquisition(p.folder, memmap=True)

            # ── mov1: spinning-disk + motion correction, single cache ───────
            sd_mc_single_cache(
                p=p,
                mov_raw=mov1_raw,
                force=force,
                max_shifts=(20, 20),
                frames_per_split=500,
                gSig_filt=(7, 7),
                apply_batch_size=2500,
            )
            del mov1_raw
            gc.collect()

            # ── mov2: crop in RAM, motion correction only if time series ────
            # (mov2_raw is None for opto acquisitions with no frames2.bin)
            if mov2_raw is not None and mov2_raw.shape[0] > 1 and (not _cached("mov2") or force):
                mov2 = crop_mov2(mov2_raw)
                run_rigid_motion_correction(
                    mov2,
                    to_scratch(p.folder) / "mov2_motion_correction_cache",
                    prefix="mov2",
                    max_shifts=(8, 8),
                    frames_per_split=100,
                    gSig_filt=(5, 5),
                    apply_batch_size=4500,
                    force_reregister=force,
                )
                del mov2
            del mov2_raw

            # Free GPU / RAM between sessions.
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as exc:
            failed_dirs.append(folder)
            print(f"Failure for path: {folder}\n  {type(exc).__name__}: {exc}")

    if failed_dirs:
        print("\nManually review these paths:")
        for d in failed_dirs:
            print(f"  {d}")

    return failed_dirs


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_dir", help="Parent directory containing acquisition folders")
    parser.add_argument("--force", action="store_true", help="Ignore cached results and recompute")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip sessions whose folder name contains this substring (repeatable)",
    )
    # Legacy aliases — both now mean the same thing as --force.
    parser.add_argument("--force-repreprocess", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force-reregister", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    batch_SD_preprocessing(
        args.parent_dir,
        force=args.force or args.force_repreprocess or args.force_reregister,
        exclude_patterns=tuple(args.exclude),
    )


if __name__ == "__main__":
    main()

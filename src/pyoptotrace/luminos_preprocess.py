from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def correct_spinning_disk_artifact(
    movie: np.ndarray,
    period: int = 12,
    discard_frames: int = 2000,
    baseline_offset: int = 100,
    output_path: str = None,
    chunk_size: int = 5000,
    end_frame: int | None = None,  # ← NEW
) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient spinning disk artifact removal.

    Streams through the movie in chunks. If output_path is given, writes
    the corrected movie directly to a memmap on disk; otherwise allocates
    a single output array.

    Parameters
    ----------
    movie : np.ndarray, shape (n_frames, H, W)
        Input movie. May be a memmap, h5py Dataset, or in-memory array.
    period : int
        Frame period of the disk artifact (default 12).
    discard_frames : int
        Frames to drop from the start (disk/trigger sync). Default 2000.
    baseline_offset : int
        Camera dark-count pedestal to subtract. Default 100.
    output_path : str or Path, optional
        If given, stream output to a .npy memmap at this path.
        If None, allocate output in RAM.
    chunk_size : int
        Frames per processing chunk. Default 5000.
    end_frame : int, optional
        If given, treat the movie as if it ended at this frame index
        (exclusive). Lets you drop trailing frames without slicing the
        input array, which would force materialisation of an h5py
        Dataset into RAM. If None, uses ``movie.shape[0]``.

    Returns
    -------
    corrected : np.ndarray or np.memmap, shape (n_frames_kept, H-2, W), float32
    ground_truth : np.ndarray, shape (H-2, W), float32
    """
    # ── Resolve effective end of movie ────────────────────────────────────
    n_frames_orig = movie.shape[0] if end_frame is None else end_frame
    if not 0 < n_frames_orig <= movie.shape[0]:
        raise ValueError(
            f"end_frame={end_frame} out of bounds for movie with {movie.shape[0]} frames"
        )
    if n_frames_orig <= discard_frames + 1:
        raise ValueError(
            f"end_frame={n_frames_orig} leaves no frames after discard_frames={discard_frames}"
        )

    H, W = movie.shape[1], movie.shape[2]
    n_frames = n_frames_orig - discard_frames - 1  # discard start + drop final
    H_out = H - 2  # border rows dropped

    # ── Pass 1: compute mean image and per-phase means in chunks ──────────
    sum_total = np.zeros((H_out, W), dtype=np.float64)
    count_total = 0
    sum_per_phase = np.zeros((period, H_out, W), dtype=np.float64)
    count_per_phase = np.zeros(period, dtype=np.int64)

    for start in range(discard_frames, n_frames_orig - 1, chunk_size):
        end = min(start + chunk_size, n_frames_orig - 1)
        chunk = movie[start:end, 1:-1, :].astype(np.float32) - baseline_offset

        sum_total += chunk.sum(axis=0)
        count_total += chunk.shape[0]

        for i, frame_idx in enumerate(range(start, end)):
            phase = frame_idx % period
            sum_per_phase[phase] += chunk[i]
            count_per_phase[phase] += 1

    ground_truth = (sum_total / count_total).astype(np.float32)

    # Per-phase mean, then ratio to global mean
    epsilon = 1e-9
    # In correct_spinning_disk_artifact, replace the artifact_pattern computation:
    gt_floor = max(float(np.percentile(ground_truth, 5)), 20.0)
    gt_safe = np.maximum(ground_truth, gt_floor)

    artifact_pattern = np.empty((period, H_out, W), dtype=np.float32)
    for phase in range(period):
        phase_mean = sum_per_phase[phase] / count_per_phase[phase]
        pattern = phase_mean / (gt_safe + epsilon)
        artifact_pattern[phase] = np.clip(pattern, 0.5, 2.0).astype(np.float32)

    del sum_total, sum_per_phase

    # ── Allocate output (memmap if path given, else in-RAM) ───────────────
    if output_path is not None:
        corrected = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_frames, H_out, W),
        )
    else:
        corrected = np.empty((n_frames, H_out, W), dtype=np.float32)

    # ── Pass 2: divide out artifact, write to output in chunks ────────────
    for start in range(discard_frames, n_frames_orig - 1, chunk_size):
        end = min(start + chunk_size, n_frames_orig - 1)
        chunk = movie[start:end, 1:-1, :].astype(np.float32) - baseline_offset

        for i, frame_idx in enumerate(range(start, end)):
            phase = frame_idx % period
            chunk[i] /= artifact_pattern[phase]

        np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        out_start = start - discard_frames
        out_end = end - discard_frames
        corrected[out_start:out_end] = chunk
        del chunk

    if output_path is not None:
        corrected.flush()

    return corrected, ground_truth


def correct_spinning_disk_artifact_gpu_streaming(
    movie: np.ndarray,  # memmap or h5py.Dataset
    output_path: str | None,  # required for >RAM movies
    period: int = 12,
    discard_frames: int = 2000,
    baseline_offset: int = 100,
    chunk_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated streaming spinning disk correction.

    Reads input chunks from disk-backed array, processes on GPU,
    writes output chunks to memmap.
    """
    output_path = Path(output_path)
    n_frames_orig, H_in, W = movie.shape
    H_out = H_in - 2
    n_frames_out = (n_frames_orig - 1) - discard_frames

    # ── Pass 1: per-phase sums on GPU ─────────────────────────────────────
    # Keep accumulators on GPU as float32 (memory-efficient) and scale up to
    # float64 only for the final mean. For movies <1M frames, float32
    # accumulation is precise enough.
    sum_per_phase = jnp.zeros((period, H_out, W), dtype=jnp.float32)
    count_per_phase = jnp.zeros(period, dtype=jnp.int32)

    pass_start, pass_end = discard_frames, n_frames_orig - 1
    n_chunks = (pass_end - pass_start + chunk_size - 1) // chunk_size

    @jax.jit
    def _accumulate(sum_per_phase, count_per_phase, chunk, phases):
        # phases: (B,) int, chunk: (B, H, W) float32
        for p in range(period):
            mask = phases == p
            sum_per_phase = sum_per_phase.at[p].add(
                jnp.where(mask[:, None, None], chunk, 0).sum(axis=0)
            )
            count_per_phase = count_per_phase.at[p].add(mask.sum())
        return sum_per_phase, count_per_phase

    for start in tqdm(range(pass_start, pass_end, chunk_size), total=n_chunks, desc="Pass 1 (GPU)"):
        end = min(start + chunk_size, pass_end)
        chunk_np = movie[start:end, 1:-1, :].astype(np.float32) - baseline_offset
        chunk_jax = jnp.asarray(chunk_np)
        phases = jnp.asarray(np.arange(start, end) % period, dtype=jnp.int32)
        sum_per_phase, count_per_phase = _accumulate(
            sum_per_phase, count_per_phase, chunk_jax, phases
        )

    # Compute artifact pattern on GPU
    phase_means = sum_per_phase / count_per_phase[:, None, None]
    ground_truth = phase_means.mean(axis=0)
    epsilon = 1e-9
    artifact_pattern = phase_means / (ground_truth + epsilon)

    # Move ground truth back to CPU for return
    ground_truth_np = jax.device_get(ground_truth)

    # Free Pass 1 accumulators on GPU
    del sum_per_phase, count_per_phase, phase_means

    # ── Allocate output memmap ────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(n_frames_out, H_out, W)
    )

    # ── Pass 2: GPU correction, write to disk ─────────────────────────────
    @jax.jit
    def _correct(chunk, pattern_per_phase, phases):
        # Gather the right pattern for each frame
        pattern = pattern_per_phase[phases]  # (B, H, W)
        out = chunk / pattern
        # Replace nans/infs (in case ground_truth had a near-zero pixel)
        out = jnp.where(jnp.isfinite(out), out, 0.0)
        return out

    for start in tqdm(range(pass_start, pass_end, chunk_size), total=n_chunks, desc="Pass 2 (GPU)"):
        end = min(start + chunk_size, pass_end)
        chunk_np = movie[start:end, 1:-1, :].astype(np.float32) - baseline_offset
        chunk_jax = jnp.asarray(chunk_np)
        phases = jnp.asarray(np.arange(start, end) % period, dtype=jnp.int32)

        corrected_jax = _correct(chunk_jax, artifact_pattern, phases)
        corrected_np = jax.device_get(corrected_jax)

        out_start = start - discard_frames
        out_end = end - discard_frames
        corrected[out_start:out_end] = corrected_np

    corrected.flush()
    return corrected, ground_truth_np


def detect_hot_pixels(
    movie: np.ndarray,
    percentile: float = 99.9,
    sample_frames: int = 2000,
    isolation_ratio: float = 3.0,
    min_absolute: float | None = None,
) -> np.ndarray:
    """
    Detect hot pixels by spatial isolation on a high-percentile projection.

    A hot pixel is defined by being bright *and* spatially isolated: its
    value is much larger than the maximum of its 8 neighbours. Biological
    features — even single spines or fine dendrites — are spread over
    multiple pixels by the PSF, so they cannot satisfy this criterion.

    A high percentile is used instead of the max projection because a
    single cosmic ray or shot-noise transient can otherwise produce a
    spurious bright pixel. Percentile ~99.9 is robust to ~0.1% outlier
    frames per pixel while still capturing persistent hot behaviour.

    Appropriate for negative-going indicators like Voltron2: spikes
    reduce fluorescence, so they never inflate the upper-tail percentile
    of a pixel's intensity distribution and cannot be mistaken for hot
    pixels.

    Parameters
    ----------
    movie : np.ndarray, shape (n_frames, n_rows, n_cols)
    percentile : float
        Per-pixel percentile used as the "bright projection". Default 99.9.
        Use 100 (equivalent to max) only if single-frame transients are
        not a concern.
    sample_frames : int
        Frames evenly sampled across the movie. 2000 gives a stable 99.9th
        percentile estimate.
    isolation_ratio : float
        A pixel is flagged if its projection value exceeds
        ``isolation_ratio`` times the maximum projection value among its
        8 immediate neighbours. Default 3.0 is conservative.
    min_absolute : float or None
        Optional absolute floor: pixels below this projection value are
        never flagged. If None, set to the 50th percentile of the
        projection image (ensures ratios in dim background regions
        don't generate spurious flags).

    Returns
    -------
    hot_mask : np.ndarray of bool, shape (n_rows, n_cols)
    """
    n_frames = movie.shape[0]
    if n_frames > sample_frames:
        idx = np.linspace(0, n_frames - 1, sample_frames).astype(np.int64)
        sample = np.asarray(movie[idx], dtype=np.float32)
    else:
        sample = np.asarray(movie, dtype=np.float32)

    # High-percentile projection (robust alternative to max).
    proj = np.percentile(sample, percentile, axis=0).astype(np.float32)  # (H, W)

    # Max of the 8 neighbours, excluding the center pixel. We cannot use
    # scipy's maximum_filter directly (it includes the center), so we
    # shift-and-reduce over the 8 offsets.
    neighbour_max = _max_excluding_center(proj)

    if min_absolute is None:
        min_absolute = float(np.percentile(proj, 50))

    eps = 1e-6
    ratio = proj / (neighbour_max + eps)
    hot_mask = (ratio > isolation_ratio) & (proj > min_absolute)

    return hot_mask


def _max_excluding_center(img: np.ndarray) -> np.ndarray:
    """Max over the 3x3 neighbourhood, excluding the center pixel."""
    H, W = img.shape
    padded = np.pad(img, 1, mode="reflect")
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    out = np.full_like(img, -np.inf)
    for di, dj in shifts:
        shifted = padded[1 + di : 1 + di + H, 1 + dj : 1 + dj + W]
        np.maximum(out, shifted, out=out)
    return out


def correct_hot_pixels(
    movie: np.ndarray,
    hot_mask: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Replace hot pixels with the median of their 8 neighbours, per frame, on GPU.

    If a hot pixel's neighbour is itself flagged (hot pixels clustering),
    that neighbour's coordinates are replaced with the first non-hot
    neighbour's coordinates, so the median is computed over genuine
    tissue values only.

    Parameters
    ----------
    movie : np.ndarray, shape (n_frames, n_rows, n_cols)
    hot_mask : np.ndarray of bool, shape (n_rows, n_cols)
    batch_size : int
        Frames per GPU batch (default 500).

    Returns
    -------
    corrected : np.ndarray, shape (n_frames, n_rows, n_cols)
    """
    if not hot_mask.any():
        return movie

    n_frames, H, W = movie.shape
    dtype_out = movie.dtype if np.issubdtype(movie.dtype, np.floating) else np.float32

    hot_coords = np.argwhere(hot_mask)  # (n_hot, 2)
    offsets = np.array(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        dtype=np.int64,
    )
    neigh = hot_coords[:, None, :] + offsets[None, :, :]  # (n_hot, 8, 2)
    neigh[..., 0] = np.clip(neigh[..., 0], 0, H - 1)
    neigh[..., 1] = np.clip(neigh[..., 1], 0, W - 1)

    # Exclude neighbours that are themselves hot by substituting the first
    # non-hot neighbour's coords into their slot. Rare unless hot pixels
    # cluster, but matters for correctness when they do.
    neigh_is_hot = hot_mask[neigh[..., 0], neigh[..., 1]]  # (n_hot, 8)
    if neigh_is_hot.any():
        for k in np.where(neigh_is_hot.any(axis=1))[0]:
            good = ~neigh_is_hot[k]
            if good.any():
                good_r = neigh[k, good, 0][0]
                good_c = neigh[k, good, 1][0]
                neigh[k, ~good, 0] = good_r
                neigh[k, ~good, 1] = good_c

    j_rows_hot = jnp.asarray(hot_coords[:, 0])
    j_cols_hot = jnp.asarray(hot_coords[:, 1])
    j_rows_n = jnp.asarray(neigh[..., 0])
    j_cols_n = jnp.asarray(neigh[..., 1])

    @jax.jit
    def _replace_batch(batch, rows_hot, cols_hot, rows_n, cols_n):
        neighbour_vals = batch[:, rows_n, cols_n]  # (B, n_hot, 8)
        replacement = jnp.median(neighbour_vals, axis=-1)  # (B, n_hot)
        return batch.at[:, rows_hot, cols_hot].set(replacement)

    out = np.empty((n_frames, H, W), dtype=dtype_out)
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        batch = jnp.asarray(movie[start:end], dtype=jnp.float32)
        fixed = _replace_batch(batch, j_rows_hot, j_cols_hot, j_rows_n, j_cols_n)
        out[start:end] = np.asarray(jax.device_get(fixed), dtype=dtype_out)

    return out


def remove_hot_pixels(
    movie: np.ndarray,
    percentile: float = 99.9,
    sample_frames: int = 2000,
    isolation_ratio: float = 3.0,
    min_absolute: float | None = None,
    batch_size: int = 500,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: detect and correct hot pixels in one call.

    Intended as the first step of voltage imaging preprocessing, before
    spinning disk correction, residual projection, or motion correction.

    Returns
    -------
    corrected : np.ndarray
    hot_mask : np.ndarray of bool, optional
    """
    hot_mask = detect_hot_pixels(
        movie,
        percentile=percentile,
        sample_frames=sample_frames,
        isolation_ratio=isolation_ratio,
        min_absolute=min_absolute,
    )
    corrected = correct_hot_pixels(movie, hot_mask, batch_size=batch_size)
    if return_mask:
        return corrected, hot_mask
    return corrected

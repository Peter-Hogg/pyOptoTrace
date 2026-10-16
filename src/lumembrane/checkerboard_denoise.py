"""
Checkerboard weighted correlation denoising for Voltron2 voltage imaging.

Memory strategy:
  - Movie stays in numpy (CPU) at all times
  - GPU processes one temporal window at a time (default 1000 frames)
  - Peak VRAM per window: 2 × window × H × W × float32
    e.g. window=1000, 800×200 → ~1.2 GB — well within 32 GB
  - Correlation sums (H, W) are accumulated on CPU between windows
  - Blend pass is also windowed — denoised movie written back to a
    pre-allocated numpy array, never fully resident on GPU

Usage:
    denoised, corr_map = checkerboard_denoise(movie)

    # Tune the window if needed:
    denoised, corr_map = checkerboard_denoise(movie, window=2000)
"""

import jax
import jax.numpy as jnp
import numpy as np

from .helpers import mean_proj, median_proj

_OFFSETS = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]


def checkerboard_denoise(
    movie: np.ndarray,
    corr_threshold: float = 0.2,
    denoise_strength: float = 1.0,
    intensity_exponent: float = 1.0,
    window: int = 1000,
    median_n_sample: int = 500,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Denoise a voltage imaging movie using checkerboard weighted spatial correlation.

    Parameters
    ----------
    movie : np.ndarray
        Shape (T, H, W), float32.
    corr_threshold : float
        Weighted correlation below this → pixel treated as noise-dominated.
        Start with 0.2; raise to suppress more, lower to suppress less.
    denoise_strength : float
        Blend weight toward neighbor average for noise pixels.
        0.0 = no change, 1.0 = full replacement.
    intensity_exponent : float
        Sharpness of background suppression via intensity weighting.
        1.0 = linear, 2.0 = quadratic.
    window : int
        Number of frames per GPU batch. Reduce if still OOMing.
        1000 frames at 800×200 ≈ 1.2 GB VRAM — safe default.
    median_n_sample : int
        Frames subsampled for median frame estimate (passed to median_proj).
    eps : float
        Numerical stability constant.

    Returns
    -------
    denoised : np.ndarray, shape (T, H, W), float32
    corr_map : np.ndarray, shape (H, W), float32
        Per-pixel weighted Pearson correlation with diagonal neighbors.
        High = membrane signal, low = background / noise.
    """
    movie = np.asarray(movie, dtype=np.float32)
    T, H, W = movie.shape

    # ------------------------------------------------------------------
    # 1. Intensity weights — median frame via subsampled GPU percentile
    # ------------------------------------------------------------------
    median_frame = median_proj(movie, n_sample=median_n_sample)  # (H, W)
    mn, mx = median_frame.min(), median_frame.max()
    median_norm = (median_frame - mn) / (mx - mn + eps)
    intensity_weight = (median_norm**intensity_exponent).astype(np.float32)

    # Precompute per-offset intensity weights (H, W) — tiny, stay on CPU
    offset_weights = {}
    for di, dj in _OFFSETS:
        nb_iw = _np_shift(intensity_weight, di, dj)
        offset_weights[(di, dj)] = (intensity_weight * nb_iw).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Z-score normalisation — chunked GPU mean, CPU std
    # ------------------------------------------------------------------
    mu = mean_proj(movie)[None]  # (1, H, W)
    std = movie.std(axis=0, keepdims=True) + eps  # (H, W), CPU
    movie_norm = ((movie - mu) / std).astype(np.float32)  # (T, H, W), CPU

    # ------------------------------------------------------------------
    # 3. Correlation pass — windowed GPU accumulation
    #    Peak VRAM: 2 × window × H × W float32 (center + one neighbor)
    # ------------------------------------------------------------------
    corr_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    n_windows = (T + window - 1) // window
    for w_idx in range(n_windows):
        s, e = w_idx * window, min((w_idx + 1) * window, T)
        chunk_norm = jnp.array(movie_norm[s:e])  # (win, H, W) → GPU
        chunk_len = e - s

        for di, dj in _OFFSETS:
            nb_chunk = _jax_shift(chunk_norm, di, dj)  # (win, H, W) GPU
            # Dot product over time dimension → (H, W), pull to CPU immediately
            dot = np.array(jnp.sum(chunk_norm * nb_chunk, axis=0))
            del nb_chunk

            w = offset_weights[(di, dj)]
            corr_sum += w * dot  # accumulate unnormalised dot
            weight_sum += w * chunk_len  # weight by window length for normalisation

        del chunk_norm
        jax.clear_caches()

    # Normalise: divide by total frames (via weight_sum) to get mean correlation
    corr_map = (corr_sum / (weight_sum + eps)).astype(np.float32)  # (H, W)

    # ------------------------------------------------------------------
    # 4. Blend pass — windowed GPU, writes into pre-allocated output
    #    Uses raw movie chunks, not movie_norm
    # ------------------------------------------------------------------
    noise_mask = np.clip((corr_threshold - corr_map) / (corr_threshold + eps), 0.0, 1.0).astype(
        np.float32
    )  # (H, W)
    blend = (denoise_strength * noise_mask).astype(np.float32)  # (H, W)

    # Pre-allocate output — avoids holding two full movies in RAM simultaneously
    denoised = np.empty_like(movie)

    blend_gpu = jnp.array(blend)  # (H, W) — tiny, fine to keep on GPU
    w_sum_gpu = sum(jnp.array(offset_weights[(di, dj)]) for di, dj in _OFFSETS)

    for w_idx in range(n_windows):
        s, e = w_idx * window, min((w_idx + 1) * window, T)
        chunk = jnp.array(movie[s:e])  # (win, H, W) → GPU

        nb_sum = jnp.zeros_like(chunk)
        for di, dj in _OFFSETS:
            nb = _jax_shift(chunk, di, dj)
            w = jnp.array(offset_weights[(di, dj)])
            nb_sum = nb_sum + w[None] * nb
            del nb

        neighbor_avg = nb_sum / (w_sum_gpu[None] + eps)
        denoised_chunk = (1.0 - blend_gpu[None]) * chunk + blend_gpu[None] * neighbor_avg
        denoised[s:e] = np.array(denoised_chunk)  # pull to CPU, write slice

        del chunk, nb_sum, neighbor_avg, denoised_chunk
        jax.clear_caches()

    return denoised, corr_map


# ------------------------------------------------------------------
# Shift helpers
# ------------------------------------------------------------------


def _np_shift(arr: np.ndarray, di: int, dj: int) -> np.ndarray:
    """Shift numpy array on last two axes with edge replication. Returns a copy."""
    arr = np.roll(arr, di, axis=-2)
    arr = np.roll(arr, dj, axis=-1)
    if di > 0:
        arr[..., :di, :] = arr[..., di : di + 1, :]
    elif di < 0:
        arr[..., di:, :] = arr[..., di - 1 : di, :]
    if dj > 0:
        arr[..., :, :dj] = arr[..., :, dj : dj + 1]
    elif dj < 0:
        arr[..., :, dj:] = arr[..., :, dj - 1 : dj]
    return arr


def _jax_shift(arr: jnp.ndarray, di: int, dj: int) -> jnp.ndarray:
    """Shift JAX array on last two axes with edge replication. di/dj are static ints."""
    arr = jnp.roll(arr, di, axis=-2)
    arr = jnp.roll(arr, dj, axis=-1)
    if di > 0:
        arr = arr.at[..., :di, :].set(arr[..., di : di + 1, :])
    elif di < 0:
        arr = arr.at[..., di:, :].set(arr[..., di - 1 : di, :])
    if dj > 0:
        arr = arr.at[..., :, :dj].set(arr[..., :, dj : dj + 1])
    elif dj < 0:
        arr = arr.at[..., :, dj:].set(arr[..., :, dj - 1 : dj])
    return arr


# ------------------------------------------------------------------
# QC helpers
# ------------------------------------------------------------------


def plot_corr_map(corr_map, median_frame=None, threshold=0.2, figsize=(12, 4)):
    """
    QC plot: correlation map + noise mask + optional median frame.
    Membrane pixels should be bright; background near zero.
    """
    import matplotlib.pyplot as plt

    ncols = 3 if median_frame is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    im = axes[0].imshow(corr_map, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Weighted correlation map")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].imshow((corr_map < threshold).astype(float), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Noise mask (corr < {threshold})")

    if median_frame is not None:
        axes[2].imshow(median_frame, cmap="gray")
        axes[2].set_title("Median frame")

    plt.tight_layout()
    return fig


def compute_snr(movie, corr_map, threshold=0.2):
    """Variance ratio between membrane and background pixels as a rough SNR check."""
    mem = corr_map > threshold
    bg = ~mem
    var_m = np.var(movie[:, mem], axis=0).mean()
    var_bg = np.var(movie[:, bg], axis=0).mean()
    print(f"Membrane pixels          : {mem.sum()}")
    print(f"Background pixels        : {bg.sum()}")
    print(f"Mean baseline (membrane) : {median_proj(movie)[mem].mean():.1f}")
    print(f"Variance (membrane)      : {var_m:.4f}")
    print(f"Variance (background)    : {var_bg:.4f}")
    print(f"Variance ratio mem/bg    : {var_m / (var_bg + 1e-6):.2f}x")


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import time

    T, H, W = 2000, 800, 200
    print(f"Synthetic movie: {T}×{H}×{W}  ({T * H * W * 4 / 1e9:.2f} GB float32)")

    rng = np.random.default_rng(42)
    mask = np.zeros((H, W), dtype=bool)
    mask[50:150, :] = True

    baseline = np.where(mask, 1000.0, 50.0).astype(np.float32)
    noise = (rng.standard_normal((T, H, W)) * 10).astype(np.float32)
    sig = np.zeros((T, H, W), dtype=np.float32)
    sig[:, mask] = (np.sin(np.linspace(0, 10 * np.pi, T)) * 50)[:, None]
    movie = baseline[None] + noise + sig

    t0 = time.time()
    denoised, corr_map = checkerboard_denoise(movie, window=1000)
    print(f"Done in {time.time() - t0:.1f}s")
    compute_snr(movie, corr_map)
    print(f"\nCorr map membrane  : {corr_map[mask].mean():.3f}")
    print(f"Corr map background: {corr_map[~mask].mean():.3f}")

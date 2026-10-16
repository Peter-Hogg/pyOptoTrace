from pathlib import Path

import cupy as cp
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def _mad_zscore(trace: np.ndarray, window: int = 3000) -> np.ndarray:
    """
    MAD z-score matching MATLAB voltage_pipeline.m:
        z = (x - movmedian(x, window)) / (MAD / 0.6745)
    Spike-robust because median and MAD are insensitive to brief transients.
    """
    baseline = median_filter(trace, size=(window,))
    residual = trace - baseline
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = mad / 0.6745 + 1e-10
    return residual / sigma


def extract_roi_trace(
    mov: np.ndarray,
    roi_mask: np.ndarray,
    hz: float = 1000.0,
    pc_index: int = 0,
    n_pcs: int = 6,
    highpass: bool = True,  # subtract 50-frame moving median before z-score
    med_filter_win=3000,
) -> dict:
    """
    Extracts voltage trace from a single ROI.
    Expects mov already cleaned by preprocess_movie() — no mc_trace needed here.
    """
    T, H, W = mov.shape
    n_pcs = min(n_pcs, int(roi_mask.sum()))

    if roi_mask.sum() < 2:
        raise ValueError(f"ROI has only {roi_mask.sum()} pixels.")

    # ── 1. Extract ROI pixels → GPU ───────────────────────────────────────────
    roi_pixels = mov[:, roi_mask].astype(np.float32)  # (T, n_px)
    roi_gpu = cp.asarray(roi_pixels, dtype=cp.float64)

    # # ── 2. PCA on GPU ─────────────────────────────────────────────────────────
    # mean_px = roi_gpu.mean(axis=0)
    # dmov    = roi_gpu - mean_px
    # cov     = dmov.T @ dmov

    # eigvals_gpu, V_gpu = cp.linalg.eigh(cov)
    # eigvals_gpu = eigvals_gpu[::-1];  V_gpu = V_gpu[:, ::-1]

    # abs_V   = cp.abs(V_gpu)
    # max_idx = cp.argmax(abs_V, axis=0)
    # signs   = cp.sign(V_gpu[max_idx, cp.arange(V_gpu.shape[1])])
    # V_gpu   = V_gpu * signs[None, :]

    # all_pc_traces_gpu = (dmov @ V_gpu[:, :n_pcs]).T

    # eigvals       = cp.asnumpy(eigvals_gpu).copy()
    # V             = cp.asnumpy(V_gpu).copy()
    # all_pc_traces = cp.asnumpy(all_pc_traces_gpu)

    # del cov, eigvals_gpu, V_gpu, abs_V, all_pc_traces_gpu, dmov, mean_px, roi_gpu
    # cp.get_default_memory_pool().free_all_blocks()

    # total_var = eigvals.sum()
    # var_exp   = eigvals[:n_pcs] / total_var

    # all_pc_maps = np.zeros((n_pcs, H, W), dtype=np.float32)
    # for i in range(n_pcs):
    #     all_pc_maps[i][roi_mask] = V[:, i].astype(np.float32)
    # ── 2. PCA on GPU (float64 for precision over long T) ─────────────────────
    mean_px = roi_gpu.mean(axis=0)
    dmov = roi_gpu - mean_px
    cov = dmov.T @ dmov

    eigvals_gpu, V_gpu = cp.linalg.eigh(cov)
    eigvals_gpu = eigvals_gpu[::-1]
    V_gpu = V_gpu[:, ::-1]

    # Sign-flip so each PC's largest-magnitude pixel is positive
    abs_V = cp.abs(V_gpu)
    max_idx = cp.argmax(abs_V, axis=0)
    signs = cp.sign(V_gpu[max_idx, cp.arange(V_gpu.shape[1])])
    V_gpu = V_gpu * signs[None, :]

    all_pc_traces_gpu = (dmov @ V_gpu[:, :n_pcs]).T

    # ── 3. Eigenvalue-weighted footprint ──────────────────────────────────────
    # Cast back to float32 for downstream — precision was needed for eigh,
    # not for storage
    # Move to CPU before freeing GPU memory
    eigvals = cp.asnumpy(eigvals_gpu).astype(np.float64).copy()
    V = cp.asnumpy(V_gpu).astype(np.float32).copy()
    all_pc_traces = cp.asnumpy(all_pc_traces_gpu).astype(np.float32)

    del cov, eigvals_gpu, V_gpu, abs_V, all_pc_traces_gpu, dmov, mean_px, roi_gpu
    cp.get_default_memory_pool().free_all_blocks()

    total_var = eigvals.sum()
    var_exp = eigvals[:n_pcs] / total_var

    all_pc_maps = np.zeros((n_pcs, H, W), dtype=np.float32)
    for i in range(n_pcs):
        all_pc_maps[i][roi_mask] = V[:, i].astype(np.float32)
    coeff = np.mean(np.abs(V[:, :n_pcs]) * eigvals[:n_pcs], axis=1)
    coeff = (coeff - coeff.min()) / (coeff.max() - coeff.min() + 1e-10)

    pc_map = np.zeros((H, W), dtype=np.float32)
    pc_map[roi_mask] = coeff.astype(np.float32)

    # ── 4. Footprint-weighted trace ────────────────────────────────────────────
    fp_norm = coeff / (coeff.sum() + 1e-10)
    fp_trace = roi_pixels @ fp_norm  # (T,)
    pc_trace = all_pc_traces[pc_index]

    # ── 5. High-pass + MAD z-score ────────────────────────────────────────────
    if highpass:
        trace_hp = fp_trace - median_filter(fp_trace, size=(50,))
        pc_hp = pc_trace - median_filter(pc_trace, size=(50,))
    else:
        trace_hp = fp_trace.copy()
        pc_hp = pc_trace.copy()

    trace_z = _mad_zscore(trace_hp, window=med_filter_win)
    pc_z = _mad_zscore(pc_hp, window=med_filter_win)

    return {
        "trace": fp_trace,
        "trace_hp": trace_hp,
        "trace_z": trace_z,
        "pc_trace": pc_trace,
        "pc_z": pc_z,
        "pc_index": pc_index,
        "pc_map": pc_map,
        "all_pc_maps": all_pc_maps,
        "all_pc_traces": all_pc_traces,
        "eigvecs": V,
        "eigvals": eigvals,
        "var_exp": var_exp,
    }


def inspect_pcs(result: dict, mov_mean: np.ndarray, cid=None, n_frames: int = 5000):
    """
    Plot PC spatial maps and time courses for manual PC selection.
    Primary trace (footprint-weighted) shown in green; selected PC in blue.
    """
    import matplotlib.pyplot as plt

    n_pcs = result["all_pc_maps"].shape[0]
    var_exp = result["var_exp"]
    selected = result["pc_index"]

    fig, axes = plt.subplots(2, n_pcs, figsize=(3 * n_pcs, 6))

    for i in range(n_pcs):
        ax = axes[0, i]
        ax.imshow(mov_mean, cmap="gray_r", interpolation="nearest")  # was 'grays_r'
        m = result["all_pc_maps"][i]
        ax.imshow(np.ma.masked_where(m == 0, m), cmap="RdBu_r", alpha=0.85, interpolation="nearest")
        title = f"PC{i + 1}  {var_exp[i] * 100:.1f}%"

        ax.set_title(
            title,
            fontsize=9,
            color="black" if i == selected else "black",  # was 'RdBu'
            fontweight="bold" if i == selected else "normal",
        )
        ax.axis("off")

        ax = axes[1, i]
        tc = result["all_pc_traces"][i][:n_frames] * -1
        ax.plot(tc, lw=0.4, color="blue" if i == selected else "gray")
        ax.set_title(f"PC{i + 1} trace", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("a.u.", fontsize=8)

    # Also show footprint trace vs selected PC z-score
    fig2, ax2 = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    ax2[0].plot(
        result["trace_z"][:n_frames] * -1,
        lw=0.4,
        color="green",
        label="footprint-weighted MAD z-score",
    )
    ax2[0].axhline(3, color="r", lw=0.6, ls="--")
    ax2[0].legend(fontsize=8)
    ax2[0].spines[["top", "right"]].set_visible(False)
    ax2[1].plot(
        result["pc_z"][:n_frames] * -1, lw=0.4, color="blue", label=f"PC{selected + 1} MAD z-score"
    )
    ax2[1].axhline(3, color="r", lw=0.6, ls="--")
    ax2[1].legend(fontsize=8)
    ax2[1].spines[["top", "right"]].set_visible(False)
    fig2.suptitle(f"ROI {cid} — footprint vs PC{selected + 1} trace", fontweight="bold")
    plt.tight_layout()

    plt.figure(fig.number)
    plt.suptitle(f"ROI {cid} — PC spatial maps", fontweight="bold")
    plt.tight_layout()
    plt.show()


def extract_all_rois(
    mov: np.ndarray,
    labels: np.ndarray,
    hz: float = 1000.0,
    pc_indices: dict = None,
    n_pcs: int = 6,
    highpass: bool = True,
    med_filter_win=3000,
) -> dict:
    cell_ids = np.unique(labels)[1:]
    results = {}

    for cid in cell_ids:
        mask = labels == cid
        pc_index = (pc_indices or {}).get(int(cid), 0)
        try:
            result = extract_roi_trace(
                mov,
                mask,
                hz=hz,
                pc_index=pc_index,
                n_pcs=n_pcs,
                highpass=highpass,
            )
            results[cid] = result
            var_str = "  ".join(
                f"PC{i + 1}={result['var_exp'][i] * 100:.1f}%"
                for i in range(len(result["var_exp"]))
            )
            print(f"ROI {cid:3d}: {mask.sum():4d} px  |  {var_str}")
        except Exception as e:
            print(f"ROI {cid:3d}: FAILED — {e}")

    return results


def denoise_voltage_movie(
    mov: np.ndarray,
    mc_trace,
    stim_vector: np.ndarray,
    chunk_size: int = 8000,
    bleach_window: int = 3000,
) -> np.ndarray:
    """Pixel-level regression denoising. Kept for reference."""
    T, H, W = mov.shape
    mov = np.ascontiguousarray(mov, dtype=np.float32)
    mc_trace = np.array(mc_trace, dtype=np.float32)
    if mc_trace.shape != (T, 2):
        raise ValueError(f"mc_trace must be shape (T, 2), got {mc_trace.shape}")

    global_F = mov.mean(axis=(1, 2))
    bkg_trace = global_F.copy()
    stim_off = stim_vector == 0

    if stim_off.any() and (~stim_off).any():
        t = np.arange(T, dtype=np.float32)
        interp_fn = interp1d(
            t[stim_off], global_F[stim_off], kind="linear", fill_value="extrapolate"
        )
        bkg_trace[~stim_off] = interp_fn(t[~stim_off])

    bkg_trace = median_filter(bkg_trace, size=(bleach_window,)).astype(np.float32)
    x = mc_trace[:, 0]
    y = mc_trace[:, 1]
    X = np.column_stack([x, y, x**2, y**2, x * y, bkg_trace, np.ones(T, dtype=np.float32)])

    X_gpu = cp.asarray(X)
    lam = 1e-6
    pinvX = cp.linalg.solve(
        X_gpu.T @ X_gpu + lam * cp.eye(X_gpu.shape[1], dtype=cp.float32), X_gpu.T
    )

    nPix = H * W
    mov_2d = mov.transpose(1, 2, 0).reshape(nPix, T)
    out_2d = np.empty_like(mov_2d)

    print(f"Denoising {nPix} pixels in chunks of {chunk_size}...")
    for start in range(0, nPix, chunk_size):
        end = min(start + chunk_size, nPix)
        Y = cp.asarray(mov_2d[start:end].T)
        betas = pinvX @ Y
        Y -= X_gpu @ betas
        out_2d[start:end] = cp.asnumpy(Y.T)
        del Y, betas
        cp.get_default_memory_pool().free_all_blocks()

    print("Denoising complete.")
    return out_2d.reshape(H, W, T).transpose(2, 0, 1)


def pcafilt(movie, npcs):
    """Time-domain PCA denoising. (T, H, W) in/out."""
    nframes, ysize, xsize = movie.shape
    if npcs > nframes:
        raise ValueError(f"npcs ({npcs}) must be <= nframes ({nframes})")

    avgimg = np.mean(movie, axis=0)
    mov2d = movie.reshape((nframes, ysize * xsize))
    dmov2d = mov2d - avgimg.reshape((1, ysize * xsize))
    covmat = dmov2d @ dmov2d.T
    eigvals, V = np.linalg.eigh(covmat)
    eigvals = eigvals[::-1].copy()
    V = V[:, ::-1].copy()

    for j in range(nframes):
        idx = np.argmax(np.abs(V[:, j]))
        V[:, j] *= np.sign(V[idx, j])

    coeffs = dmov2d.T @ V[:, :npcs]
    mov2dfilt = (V[:, :npcs] @ coeffs.T).reshape((nframes, ysize, xsize))
    mov2dfilt += np.expand_dims(avgimg, axis=0)

    return mov2dfilt, V, eigvals


def see_residuals(
    mov: np.ndarray,  # (T, H, W) — memmap or in-RAM array
    ref_sigs: np.ndarray,  # (N, T) reference traces
    rem_offset: bool = True,
    chunk_size_T: int = 10000,  # frames per time-chunk
    output_path: str | Path | None = None,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regress reference signals out of every pixel of a movie.

    Python port of the Cohen lab ``SeeResiduals.m`` (AEC 2015, FPB 2022)
    using a time-chunked streaming algorithm. Suitable for movies that
    are too large to fit in RAM as a (nPix, T) matrix.

    For each pixel's time trace y (length T), solves the least-squares
    problem:  find coefficients c such that y ≈ c @ I, where I holds
    the reference traces as rows. Then returns y - c @ I (the residual)
    for every pixel.

    Closed-form solution per pixel:
        c    = y @ I.T @ inv(I @ I.T)
        yres = y - c @ I

    Algorithm
    ---------
    Naive implementations build the full (nPix, T) matrix and iterate
    over pixel-chunks. For an (T=600k, H=200, W=200) movie that's
    ~96 GB of float32 — too big to allocate in RAM, and reshaping a
    memmap-backed (T, H, W) array to (nPix, T) produces a non-contiguous
    layout that triggers a full-size copy on first use.

    This implementation uses **two passes through the movie in time-chunks**:

      Pass 1: accumulate sufficient statistics M = I @ Y, where Y is the
              (T, nPix) view of the movie. Reading time-chunks `mov[t0:t1]`
              is a contiguous read on a frame-major memmap, which is fast
              on any storage. M has shape (N, nPix) — small, fits in RAM.

      Pass 2: compute coefficients C = (IIt_inv @ M).T and write residuals
              out chunk by chunk: residual[t,j] = Y[t,j] - sum_k C[j,k]*I[k,t]

    Memory usage:
        - Per chunk in RAM: chunk_size_T * H * W * 4 bytes (~1-2 GB)
        - Persistent state: M (N * nPix * 8 bytes ≈ a few MB) and
          C_2d (nPix * N * 4 bytes ≈ a few MB)
        - Output: shape (T, H, W) float32, allocated in RAM or as a
          memmap to disk.

    The math is identical to the original pixel-chunked version. The
    only difference is the order of operations, which gives a far more
    favourable I/O pattern when the input is a large memmap.

    Parameters
    ----------
    mov : np.ndarray, shape (T, H, W)
        Input movie. May be a memmap, h5py Dataset, or in-memory array.
        Must support sliced reads of the form ``mov[t_start:t_end]``.
    ref_sigs : np.ndarray, shape (N, T) or (T,)
        Reference traces to project out. A single 1D trace is accepted.
    rem_offset : bool
        If True, mean-center each reference trace and prepend an all-ones
        row, so the regression also absorbs each pixel's DC level.
    chunk_size_T : int
        Frames per chunk during both passes. RAM per chunk is roughly
        chunk_size_T * H * W * 4 bytes. Default 10000 gives ~1.6 GB
        chunks for 200×200 movies.
    output_path : str or Path, optional
        If given, write the residual movie directly to a .npy memmap.
        For movies whose float32 representation exceeds available RAM,
        this is the only option that works. If None, allocate in RAM.
    use_gpu : bool
        If True, run the per-chunk matmuls on GPU via cupy. Saves a few
        seconds per chunk for large H*W. The accumulator M and the
        coefficient matrix C stay on CPU regardless.

    Returns
    -------
    residual_mov : np.ndarray or np.memmap, shape (T, H, W), float32
        C-contiguous. If ``output_path`` was given, this is a memmap
        backed by that file.
    scale_imgs : np.ndarray, shape (H, W, N) or (H, W, N+1)
        Per-pixel regression coefficients, reshaped to images. One image
        per reference trace (plus one for the offset if ``rem_offset=True``).
        Useful for diagnostics — e.g. the x-shift coefficient image shows
        which pixels are most motion-sensitive.
    """
    T, H, W = mov.shape
    nPix = H * W

    # ── Validate and shape ref_sigs ────────────────────────────────────
    ref_sigs = np.asarray(ref_sigs, dtype=np.float32)
    if ref_sigs.ndim == 1:
        ref_sigs = ref_sigs[np.newaxis, :]
    assert ref_sigs.shape[1] == T, f"ref_sigs.shape[1]={ref_sigs.shape[1]} must equal T={T}"

    # ── Build the regressor matrix I ──────────────────────────────────
    # If rem_offset: mean-center each reference and prepend a row of ones.
    # The ones row absorbs each pixel's DC level so that the other
    # coefficients reflect only the time-varying relationship.
    if rem_offset:
        dref = ref_sigs - ref_sigs.mean(axis=1, keepdims=True)
        design = np.vstack([np.ones((1, T), dtype=np.float32), dref])  # (N, T)
    else:
        design = ref_sigs
    N = design.shape[0]

    # ── Precompute the per-pixel projection operator ──────────────────
    # The closed-form for per-pixel coefficients is:
    #     C[j] = Y[:, j] @ I.T @ inv(I @ I.T)
    # which we can vectorize across pixels as:
    #     C = (IIt_inv @ M).T  where M = I @ Y  is shape (N, nPix)
    #
    # Ridge term lam guards against singular I @ I.T when references
    # are nearly collinear. A tiny lam is regularization, not denoising.
    IIt = design @ design.T
    lam = 1e-6
    IIt_inv = np.linalg.inv(IIt + lam * np.eye(N, dtype=np.float64)).astype(np.float32)

    # ── Pass 1: accumulate sufficient statistics M = I @ Y ────────────
    # We never materialize Y as (T, nPix). Instead we accumulate
    #     M += I[:, t_start:t_end] @ Y_chunk
    # where Y_chunk is the (chunk_T, nPix) reshape of the time-slice
    # mov[t_start:t_end]. Because mov is frame-major, this slice is a
    # contiguous read — fast on any storage.
    #
    # M is accumulated in float64 because it sums O(T) float32 values
    # per pixel and float32 starts losing precision around 10^7 sums.
    M = np.zeros((N, nPix), dtype=np.float64)

    # Optional GPU setup. The GPU path avoids transferring I once per
    # chunk; we send it up front and reuse.
    if use_gpu:
        design_gpu = cp.asarray(design)

    print(f"  Pass 1 of 2: accumulating sufficient statistics ({T} frames)")
    for t_start in range(0, T, chunk_size_T):
        t_end = min(t_start + chunk_size_T, T)
        chunk_T = t_end - t_start

        # Read this time-chunk as a contiguous block. For a memmap on
        # disk this triggers actual I/O of chunk_T * H * W * 4 bytes.
        chunk = np.asarray(mov[t_start:t_end], dtype=np.float32)
        # (chunk_T, H, W) -> (chunk_T, nPix). Free view since contiguous.
        Y_chunk = chunk.reshape(chunk_T, nPix)
        # I[:, t_start:t_end] is a (N, chunk_T) view, free.
        design_slice = design[:, t_start:t_end]

        # M += I_slice @ Y_chunk: (N, chunk_T) @ (chunk_T, nPix) = (N, nPix)
        # This is the dominant compute. On CPU it uses MKL/OpenBLAS;
        # on GPU the matmul is much faster but adds transfer overhead.
        if use_gpu:
            Y_gpu = cp.asarray(Y_chunk, dtype=cp.float64)
            design_slice_gpu = cp.asarray(design_slice, dtype=cp.float64)
            M_chunk = cp.asnumpy(design_slice_gpu @ Y_gpu)  # already float64
            del Y_gpu, design_slice_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            M_chunk = design_slice.astype(np.float64) @ Y_chunk.astype(np.float64)

        M += M_chunk.astype(np.float64)

        del chunk, Y_chunk, M_chunk

    # ── Compute per-pixel coefficients C from M ───────────────────────
    # C[j] = M[:, j] @ IIt_inv ↔ C.T = IIt_inv @ M ↔ C = (IIt_inv @ M).T
    # C has shape (nPix, N) — small enough to keep in RAM.
    C_2d = (IIt_inv @ M.astype(np.float32)).T

    # Free M now — we don't need it for Pass 2.
    del M

    # ── Allocate output in C-contiguous (T, H, W) layout ──────────────
    # Writing residuals frame-by-frame gives a contiguous output. This
    # is important for downstream code that may want
    # ``residual.reshape(T, nPix)`` as a free view.
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(T, H, W),
        )
    else:
        out = np.empty((T, H, W), dtype=np.float32)

    # ── Pass 2: compute residuals and write to output in time-chunks ──
    # For each time-chunk, the residual at frame t is:
    #     residual[t, j] = Y[t, j] - sum_k C_2d[j, k] * I[k, t]
    # which in matrix form is:
    #     residual_chunk = Y_chunk - (C_2d @ I[:, t_start:t_end]).T
    # The per-chunk reconstruction (C_2d @ I_slice) has shape
    # (nPix, chunk_T); transposing gives (chunk_T, nPix), matching Y_chunk.
    print("  Pass 2 of 2: computing residuals")

    if use_gpu:
        C_gpu = cp.asarray(C_2d)

    for t_start in range(0, T, chunk_size_T):
        t_end = min(t_start + chunk_size_T, T)
        chunk_T = t_end - t_start

        chunk = np.asarray(mov[t_start:t_end], dtype=np.float32)
        Y_chunk = chunk.reshape(chunk_T, nPix)
        design_slice = design[:, t_start:t_end]  # (N, chunk_T)

        if use_gpu:
            design_slice_gpu = cp.asarray(design_slice)
            recon = cp.asnumpy((C_gpu @ design_slice_gpu).T)  # (chunk_T, nPix)
            del design_slice_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            recon = (C_2d @ design_slice).T  # (chunk_T, nPix)

        residual = Y_chunk - recon  # (chunk_T, nPix)
        out[t_start:t_end] = residual.reshape(chunk_T, H, W)

        del chunk, Y_chunk, recon, residual

    # ── Cleanup ───────────────────────────────────────────────────────
    if use_gpu:
        del design_gpu, C_gpu
        cp.get_default_memory_pool().free_all_blocks()

    if output_path is not None:
        out.flush()

    # Reshape coefficients back to (H, W, N) image stack
    scale_imgs = C_2d.reshape(H, W, N)

    return out, scale_imgs


def preprocess_movie(
    mov,
    mc_trace,
    bleach_order=3,
    hz=1000.0,
    chunk_size=2000,
    output_path=None,
):
    T, H, W = mov.shape
    mc_trace = np.array(mc_trace, dtype=np.float32)

    # Streaming global F (avoid materializing mov)
    global_F = np.empty(T, dtype=np.float32)
    chunk_T = 10000
    for s in range(0, T, chunk_T):
        e = min(s + chunk_T, T)
        global_F[s:e] = np.asarray(mov[s:e]).mean(axis=(1, 2))

    x = mc_trace[:, 0]
    y = mc_trace[:, 1]
    ref_sigs = np.vstack(
        [
            global_F,
            x,
            y,
            x**2,
            y**2,
            x * y,
        ]
    ).astype(np.float32)

    print("Projecting out global fluorescence + motion regressors...")
    mov_clean, scale_imgs = see_residuals(
        mov,
        ref_sigs,
        rem_offset=True,
        output_path=output_path,  # forwards to memmap if given
    )

    # ── Per-pixel polynomial bleach removal on GPU ──────────────────────
    print("Fitting per-pixel polynomial bleach trend...")
    t_norm = np.linspace(-1, 1, T, dtype=np.float32)
    P = np.column_stack([t_norm**k for k in range(bleach_order + 1)]).astype(np.float32)

    P_gpu = cp.asarray(P)
    PtP_inv = cp.linalg.inv(P_gpu.T @ P_gpu)
    pinvP = PtP_inv @ P_gpu.T

    # Now mov_clean is C-contiguous (T, H, W), so this reshape is a free view.
    nPix = H * W
    mov_hw = mov_clean.reshape(T, nPix)

    for start in range(0, nPix, chunk_size):
        end = min(start + chunk_size, nPix)
        Y = cp.asarray(np.ascontiguousarray(mov_hw[:, start:end].T))  # (chunk, T)
        betas = Y @ pinvP.T
        Y -= betas @ P_gpu.T
        mov_hw[:, start:end] = cp.asnumpy(Y).T
        del Y, betas
        cp.get_default_memory_pool().free_all_blocks()

    del P_gpu, PtP_inv, pinvP
    cp.get_default_memory_pool().free_all_blocks()

    if output_path is not None:
        mov_clean.flush()

    return mov_clean, global_F


def preprocess_movie_light(
    mov,
    mc_trace,
    hz=1000.0,
    chunk_size=2000,
    chunk_size_T=10000,
    output_path=None,
    smooth_motion=True,
    motion_smooth_method="savgol",
    motion_window_length=9,
    motion_polyorder=2,
    include_global=False,
    include_quadratic=True,
    include_interaction=True,
):
    """
    Lighter preprocessing for voltage movies.

    Compared with the heavier version, this one is more conservative:
    1. optionally computes global fluorescence
    2. regresses out motion terms (and optionally global_F)
    3. skips per-pixel polynomial bleach fitting

    This is useful when the full preprocessing stack seems to suppress real
    voltage signals more than it helps.

    Parameters
    ----------
    mov : array-like, shape (T, H, W)
        Input movie. Can be in RAM or memmap-backed.
    mc_trace : array-like, shape (T, 2)
        Motion trace with columns [x, y].
    hz : float
        Sampling rate in Hz.
    chunk_size : int
        Pixel chunk size for downstream GPU ops inside see_residuals.
    chunk_size_T : int
        Time chunk size for streaming global_F and see_residuals.
    output_path : str or Path or None
        If provided, writes the cleaned movie to a memmap on disk.
    smooth_motion : bool
        Whether to lightly smooth x/y motion traces before building regressors.
    motion_smooth_method : {"savgol", "none"}
        Smoothing method for motion traces.
    motion_window_length : int
        Savitzky-Golay window length in samples. Must be odd.
    motion_polyorder : int
        Savitzky-Golay polynomial order.
    include_global : bool
        If True, include global fluorescence as a nuisance regressor.
        Start with False if you are worried about removing real shared signal.
    include_quadratic : bool
        If True, include x^2 and y^2 terms.
    include_interaction : bool
        If True, include x*y term.

    Returns
    -------
    mov_clean : np.ndarray or np.memmap
        Cleaned movie, shape (T, H, W), float32.
    global_F : np.ndarray
        Streaming global fluorescence trace, shape (T,).
    ref_sigs : np.ndarray
        Regressor matrix actually used, shape (N, T).
    scale_imgs : np.ndarray
        Per-pixel regression coefficient images from see_residuals.
    """
    T, H, W = mov.shape
    mc_trace = np.asarray(mc_trace, dtype=np.float32)
    if mc_trace.shape != (T, 2):
        raise ValueError(f"mc_trace must have shape {(T, 2)}, got {mc_trace.shape}")

    # ── 1. Streaming global fluorescence ──────────────────────────────────
    global_F = np.empty(T, dtype=np.float32)
    for s in range(0, T, chunk_size_T):
        e = min(s + chunk_size_T, T)
        global_F[s:e] = np.asarray(mov[s:e], dtype=np.float32).mean(axis=(1, 2))

    # ── 2. Motion regressors (optionally smoothed) ────────────────────────
    x = mc_trace[:, 0].copy()
    y = mc_trace[:, 1].copy()

    if smooth_motion and motion_smooth_method.lower() == "savgol":
        wl = int(motion_window_length)
        if wl < 3:
            wl = 3
        if wl % 2 == 0:
            wl += 1
        if wl <= motion_polyorder:
            wl = motion_polyorder + 2
            if wl % 2 == 0:
                wl += 1
        if wl > T:
            wl = T if T % 2 == 1 else T - 1

        if wl > motion_polyorder:
            x = savgol_filter(
                x, window_length=wl, polyorder=motion_polyorder, mode="interp"
            ).astype(np.float32)
            y = savgol_filter(
                y, window_length=wl, polyorder=motion_polyorder, mode="interp"
            ).astype(np.float32)

    elif smooth_motion and motion_smooth_method.lower() != "none":
        raise ValueError("motion_smooth_method must be 'savgol' or 'none'")

    regressors = []
    if include_global:
        regressors.append(global_F)
    regressors.extend([x, y])
    if include_quadratic:
        regressors.extend([x**2, y**2])
    if include_interaction:
        regressors.append(x * y)

    ref_sigs = np.vstack(regressors).astype(np.float32)

    reg_names = []
    if include_global:
        reg_names.append("global_F")
    reg_names.extend(["x", "y"])
    if include_quadratic:
        reg_names.extend(["x2", "y2"])
    if include_interaction:
        reg_names.append("xy")

    print(f"Projecting out regressors: {', '.join(reg_names)}")
    mov_clean, scale_imgs = see_residuals(
        mov,
        ref_sigs,
        rem_offset=True,
        chunk_size_T=chunk_size_T,
        output_path=output_path,
        use_gpu=True,
    )

    if output_path is not None:
        mov_clean.flush()

    return mov_clean, global_F, ref_sigs, scale_imgs


def plot_pcs_large(
    result: dict,
    mov_mean: np.ndarray,
    cid=None,
    crop_padding: int = 15,
    figsize_per_pc: tuple = (4, 4),
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """
    Large standalone figure of PC spatial maps, auto-cropped to the ROI bounding box.
    Intended for sharing with collaborators for PC inspection.

    Parameters
    ----------
    result : dict
        Output of extract_roi_trace.
    mov_mean : np.ndarray, shape (H, W)
        Mean image of the movie, used as background.
    cid : int or str, optional
        ROI id, used in the figure title.
    crop_padding : int
        Pixels of context to keep around the ROI bounding box.
    figsize_per_pc : tuple
        (width, height) in inches for each PC subplot.
    save_path : str or Path, optional
        If given, save figure to this path. Otherwise just display.
    dpi : int
        Figure DPI for saved output.
    """
    import matplotlib.pyplot as plt

    n_pcs = result["all_pc_maps"].shape[0]
    var_exp = result["var_exp"]
    selected = result["pc_index"]
    pc_map = result["pc_map"]  # has the ROI footprint — use to find bbox
    H, W = mov_mean.shape

    # ── Compute bounding box from the ROI footprint ──────────────────────────
    roi_pixels = pc_map != 0
    if not roi_pixels.any():
        # Fall back to any PC having content
        roi_pixels = np.any(result["all_pc_maps"] != 0, axis=0)

    rows = np.where(roi_pixels.any(axis=1))[0]
    cols = np.where(roi_pixels.any(axis=0))[0]
    r0 = max(0, rows.min() - crop_padding)
    r1 = min(H, rows.max() + crop_padding + 1)
    c0 = max(0, cols.min() - crop_padding)
    c1 = min(W, cols.max() + crop_padding + 1)

    # ── Lay out as a grid ─────────────────────────────────────────────────────
    # Aim for ~roughly-square grid: e.g. 6 PCs → 2x3, 12 PCs → 3x4
    n_cols = int(np.ceil(np.sqrt(n_pcs)))
    n_rows = int(np.ceil(n_pcs / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_pc[0] * n_cols, figsize_per_pc[1] * n_rows),
        squeeze=False,
    )

    bg_cropped = mov_mean[r0:r1, c0:c1]

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        if i >= n_pcs:
            ax.axis("off")
            continue

        # Background: mean image, cropped
        ax.imshow(bg_cropped, cmap="gray_r", interpolation="nearest")

        # Overlay: PC map, cropped, with symmetric color limits around zero
        m = result["all_pc_maps"][i][r0:r1, c0:c1]
        m_nonzero = m[m != 0]
        if m_nonzero.size > 0:
            vmax = np.abs(m_nonzero).max()
        else:
            vmax = 1.0

        im = ax.imshow(
            np.ma.masked_where(m == 0, m),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.85,
            interpolation="nearest",
        )

        title = f"PC{i + 1}   {var_exp[i] * 100:.1f}% var"
        ax.set_title(
            title,
            fontsize=14,
            color="black" if i == selected else "black",
            fontweight="bold" if i == selected else "normal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colorbar per subplot — small and tucked in
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"ROI {cid} — PC spatial maps (cropped to ROI bbox + {crop_padding}px)",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
    return fig

"""Z-focus (image sharpness) estimates for motion-corrected movies.

The metric is the variance of the 5-point Laplacian inside a mask, with each
frame divided by its mask mean first so that whole-frame brightness changes do
not move it (as in 260817_camp_recording_focus_qc_clean.ipynb).

At high frame rates the raw metric is dominated by shot noise (~75 % of it at
1 kHz on 10-frame bins in the 20261002 PV-Cre movies), and that term scales as
1/F, so dimming alone raises it. `laplacian_focus` removes it by default: each
bin of B frames is split into even and odd frames, D = (mean_even - mean_odd)/2
has the same noise as the bin mean M but no image structure, and
var(lap(M)) - var(lap(D)) is the sharpness with the noise term removed.

Interpretation: lower = less sharp (defocus). The metric is flattest at best
focus (small z moves from a well-focused plane change it little), it does not
tell up from down, and it is relative, not calibrated in um. Masks should
exclude the cell under study (use an annulus) so its own signal does not leak in.

A MATLAB port with the same defaults lives next to this file:
focus_masks.m and laplacian_focus.m.
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation


def laplacian2d(x: np.ndarray) -> np.ndarray:
    """5-point Laplacian of a (n, H, W) stack, valid part only -> (n, H-2, W-2)."""
    return (x[:, :-2, 1:-1] + x[:, 2:, 1:-1] + x[:, 1:-1, :-2] + x[:, 1:-1, 2:]
            - 4 * x[:, 1:-1, 1:-1])


def focus_masks(
    proj: np.ndarray,
    labels: Optional[np.ndarray] = None,
    shifts: Optional[np.ndarray] = None,
    roi_ids: Optional[Iterable[int]] = None,
    tissue_pct: float = 40,
    noncell_excl_px: int = 8,
    annulus_px: Tuple[int, int] = (3, 15),
    border_px: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Masks for `laplacian_focus`.

    Parameters
    ----------
    proj : (H, W) mean projection of the motion-corrected movie.
    labels : (H, W) int ROI labels (0 = background), or None for tissue only.
    shifts : (T, 2) rigid shifts from motion correction. The border is set to
        the largest excursion from the median shift + 2 px, so shifted frame
        edges never enter a mask.
    roi_ids : ROIs that get an annulus / roi mask (default: all in `labels`).
    tissue_pct : tissue = proj above this percentile (inside the border).
    noncell_excl_px : 'noncell' stays this many px away from every ROI.
    annulus_px : (inner, outer) dilation of each ROI for 'annulus<id>'.
    border_px : override the shift-derived border.

    Returns
    -------
    masks : dict with 'tissue', 'noncell' (if labels), and per ROI
        'annulus<id>' (ring, excludes the cell) and 'roi<id>' (cell + inner
        ring; contaminated by the cell's own signal, reference only).
    border : border width used, px.
    """
    H, W = proj.shape
    if border_px is None:
        if shifts is not None and len(shifts):
            s = np.asarray(shifts, dtype=float)
            border_px = int(np.ceil(np.abs(s - np.median(s, axis=0)).max())) + 2
        else:
            border_px = 2
    core = np.zeros((H, W), bool)
    core[border_px:H - border_px, border_px:W - border_px] = True
    tissue = (proj > np.percentile(proj[core], tissue_pct)) & core
    masks = {'tissue': tissue}
    if labels is not None:
        masks['noncell'] = tissue & ~binary_dilation(labels > 0, iterations=noncell_excl_px)
        ids = [int(c) for c in np.unique(labels) if c != 0] if roi_ids is None else roi_ids
        for c in ids:
            roi = labels == c
            inner = binary_dilation(roi, iterations=annulus_px[0])
            masks[f'annulus{c}'] = binary_dilation(roi, iterations=annulus_px[1]) & ~inner & core
            masks[f'roi{c}'] = inner & core
    empty = [k for k, m in masks.items() if not m.any()]
    if empty:
        raise ValueError(f"empty focus mask(s): {empty} (border {border_px} px)")
    return masks, border_px


def laplacian_focus(
    mov,
    masks: Dict[str, np.ndarray],
    bin_frames: int = 10,
    debias: bool = True,
    chunk: int = 2000,
    relative: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Laplacian-variance focus per mask, on bins of `bin_frames` frames.

    Parameters
    ----------
    mov : (T, H, W) array or memmap (motion-corrected).
    masks : dict name -> (H, W) bool, e.g. from `focus_masks`.
    bin_frames : frames per bin. Must be even when debias=True. Use
        bin_frames=1, debias=False for the original per-frame metric.
    debias : subtract the shot-noise term (even/odd split, see module doc).
    chunk : frames read per pass (rounded down to a multiple of bin_frames).
    relative : also return focus as fractional change from its median.

    Returns
    -------
    dict name -> {'focus': (nb,) debiased (or raw) metric,
                  'focus_rel': focus / median - 1 (if relative),
                  'noise': (nb,) removed noise term (zeros if not debias),
                  'F': (nb,) mask mean of the bin}
    plus out['_bin_index'] = (nb,) index of each bin's first frame.
    Bins cover frames [0, nb * bin_frames); a trailing partial bin is dropped.
    """
    if debias and (bin_frames < 2 or bin_frames % 2):
        raise ValueError("debias=True needs an even bin_frames >= 2")
    T, H, W = mov.shape
    nb = T // bin_frames
    inner = {k: np.asarray(m, bool)[1:-1, 1:-1] for k, m in masks.items()}
    full = {k: np.asarray(m, bool) for k, m in masks.items()}
    out = {k: {'focus': np.zeros(nb), 'noise': np.zeros(nb), 'F': np.zeros(nb)} for k in masks}
    step = max(chunk // bin_frames, 1) * bin_frames
    for s in range(0, nb * bin_frames, step):
        blk = np.asarray(mov[s:min(s + step, nb * bin_frames)], dtype=np.float32)
        r = blk.reshape(-1, bin_frames, H, W)
        M = r.mean(axis=1)
        LM = laplacian2d(M)
        if debias:
            D = (r[:, 0::2].mean(axis=1) - r[:, 1::2].mean(axis=1)) / 2
            LD = laplacian2d(D)
        i = s // bin_frames
        for k in masks:
            mu = M[:, full[k]].mean(axis=1).astype(np.float64)
            vm = LM[:, inner[k]].var(axis=1) / mu ** 2
            vd = LD[:, inner[k]].var(axis=1) / mu ** 2 if debias else np.zeros_like(vm)
            out[k]['focus'][i:i + len(M)] = vm - vd
            out[k]['noise'][i:i + len(M)] = vd
            out[k]['F'][i:i + len(M)] = mu
    if relative:
        for k in masks:
            f = out[k]['focus']
            out[k]['focus_rel'] = f / np.median(f) - 1
    out['_bin_index'] = np.arange(nb) * bin_frames
    return out


def bin_times(t: np.ndarray, bin_frames: int = 10) -> np.ndarray:
    """Mean frame time of each focus bin (same binning as laplacian_focus)."""
    nb = len(t) // bin_frames
    return np.asarray(t[:nb * bin_frames], dtype=float).reshape(nb, bin_frames).mean(axis=1)

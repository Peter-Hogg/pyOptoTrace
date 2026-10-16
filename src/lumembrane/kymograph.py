"""Kymographs along dendrites and other elongated structures.

Ported from the Cohen Lab MATLAB tools polyLineKymo2.m / apply_polyLineKymo2.m
/ lineKymo.m / wedgeKymo.m (AEC). The interactive getline() step is replaced
by napari: draw a Path shape along the structure and pass its vertices in.

    viewer = napari.view_image(mean_img)
    # add a Shapes layer, draw a Path along the dendrite, then:
    verts = verts_from_napari(viewer.layers['Shapes'])
    res   = polyline_kymograph(mov, verts, dl=4.0, dp=6.0)
    plot_kymograph(res, img=mean_img, hz=970.87)

Conventions (matching the rest of utils/ and napari):
  - movies are (T, H, W); vertices are (row, col) pixel coordinates
    (pass xy=True for (x, y) points from e.g. matplotlib ginput)
  - a straight-line kymograph (lineKymo) is just a two-vertex path
  - dl and dp are in pixels: dl is the bin length along the path, dp the full
    width perpendicular to it; helpers.pixel_size_um gives the um_per_px that
    plot_kymograph takes for a micron distance axis

The movie is streamed in time chunks, so a memmapped cache
(mov1_corrected_mov.npy opened with mmap_mode='r') can be passed directly
without loading it into RAM. The kymograph comes back as (n_bins, T):
rows are position along the structure, columns are time.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as _MplPath
from tqdm.auto import tqdm


# ── Bin geometry ──────────────────────────────────────────────────────────────
def _polyline_boxes(verts, dl, dp, shape):
    """Assign pixels to dl x dp rectangles marching along a polyline.

    Box centres sit at arc positions 0, dl, 2*dl, ... along each segment
    (the MATLAB placement); a pixel belongs to the nearest centre of the
    segment when it lies within dp/2 of the segment axis. At a joint between
    segments the end boxes overlap, so a pixel there can appear in two bins —
    same behaviour as the MATLAB rectangles.

    Returns (pixel index lists, arc distance of each bin centre, outlines).
    """
    H, W = shape
    verts = np.asarray(verts, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 2:
        raise ValueError("verts must be an (N >= 2, 2) array of (row, col) points")

    idx_lists, dists, outlines = [], [], []
    cum_len = 0.0
    for p0, p1 in zip(verts[:-1], verts[1:]):
        seg_len = float(np.hypot(*(p1 - p0)))
        if seg_len == 0.0:
            continue
        t = (p1 - p0) / seg_len  # unit tangent, (row, col)
        n = np.array([-t[1], t[0]])  # unit normal
        n_steps = int(seg_len // dl) + 1  # centres at 0, dl, ..., <= seg_len

        # Pixel grid restricted to this segment's bounding box, padded by the
        # half-diagonal of one box so no member pixel is cropped away.
        pad = 0.5 * float(np.hypot(dl, dp)) + 1.0
        r0 = max(int(np.floor(min(p0[0], p1[0]) - pad)), 0)
        r1 = min(int(np.ceil(max(p0[0], p1[0]) + pad)) + 1, H)
        c0 = max(int(np.floor(min(p0[1], p1[1]) - pad)), 0)
        c1 = min(int(np.ceil(max(p0[1], p1[1]) + pad)) + 1, W)
        # A segment entirely off-image inverts the clamped box; make it empty
        # so its bins come back pixel-less (-> NaN rows) instead of crashing.
        r1, c1 = max(r1, r0), max(c1, c0)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        rel_r, rel_c = rr - p0[0], cc - p0[1]
        s = rel_r * t[0] + rel_c * t[1]  # along-segment coordinate
        p = rel_r * n[0] + rel_c * n[1]  # perpendicular coordinate

        bin_id = np.rint(s / dl).astype(int)
        valid = (np.abs(p) <= dp / 2) & (bin_id >= 0) & (bin_id < n_steps)
        flat = (rr * W + cc)[valid]
        bins = bin_id[valid]

        half_l, half_p = t * (dl / 2), n * (dp / 2)
        for j in range(n_steps):
            idx_lists.append(flat[bins == j])
            dists.append(cum_len + j * dl)
            centre = p0 + j * dl * t
            outlines.append(
                np.array(
                    [
                        centre - half_l - half_p,
                        centre - half_l + half_p,
                        centre + half_l + half_p,
                        centre + half_l - half_p,
                        centre - half_l - half_p,
                    ]
                )
            )
        cum_len += seg_len
    return idx_lists, np.asarray(dists), outlines


# ── Chunked extraction ────────────────────────────────────────────────────────
def _chunked_box_means(mov, idx_lists, chunk_size):
    """Mean intensity per pixel-index list per frame, streaming over time.

    Only the union of member pixels is pulled from each chunk, so the working
    set stays at (chunk_size x n_union_pixels) regardless of movie size.
    """
    T = mov.shape[0]
    flat_mov = mov.reshape(T, -1)
    n_boxes = len(idx_lists)
    counts = np.array([len(ix) for ix in idx_lists])

    nonempty = [ix for ix in idx_lists if len(ix)]
    union = np.unique(np.concatenate(nonempty)) if nonempty else np.array([], dtype=int)

    weights = np.zeros((n_boxes, len(union)), dtype=np.float32)
    for i, ix in enumerate(idx_lists):
        if len(ix):
            weights[i, np.searchsorted(union, ix)] = 1.0 / len(ix)

    kymo = np.empty((n_boxes, T), dtype=np.float32)
    for t0 in tqdm(range(0, T, chunk_size), desc="kymograph", leave=False):
        t1 = min(t0 + chunk_size, T)
        sub = flat_mov[t0:t1][:, union].astype(np.float32, copy=False)
        kymo[:, t0:t1] = weights @ sub.T

    if (counts == 0).any():
        kymo[counts == 0] = np.nan
        print(
            f"  [kymo] {int((counts == 0).sum())}/{n_boxes} bins have no "
            f"pixels (off-image or dp too thin) -> NaN rows"
        )
    return kymo, counts


# ── Public API ────────────────────────────────────────────────────────────────
def polyline_kymograph(mov, verts, dl=4.0, dp=6.0, xy=False, chunk_size=2000):
    """Kymograph along a polyline: mean of dl x dp boxes vs time.

    Port of polyLineKymo2 / apply_polyLineKymo2 (and lineKymo, which is the
    two-vertex case). Non-interactive: draw the path in napari and pass its
    vertices, e.g. via verts_from_napari().

    Parameters
    ----------
    mov : (T, H, W) array or memmap
    verts : (N, 2) array
        Polyline vertices as (row, col); (x, y) if xy=True.
    dl, dp : float
        Bin length along the path and full width perpendicular to it, pixels.
    chunk_size : int
        Frames per read when streaming a memmapped movie.

    Returns
    -------
    dict with
        kymo     : (n_bins, T) float32, NaN rows for empty bins
        dist_px  : (n_bins,) arc distance of each bin centre along the path
        n_px     : (n_bins,) pixel count per bin
        outlines : list of (5, 2) box corner paths, for QC plotting
        verts, dl, dp, bin_px
    """
    if mov.ndim != 3:
        raise ValueError(f"mov must be (T, H, W), got shape {mov.shape}")
    if dl <= 0 or dp <= 0:
        raise ValueError("dl and dp must be positive")
    verts = np.asarray(verts, dtype=float)
    if xy:
        verts = verts[:, ::-1]

    idx_lists, dists, outlines = _polyline_boxes(verts, dl, dp, mov.shape[1:])
    kymo, counts = _chunked_box_means(mov, idx_lists, chunk_size)
    return {
        "kymo": kymo,
        "dist_px": dists,
        "n_px": counts,
        "outlines": outlines,
        "verts": verts,
        "dl": dl,
        "dp": dp,
        "bin_px": float(dl),
    }


def wedge_kymograph(mov, line1, line2, n_segments, xy=False, chunk_size=2000):
    """Kymograph between two boundary lines, in n_segments quadrilaterals.

    Port of wedgeKymo. Each boundary is a two-point line; both are split into
    n_segments equal parts and each quadrilateral's mean intensity is returned
    per frame. Useful for radial / tapering structures where box width should
    grow along the path.

    Parameters
    ----------
    mov : (T, H, W) array or memmap
    line1, line2 : (2, 2) arrays
        Endpoints of the two boundaries as (row, col); (x, y) if xy=True.

    Returns
    -------
    dict with the same keys as polyline_kymograph (dist_px is the arc
    position of each bin centre along line1).
    """
    if mov.ndim != 3:
        raise ValueError(f"mov must be (T, H, W), got shape {mov.shape}")
    line1 = np.asarray(line1, dtype=float)
    line2 = np.asarray(line2, dtype=float)
    if xy:
        line1, line2 = line1[:, ::-1], line2[:, ::-1]
    H, W = mov.shape[1:]

    b1 = np.linspace(line1[0], line1[1], n_segments + 1)
    b2 = np.linspace(line2[0], line2[1], n_segments + 1)

    idx_lists, outlines = [], []
    for j in range(n_segments):
        quad = np.array([b1[j], b1[j + 1], b2[j + 1], b2[j]])
        r0 = max(int(np.floor(quad[:, 0].min())) - 1, 0)
        r1 = min(int(np.ceil(quad[:, 0].max())) + 2, H)
        c0 = max(int(np.floor(quad[:, 1].min())) - 1, 0)
        c1 = min(int(np.ceil(quad[:, 1].max())) + 2, W)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        pts = np.column_stack([rr.ravel(), cc.ravel()])
        inside = _MplPath(quad).contains_points(pts)
        idx_lists.append(pts[inside, 0] * W + pts[inside, 1])
        outlines.append(np.vstack([quad, quad[:1]]))

    kymo, counts = _chunked_box_means(mov, idx_lists, chunk_size)
    seg_len = float(np.hypot(*(line1[1] - line1[0]))) / n_segments
    dists = (np.arange(n_segments) + 0.5) * seg_len
    return {"kymo": kymo, "dist_px": dists, "n_px": counts, "outlines": outlines, "bin_px": seg_len}


def verts_from_napari(shapes_layer, index=0):
    """Vertices of a napari Shapes path as an (N, 2) (row, col) array.

    Extra leading dims (e.g. a time axis when the path was drawn on a movie
    layer) are dropped. Draw with the Path tool; a Polygon also works but is
    treated as an open path — the closing edge is ignored.
    """
    data = np.asarray(shapes_layer.data[index], dtype=float)
    return data[:, -2:]


# ── QC plot ───────────────────────────────────────────────────────────────────
def plot_kymograph(result, img=None, hz=None, um_per_px=None, cmap="magma", figsize=(12, 4)):
    """Two-panel QC: bin layout over the image, kymograph as distance vs time.

    Pass hz for a seconds time axis (frames otherwise) and um_per_px for a
    micron distance axis (pixels otherwise). img is typically a mean or
    percentile projection of the movie.
    """
    kymo = result["kymo"]
    scale = um_per_px if um_per_px else 1.0
    dist = result["dist_px"] * scale
    half_bin = result["bin_px"] / 2 * scale
    n_bins, T = kymo.shape
    t_max = T / hz if hz else T

    if img is not None:
        fig, (ax_im, ax_ky) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 2]}
        )
        vmin, vmax = np.percentile(img, [1, 99.5])
        ax_im.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        for corners in result["outlines"]:
            ax_im.plot(corners[:, 1], corners[:, 0], color="cyan", lw=0.4)
        if "verts" in result:
            ax_im.plot(result["verts"][:, 1], result["verts"][:, 0], "r-", lw=0.8)
        ax_im.set_axis_off()
        ax_im.set_title(f"{n_bins} bins")
    else:
        fig, ax_ky = plt.subplots(figsize=figsize)

    im = ax_ky.imshow(
        kymo,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=(0, t_max, dist[-1] + half_bin, dist[0] - half_bin),
    )
    ax_ky.set_xlabel("Time (s)" if hz else "Frame")
    ax_ky.set_ylabel("Distance (µm)" if um_per_px else "Distance (px)")
    fig.colorbar(im, ax=ax_ky, label="F")
    fig.tight_layout()
    return fig

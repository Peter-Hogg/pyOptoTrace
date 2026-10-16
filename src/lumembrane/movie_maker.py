"""
array_to_movie.py
-----------------
Convert a numpy array (T, H, W) or (T, H, W, C) to an MP4 movie.

Dependencies:
    pip install matplotlib imageio[ffmpeg] numpy

Optional:
    pip install tqdm  # for progress bar
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def array_to_movie(
    data: np.ndarray,
    output_path: str = "output.mp4",
    fps: float = 30.0,
    acquisition_rate: float | None = None,  # Hz; if set, auto-generates timestamps
    # --- Colormap & intensity ---
    cmap: str = "inferno",
    clim: tuple[float, float] | None = None,
    clim_percentile: tuple[float, float] = (1, 99),
    # --- Colorbar ---
    colorbar: bool = True,
    colorbar_label: str = "",
    colorbar_ticks: list[float] | None = None,
    # --- Timestamps ---
    timestamps: np.ndarray | None = None,
    timestamp_units: str = "s",
    timestamp_fmt: str = "{:.2f} {}",
    timestamp_position: str = "top-left",  # top-left | top-right | bottom-left | bottom-right
    timestamp_color: str = "white",
    timestamp_fontsize: float = 12,
    # --- Overlays ---
    overlay_fn: Callable[..., None] | list[Callable[..., None]] | None = None,
    # fn(ax, t) or list of fns
    # --- Crop ---
    crop: tuple[int, int, int, int] | None = None,  # (y0, y1, x0, x1)
    # --- Layout ---
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
    title_fontsize: float = 14,
    # --- Output quality ---
    quality: int = 8,  # imageio quality 1-10
    codec: str = "libx264",
    verbose: bool = True,
) -> str:
    """
    Render a (T, H, W) or (T, H, W, C) numpy array as an MP4 movie.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, H, W)       – single-channel, rendered with `cmap`
        Shape (T, H, W, 3/4) – RGB/RGBA, colormap ignored unless forced
    output_path : str
        Destination file (must end in .mp4 or .gif).
    fps : float
        Frames per second.
    cmap : str
        Matplotlib colormap name (single-channel only).
    clim : (vmin, vmax) or None
        Intensity limits. If None, uses `clim_percentile`.
    clim_percentile : (low, high)
        Percentiles used to auto-set clim when clim=None.
    colorbar : bool
        Show a colorbar (single-channel only).
    colorbar_label : str
        Label on the colorbar axis.
    colorbar_ticks : list or None
        Explicit tick positions on colorbar.
    timestamps : np.ndarray or None
        1-D array of length T. If provided, stamps each frame.
    timestamp_units : str
        Unit string appended to timestamp value.
    timestamp_fmt : str
        Format string; receives (value, units).
    timestamp_position : str
        Corner for timestamp text.
    timestamp_color : str
        Color of timestamp text.
    timestamp_fontsize : float
    scalebar : bool
        Overlay a physical scale bar.
    scalebar_um_per_px : float
        Physical size of one image pixel (µm, nm, etc.).
    scalebar_length_um : float or None
        Desired bar length in physical units. Auto-chosen if None.
    scalebar_label : str or None
        Text label for bar. Auto-generated from length if None.
    scalebar_position : str
        Corner for scale bar.
    scalebar_color : str
    scalebar_thickness : float
        Height of bar rectangle in pixels.
    scalebar_fontsize : float
    crop : (y0, y1, x0, x1) or None
        Pixel crop applied before rendering.
    figsize : (w, h) in inches or None
        Auto-derived from image aspect ratio if None.
    dpi : int
    title : str or None
        Static title drawn above image.
    title_fontsize : float
    quality : int
        1–10 encoding quality for imageio.
    codec : str
        ffmpeg codec string.
    verbose : bool
        Print progress.

    Returns
    -------
    str
        Absolute path to the saved movie.
    """

    # ------------------------------------------------------------------ #
    # 0. Validate & pre-process
    # ------------------------------------------------------------------ #
    if data.ndim == 3:
        T, H, W = data.shape
        is_rgb = False
    elif data.ndim == 4 and data.shape[-1] in (3, 4):
        T, H, W, _ = data.shape
        is_rgb = True
    else:
        raise ValueError(f"data must be (T,H,W) or (T,H,W,3/4), got shape {data.shape}")

    if verbose:
        print(f"[array_to_movie] Input shape: {data.shape}  dtype: {data.dtype}")

    # Auto-generate timestamps from acquisition rate
    if acquisition_rate is not None and timestamps is None:
        timestamps = np.arange(T) / acquisition_rate
        if verbose:
            print(
                f"[array_to_movie] Auto timestamps: 0 – {timestamps[-1]:.3f} s @ {acquisition_rate} Hz"
            )

    # Crop
    if crop is not None:
        y0, y1, x0, x1 = crop
        if is_rgb:
            data = data[:, y0:y1, x0:x1, :]
        else:
            data = data[:, y0:y1, x0:x1]
        H, W = data.shape[1], data.shape[2]
        if verbose:
            print(f"[array_to_movie] After crop: H={H}, W={W}")

    # Convert to float32 for processing
    arr = data.astype(np.float32)

    # Intensity limits (single-channel)
    if not is_rgb:
        if clim is None:
            vmin = float(np.nanpercentile(arr, clim_percentile[0]))
            vmax = float(np.nanpercentile(arr, clim_percentile[1]))
        else:
            vmin, vmax = float(clim[0]), float(clim[1])
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        if verbose:
            print(f"[array_to_movie] clim: [{vmin:.4g}, {vmax:.4g}]")
    else:
        # Normalise RGB to [0,1] if needed
        if arr.max() > 1.0:
            arr = arr / 255.0
        norm = None
        cmap_obj = None

    # ------------------------------------------------------------------ #
    # 1. Figure layout
    # ------------------------------------------------------------------ #
    aspect = H / W
    if figsize is None:
        fig_w = 6.0
        fig_w += 0.8 if (colorbar and not is_rgb) else 0
        fig_h = fig_w * aspect
        if title:
            fig_h += 0.4
        figsize = (fig_w, fig_h)

    # ------------------------------------------------------------------ #
    # 2. Scale bar helpers
    # ------------------------------------------------------------------ #
    def _nice_scalebar_length(um_per_px: float, W_px: int) -> float:
        """Choose a round scale bar length ~15% of frame width."""
        target_um = um_per_px * W_px * 0.15
        magnitude = 10 ** np.floor(np.log10(target_um))
        for nice in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
            if nice * magnitude >= target_um * 0.5:
                return nice * magnitude
        return round(target_um)

    def _corner_xy(
        position: str, ax_w: float, ax_h: float, pad_x: float = 0.05, pad_y: float = 0.05
    ):
        """Return (x_frac, y_frac) in axes-fraction coords for a given corner."""
        pos = position.lower()
        x = pad_x if "left" in pos else 1 - pad_x
        y = pad_y if "bottom" in pos else 1 - pad_y
        return x, y

    # ------------------------------------------------------------------ #
    # 3. Frame rendering loop
    # ------------------------------------------------------------------ #
    frames = []

    try:
        from tqdm import trange

        iterator = trange(T, desc="Rendering", disable=not verbose)
    except ImportError:
        iterator = range(T)
        if verbose:
            print("[array_to_movie] (install tqdm for a progress bar)")

    for t in iterator:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.subplots_adjust(
            left=0,
            right=1 if is_rgb or not colorbar else 0.88,
            top=1 if not title else 0.93,
            bottom=0,
        )

        # ---- Image ----
        if is_rgb:
            im = ax.imshow(arr[t], interpolation="nearest")
        else:
            im = ax.imshow(arr[t], cmap=cmap_obj, norm=norm, interpolation="nearest")

        ax.set_axis_off()
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)

        # ---- Title ----
        if title:
            fig.suptitle(title, fontsize=title_fontsize, color="white", y=0.98, fontweight="bold")
            fig.patch.set_facecolor("black")

        # ---- Colorbar ----
        if colorbar and not is_rgb:
            cbar_ax = fig.add_axes([0.89, 0.08, 0.03, 0.84])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label(colorbar_label, color="white", fontsize=10)
            if colorbar_ticks is not None:
                cb.set_ticks(colorbar_ticks)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cbar_ax.set_facecolor("black")
            fig.patch.set_facecolor("black")

        # ---- Timestamp ----
        if timestamps is not None:
            ts_str = timestamp_fmt.format(timestamps[t], timestamp_units)
            tx, ty = _corner_xy(timestamp_position, W, H)
            ha = "left" if "left" in timestamp_position.lower() else "right"
            va = "bottom" if "bottom" in timestamp_position.lower() else "top"
            ax.text(
                tx,
                ty,
                ts_str,
                transform=ax.transAxes,
                color=timestamp_color,
                fontsize=timestamp_fontsize,
                ha=ha,
                va=va,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, ec="none"),
            )

        # ---- User overlays ----
        if overlay_fn is not None:
            fns = overlay_fn if isinstance(overlay_fn, list) else [overlay_fn]
            for fn in fns:
                fn(ax, t)

        # ---- Capture frame ----
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_rgb = np.asarray(buf)[..., :3]
        frames.append(frame_rgb)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Write movie
    # ------------------------------------------------------------------ #
    output_path = os.path.abspath(output_path)
    ext = os.path.splitext(output_path)[-1].lower()

    if verbose:
        print(f"[array_to_movie] Writing {T} frames → {output_path}")

    if ext == ".gif":
        import imageio

        # High quality: use pillow writer with dithering + full palette
        with imageio.get_writer(
            output_path,
            mode="I",
            fps=fps,
            loop=0,
            plugin="pillow",
            quantizer="nq",
            palettesize=256,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    if verbose:
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"[array_to_movie] Done. File size: {size_mb:.2f} MB")

    return output_path

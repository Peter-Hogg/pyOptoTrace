from functools import partial

import numpy as np

try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:  # optional visualization dependency
    ScaleBar = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # NumPy remains a supported CPU fallback
    jax = None
    jnp = np


def pixel_size_um(
    objective_mag: float,
    tube_lens_fl_mm: float,
    objective_brand: str = "olympus",  # nikon/leica=200, olympus=180, zeiss=165
    sensor_pixel_um: float = 6.5,
) -> float:
    """
    Calculate image pixel size in microns.

    Parameters
    ----------
    objective_mag : float
        Objective magnification (e.g. 40.0)
    tube_lens_fl_mm : float
        Tube lens focal length in mm (e.g. 100.0)
    objective_brand : str
        Objective manufacturer — sets the reference focal length
    sensor_pixel_um : float
        Camera sensor pixel pitch in microns (default: 6.5 for ORCA-Flash 4.0)
    """
    reference_fl = {"nikon": 200, "leica": 200, "olympus": 180, "zeiss": 165}
    ref_fl = reference_fl[objective_brand.lower()]
    tube_lens_mag = tube_lens_fl_mm / ref_fl
    total_mag = objective_mag * tube_lens_mag
    um_per_px = sensor_pixel_um / total_mag
    print(f"Tube lens mag:  {tube_lens_mag:.4f}x  ({tube_lens_fl_mm}mm / {ref_fl}mm)")
    print(f"Total mag:      {total_mag:.4f}x")
    print(f"Pixel size:     {um_per_px:.4f} µm/px  ({1 / um_per_px:.2f} px/µm)")
    return um_per_px


# ── Standalone: add scalebar to any existing axes ─────────────────────
def add_scalebar(
    ax,
    um_per_px: float,
    color: str = "white",
    location: str = "lower right",
    length_fraction: float = 0.2,
    frameon: bool = False,
    fixed_value: float = None,  # e.g. 100 for a fixed 100 µm bar
    fixed_units: str = "um",
):
    """
    Add a scale bar to a matplotlib axes showing a calibrated image.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    um_per_px : float
        Pixel size in microns (e.g. from pixel_size_um())
    color : str
        Bar and text color
    location : str
        matplotlib legend-style location string
    length_fraction : float
        Desired bar length as fraction of axes width (auto-snaps to round values)
    frameon : bool
        Draw a box behind the scalebar
    fixed_value : float or None
        Force a specific bar length (e.g. 100). Auto if None.
    fixed_units : str
        Units for fixed_value
    """
    if ScaleBar is None:
        raise ModuleNotFoundError(
            "add_scalebar requires the 'visualization' extra: "
            "pip install 'lumembrane[visualization]'"
        )
    scalebar = ScaleBar(
        um_per_px,
        units="um",
        dimension="si-length",
        color=color,
        location=location,
        length_fraction=length_fraction,
        frameon=frameon,
        fixed_value=fixed_value,
        fixed_units=fixed_units if fixed_value else None,
        box_alpha=0.4,
        scale_loc="bottom",
    )
    ax.add_artist(scalebar)
    return scalebar


# ---------------------------------------------------------------------------
# JIT-compiled core functions
# ---------------------------------------------------------------------------


def _mean_proj(arr):
    return jnp.mean(arr, axis=0)


def _max_proj(arr):
    return jnp.max(arr, axis=0)


def _min_proj(arr):
    return jnp.min(arr, axis=0)


def _std_proj(arr):
    return jnp.std(arr, axis=0)


def _pct_proj(arr, p):
    return jnp.percentile(arr, p, axis=0)


if jax is not None:
    _mean_proj = jax.jit(_mean_proj)
    _max_proj = jax.jit(_max_proj)
    _min_proj = jax.jit(_min_proj)
    _std_proj = jax.jit(_std_proj)
    _pct_proj = partial(jax.jit, static_argnames=("p",))(_pct_proj)


def _chunked(mov, fn, batch_size):
    return np.stack(
        [
            np.array(fn(jnp.asarray(mov[i : i + batch_size].astype(np.float32))))
            for i in range(0, mov.shape[0], batch_size)
        ]
    )


def mean_proj(mov: np.ndarray, batch_size: int = 500) -> np.ndarray:
    n = mov.shape[0]
    chunks = _chunked(mov, _mean_proj, batch_size)
    sizes = [min(batch_size, n - i) for i in range(0, n, batch_size)]
    w = np.array(sizes, dtype=np.float32) / n
    return np.sum(chunks * w[:, None, None], axis=0)


def max_proj(mov: np.ndarray, batch_size: int = 500) -> np.ndarray:
    return np.max(_chunked(mov, _max_proj, batch_size), axis=0)


def min_proj(mov: np.ndarray, batch_size: int = 500) -> np.ndarray:
    return np.min(_chunked(mov, _min_proj, batch_size), axis=0)


def std_proj(mov: np.ndarray, batch_size: int = 500) -> np.ndarray:
    return np.mean(_chunked(mov, _std_proj, batch_size), axis=0)


def median_proj(mov: np.ndarray, n_sample: int = 500, seed: int = 42) -> np.ndarray:
    idx = np.random.default_rng(seed).choice(
        mov.shape[0], size=min(n_sample, mov.shape[0]), replace=False
    )
    return np.array(_pct_proj(jnp.asarray(mov[idx].astype(np.float32)), 50))


def percentile_proj(
    mov: np.ndarray, p: int = 10, n_sample: int = 500, seed: int = 42
) -> np.ndarray:
    idx = np.random.default_rng(seed).choice(
        mov.shape[0], size=min(n_sample, mov.shape[0]), replace=False
    )
    return np.array(_pct_proj(jnp.asarray(mov[idx].astype(np.float32)), p))

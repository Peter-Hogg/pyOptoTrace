# Lumembrane

> From fluorescence movies to membrane-voltage traces.

Lumembrane is a Python toolkit for exploratory functional-imaging workflows. It collects
reusable code for loading Luminos acquisitions, correcting fluorescence traces, extracting
ROI signals, detecting spikes, building kymographs, and exporting compatible data to NWB.

This repository is a clean, public-facing extraction of a working research codebase. It does
not contain raw recordings, subject identifiers, private paths, notebook outputs, or the
original repository history.

> [!IMPORTANT]
> Lumembrane is pre-release research software. Its API may change, results still require
> scientific validation, and it must not be used for clinical decisions.

## What is included

- Photobleaching correction for fast voltage and slow cAMP indicators
- Spike detection, classification, firing-rate estimation, and triggered averages
- PCA-based ROI trace extraction and residual movie preprocessing
- Luminos binary/HDF5 metadata readers and optional NWB export
- Kymographs, calibrated projections, and movie rendering helpers
- Optional GPU acceleration, motion correction, and Pynapple integration
- A small command-line demo, tests, packaging metadata, and release documentation

The original dated analysis notebooks are intentionally excluded. They mix private paths,
embedded outputs, experimental identifiers, and dataset-specific assumptions. Public examples
in this repository use synthetic data instead.

## Install for development

Lumembrane currently targets Python 3.10–3.12.

```bash
git clone <your-future-repository-url>
cd lumembrane
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Install only the extras needed for a workflow:

```bash
# File formats and NWB
python -m pip install -e ".[io]"

# Wavelets and Pynapple session objects
python -m pip install -e ".[analysis]"

# Plotting helpers and MP4 export
python -m pip install -e ".[visualization]"

# CUDA/JAX and motion-correction paths
python -m pip install -e ".[gpu,motion]"
```

CuPy and JAX wheels depend on the local CUDA/driver combination. Check their installation
guides before selecting a GPU extra on a new workstation.

## Quick start

Run the self-contained command-line demo:

```bash
lumembrane demo
```

Or detect spikes in a trace:

```python
import numpy as np

from lumembrane import detect_peaks, spike_triggered_average

sampling_rate = 1_000.0
trace = np.load("trace.npy")
peaks = detect_peaks(trace, fs=sampling_rate, threshold_std=4.0)
time_ms, mean, sem, snippets = spike_triggered_average(
    trace,
    peaks,
    window_ms=(-20, 40),
    fs=sampling_rate,
)
```

Array conventions are consistent across the package:

- movies: `(time, y, x)`
- traces: `(time,)` or `(cell, time)` where documented
- ROI masks: `(y, x)` boolean arrays
- sampling rates: hertz
- event windows: milliseconds unless the parameter says otherwise

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable synthetic example.

## Package map

| Module | Purpose | Extra |
| --- | --- | --- |
| `bleach_correction`, `camp_preprocess` | Trace detrending and ΔF/F₀ | core |
| `spike_analysis`, `spike_tools`, `trace_analysis` | Spike and waveform analysis | core |
| `luminos_io`, `luminos_nwb`, `read_vr_data` | Acquisition, NWB, and VR I/O | `io` |
| `kymograph`, `movie_maker`, `helpers` | Visual and projection utilities | `visualization` |
| `wavelet_voltage`, `session_to_nap` | Wavelet and Pynapple workflows | `analysis` |
| `signal_extraction`, `checkerboard_denoise`, `luminos_preprocess` | Accelerated movie processing | `gpu` |
| `batch_preprocess` | Batch spinning-disk and motion correction | `gpu,motion,io` |

## Data and privacy

Do not commit experimental data to this repository. The ignore rules cover common microscopy,
array, archive, NWB, and notebook-output formats, but an ignore file is not a security boundary.
Inspect staged files before every public push:

```bash
git diff --cached --stat
git diff --cached
```

Use a dedicated data store and pass paths at runtime. For batch processing, set
`LUMEMBRANE_ARCHIVE` and `LUMEMBRANE_SCRATCH`; no researcher-specific path is built into the
package.

## Project status and release path

The current version is `0.1.0a0`. Before publishing, work through
[`docs/release-checklist.md`](docs/release-checklist.md), especially provenance review,
validation on shareable data, GPU installation testing, and formal name/trademark clearance.

The working name **Lumembrane** combines “lumen” and “membrane.” It is provisional until that
clearance is complete.

## Contributing

Bug reports and focused pull requests are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md)
before proposing a change.

## License

Original code in this repository is available under the [MIT License](LICENSE). Algorithms or
ports whose provenance still needs confirmation are called out in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); resolve those items before the first public
release.

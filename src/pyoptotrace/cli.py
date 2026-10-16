"""Command-line entry points for pyOptoTrace."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from . import __version__
from .spike_analysis import detect_peaks


def _synthetic_trace(seed: int, duration: float, sampling_rate: float):
    """Return a deterministic noisy trace and the injected spike indices."""
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if sampling_rate <= 0:
        raise ValueError("sampling rate must be greater than zero")

    rng = np.random.default_rng(seed)
    n_samples = int(round(duration * sampling_rate))
    trace = rng.normal(0.0, 0.20, n_samples)
    kernel = np.array([0.3, 1.4, 5.0, 1.8, 0.5])
    spike_times_s = np.arange(0.5, max(0.5, duration - 0.25), 0.75)
    injected = (spike_times_s * sampling_rate).astype(int)
    injected = injected[(injected >= 2) & (injected < n_samples - 2)]
    for index in injected:
        trace[index - 2 : index + 3] += kernel
    return trace, injected


def _run_demo(args: argparse.Namespace) -> int:
    trace, injected = _synthetic_trace(args.seed, args.duration, args.sampling_rate)
    detected = detect_peaks(
        trace,
        fs=args.sampling_rate,
        threshold_std=args.threshold,
        min_distance_ms=5.0,
    )
    summary = {
        "version": __version__,
        "samples": int(trace.size),
        "sampling_rate_hz": args.sampling_rate,
        "injected_spikes": injected.tolist(),
        "detected_spikes": detected.tolist(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyoptotrace",
        description="Tools for functional fluorescence imaging analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show package and data-shape conventions")
    info.set_defaults(
        handler=lambda _args: (
            print(
                f"pyOptoTrace {__version__}\n"
                "Movies use (time, y, x); traces use (time,); masks use (y, x)."
            )
            or 0
        )
    )

    demo = subparsers.add_parser("demo", help="run spike detection on a synthetic trace")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--duration", type=float, default=3.0, help="duration in seconds")
    demo.add_argument("--sampling-rate", type=float, default=1000.0, help="samples per second")
    demo.add_argument("--threshold", type=float, default=4.0, help="detection threshold in SD")
    demo.set_defaults(handler=_run_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pyOptoTrace command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

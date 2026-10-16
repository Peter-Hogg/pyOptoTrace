"""Lumembrane: tools for functional fluorescence imaging analysis."""

from .bleach_correction import correct_bleaching, fit_bleach_curve
from .camp_preprocess import deltaFOverF0_camp
from .spike_analysis import detect_peaks, estimate_firing_rate, spike_triggered_average

__all__ = [
    "correct_bleaching",
    "deltaFOverF0_camp",
    "detect_peaks",
    "estimate_firing_rate",
    "fit_bleach_curve",
    "spike_triggered_average",
]

__version__ = "0.1.0a0"

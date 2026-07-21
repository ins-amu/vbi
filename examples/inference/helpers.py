"""
Visualisation helpers for vbi.inference demos.

All functions live in the canonical vbi.plot / vbi.inference modules.
This file re-exports them so existing demo scripts keep working with a plain
``from helpers import pairplot`` import.
"""
from vbi.plot import posterior_1d, coverage_plot  # noqa: F401
from vbi.inference import pairplot, plot_loss      # noqa: F401

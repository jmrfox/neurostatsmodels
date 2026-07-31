"""Reusable computational neuroscience models and analysis utilities.

Library code for the toy projects in this repo lives here. Notebooks and
scripts should import from this package rather than duplicating shared logic.
"""

from .integration import (
    Integrator,
    IntegratorOptions,
    BatchSignalAnalyzer,
    NonnegativeBatchSignalAnalyzer,
    make_epoch_times,
)
from .decomposition import SpikeDeconvolver
from .adexlif import adexlif_simulation, get_step_current
from .populations import GaussianTunedPopulation
from .plotting import plot_spike_raster
from .optimization import (
    total_average_rate,
    evaluate_tuning_width,
    sweep_tuning_widths,
    optimize_tuning_width,
)

__all__ = [
    "Integrator",
    "IntegratorOptions",
    "BatchSignalAnalyzer",
    "NonnegativeBatchSignalAnalyzer",
    "make_epoch_times",
    "SpikeDeconvolver",
    "adexlif_simulation",
    "get_step_current",
    "GaussianTunedPopulation",
    "plot_spike_raster",
    "total_average_rate",
    "evaluate_tuning_width",
    "sweep_tuning_widths",
    "optimize_tuning_width",
]

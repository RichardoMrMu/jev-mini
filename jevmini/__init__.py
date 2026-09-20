"""jev-mini: a local, inspectable reimplementation of the System One Model idea."""

from jevmini.core import Choice, Decision, JevMini, Noul, Schema, Score
from jevmini.calibration import (
    CalibrationReport,
    bootstrap_ci,
    compare_ece,
    compute_calibration,
    fit_temperature,
    fit_temperature_cv,
    routing_table,
)

__version__ = "0.2.0"
__all__ = [
    "JevMini",
    "Schema",
    "Choice",
    "Noul",
    "Score",
    "Decision",
    "CalibrationReport",
    "compute_calibration",
    "fit_temperature",
    "fit_temperature_cv",
    "bootstrap_ci",
    "compare_ece",
    "routing_table",
]

"""thz_analyzer package."""

from thz_analyzer.core.correction import CorrectionModel
from thz_analyzer.core.covariance import compute_noise_covariance_matrix

__all__ = ["CorrectionModel", "compute_noise_covariance_matrix"]


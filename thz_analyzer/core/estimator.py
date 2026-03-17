from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from .covariance import (
    compute_combined_ncm,
    compute_covariance_diagnostics,
    compute_noise_covariance_matrix,
    compute_precision_matrix,
)

__all__ = ["CovarianceEstimator"]


class CovarianceEstimator:
    """Small stateful wrapper around covariance helpers."""

    def __init__(
        self,
        method: Literal["ledoit_wolf", "oas", "graphical_lasso", "empirical"] = "ledoit_wolf",
        backend: Literal["auto", "jax", "sklearn"] = "auto",
        device: Optional[Any] = None,
        max_gpu_memory_mb: float = 4096.0,
        chunk_size: int = 64,
    ):
        self.method = method
        self.backend = backend
        self.device = device
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.chunk_size = chunk_size

        self.ncm: Optional[np.ndarray] = None
        self.ncm_sample = None
        self.ncm_ref_sim = None
        self.precision: Optional[np.ndarray] = None
        self.info: Optional[dict] = None
        self.diagnostics: Optional[dict] = None

    def fit(
        self,
        corrected_traces: np.ndarray,
        reference_traces: Optional[np.ndarray] = None,
        *,
        compute_precision: bool = True,
        compute_diagnostics: bool = True,
    ) -> "CovarianceEstimator":
        """Estimate covariance data from corrected traces."""
        if reference_traces is not None:
            self.ncm, self.ncm_sample, self.ncm_ref_sim, self.info = compute_combined_ncm(
                corrected_traces,
                reference_traces,
                method=self.method,
                backend=self.backend,
                device=self.device,
                max_gpu_memory_mb=self.max_gpu_memory_mb,
                chunk_size=self.chunk_size,
            )
        else:
            self.ncm, self.info = compute_noise_covariance_matrix(
                corrected_traces,
                method=self.method,
                backend=self.backend,
                device=self.device,
                max_gpu_memory_mb=self.max_gpu_memory_mb,
                chunk_size=self.chunk_size,
            )
            self.ncm_sample = None
            self.ncm_ref_sim = None

        self.precision = None
        self.diagnostics = None

        if compute_precision:
            backend_used = str((self.info or {}).get("backend", "")).lower()
            if backend_used == "jax":
                try:
                    self.precision = compute_precision_matrix(
                        self.ncm,
                        backend="jax",
                        device=self.device,
                    )
                    if self.info is not None:
                        self.info["precision_backend"] = "jax"
                except Exception as exc:
                    self.precision = compute_precision_matrix(self.ncm)
                    if self.info is not None:
                        self.info["precision_backend"] = "numpy"
                        self.info["precision_fallback_reason"] = str(exc)
            else:
                self.precision = compute_precision_matrix(self.ncm)
                if self.info is not None:
                    self.info["precision_backend"] = "numpy"

        if compute_diagnostics and self.precision is not None:
            self.diagnostics = compute_covariance_diagnostics(
                self.ncm,
                self.precision,
            )

        return self

    def export(self, path: str, metadata: Dict[str, Any]) -> None:
        """Save the fitted matrices to an NPZ file."""
        if self.ncm is None or self.precision is None:
            raise RuntimeError("Must call fit() before export().")

        out_path = Path(path)
        if out_path.suffix.lower() != ".npz":
            out_path = out_path.with_suffix(".npz")

        payload = dict(metadata or {})
        if self.info is not None:
            payload["info"] = self.info
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics

        save_kw = {
            "ncm": np.asarray(self.ncm, dtype=np.float64),
            "precision": np.asarray(self.precision, dtype=np.float64),
            "metadata": np.array(payload, dtype=object),
        }
        if self.ncm_sample is not None:
            save_kw["ncm_sample"] = np.asarray(self.ncm_sample, dtype=np.float64)
        if self.ncm_ref_sim is not None:
            save_kw["ncm_ref_sim"] = np.asarray(self.ncm_ref_sim, dtype=np.float64)
        np.savez_compressed(out_path, **save_kw)

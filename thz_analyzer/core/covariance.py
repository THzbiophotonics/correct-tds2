"""Noise covariance matrix (NCM) estimation for THz traces.

Public API: compute_noise_covariance_matrix, compute_precision_matrix,
compute_covariance_diagnostics, compute_combined_ncm.

backend="auto" tries JAX first (GPU if available), then falls back to sklearn.
GPU memory is managed automatically — float32, chunked, and CPU fallbacks as needed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import validate_2d_array, validate_square_matrix

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

try:
    from .jax_covariance import (
        empirical_covariance_jax,
        ledoit_wolf_jax,
        oas_jax,
        precision_matrix_jax,
        _ledoit_wolf_from_cov_jax,
        _oas_from_cov_jax,
    )
except Exception:
    empirical_covariance_jax = None
    ledoit_wolf_jax = None
    oas_jax = None
    precision_matrix_jax = None
    _ledoit_wolf_from_cov_jax = None
    _oas_from_cov_jax = None

__all__ = [
    "compute_noise_covariance_matrix",
    "compute_precision_matrix",
    "compute_covariance_diagnostics",
    "compute_combined_ncm",
]


# --- Small private helpers ---

def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (matrix + matrix.T)


def _empirical_covariance(traces: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = traces - np.mean(traces, axis=0, keepdims=True)
    return centered.T @ centered / float(traces.shape[0])


def _resolve_backend(backend: str) -> str:
    backend_key = (backend or "auto").strip().lower()
    if backend_key not in {"auto", "jax", "sklearn", "numpy"}:
        raise ValueError("backend must be one of {'auto','jax','sklearn','numpy'}.")
    return backend_key


def _jax_ready() -> bool:
    return all(
        x is not None
        for x in (jax, jnp, empirical_covariance_jax, ledoit_wolf_jax,
                   oas_jax, precision_matrix_jax, _ledoit_wolf_from_cov_jax, _oas_from_cov_jax)
    )


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg or "CUDA_ERROR_OUT_OF_MEMORY" in msg


def _eigenvalues_symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _symmetrize(np.asarray(matrix, dtype=np.float64))
    try:
        return np.asarray(jnp.linalg.eigvalsh(jnp.asarray(matrix)), dtype=np.float64)
    except Exception:
        return np.linalg.eigvalsh(matrix)


# --- JAX covariance helpers (module-level, testable in isolation) ---

def _get_free_gpu_memory_mb() -> float | None:
    """Return free GPU memory in MB, or None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                try:
                    return float(line.strip())
                except ValueError:
                    continue
    except Exception:
        pass
    if jax is not None:
        try:
            gpus = [d for d in jax.devices() if d.platform == "gpu"]
            if gpus:
                stats = gpus[0].memory_stats()
                limit = float(stats.get("bytes_limit") or 0)
                in_use = float(stats.get("bytes_in_use") or 0)
                if limit > 0:
                    return max(0.0, limit - in_use) / 1e6
        except Exception:
            pass
    return None


def _estimate_covariance_memory_mb(n_traces: int, n_samples: int, use_f64: bool = True) -> float:
    """Rough upper-bound on GPU memory needed to compute the covariance matrix."""
    bytes_per_element = 8 if use_f64 else 4
    return 2.5 * (n_traces * n_samples + n_samples * n_samples) * bytes_per_element / 1e6


def _jax_gram_chunked(residuals_jax, chunk_size: int = 64):
    """Accumulate R^T R and sum_k ||x_k||^4 in chunks to avoid one large allocation."""
    n_samples = residuals_jax.shape[1]
    acc = jnp.zeros((n_samples, n_samples), dtype=residuals_jax.dtype)
    norm4 = jnp.zeros((), dtype=residuals_jax.dtype)
    for start in range(0, residuals_jax.shape[0], chunk_size):
        chunk = residuals_jax[start: start + chunk_size]
        acc = acc + chunk.T @ chunk
        norm4 = norm4 + jnp.sum(jnp.sum(chunk ** 2, axis=1) ** 2)
    return acc, norm4


def _compute_gram_on_device(
    residuals_np: NDArray,
    device: jax.Device,
    is_gpu: bool,
    dtype: Any,
    chunk_size: int,
) -> jax.Array:
    """Upload residuals and compute R^T R on the target device; halves chunk size on OOM."""
    np_dtype = np.float32 if dtype == jnp.float32 else np.float64
    chunk_size_used = max(1, int(chunk_size))

    if is_gpu:
        # Upload chunk-by-chunk to avoid one large host→device transfer.
        while True:
            try:
                n_samples = residuals_np.shape[1]
                gram = jnp.zeros((n_samples, n_samples), dtype=dtype)
                norm4 = jnp.zeros((), dtype=dtype)
                for start in range(0, residuals_np.shape[0], chunk_size_used):
                    chunk = jax.device_put(
                        np.asarray(residuals_np[start: start + chunk_size_used], dtype=np_dtype),
                        device,
                    )
                    gram = gram + chunk.T @ chunk
                    norm4 = norm4 + jnp.sum(jnp.sum(chunk ** 2, axis=1) ** 2)
                return gram, norm4
            except Exception as exc:
                if _is_oom_error(exc) and chunk_size_used > 1:
                    chunk_size_used //= 2
                    continue
                raise
    else:
        residuals_jax = jax.device_put(np.asarray(residuals_np, dtype=np_dtype), device)
        return _jax_gram_chunked(residuals_jax, chunk_size=chunk_size_used)


def _apply_shrinkage_to_gram(
    gram: jax.Array,
    n_traces: int,
    method: str,
    norm4: jax.Array,
) -> tuple[jax.Array, dict]:
    """Divide the gram matrix by n and apply the requested shrinkage estimator."""
    sample_cov = gram / float(n_traces)
    if method == "ledoit_wolf":
        cov, shrinkage = _ledoit_wolf_from_cov_jax(sample_cov, n_traces, norm4)
        return cov, {"shrinkage": float(shrinkage), "chunked": True}
    if method == "oas":
        cov, shrinkage = _oas_from_cov_jax(sample_cov, n_traces)
        return cov, {"shrinkage": float(shrinkage), "chunked": True}
    if method == "empirical":
        return sample_cov, {"ddof": 0, "chunked": True}
    raise ValueError(f"JAX does not support method={method!r}.")


def _compute_cov_direct(residuals_jax: jax.Array, method: str) -> tuple[jax.Array, dict]:
    """Compute covariance directly (no pre-built gram matrix)."""
    if method == "ledoit_wolf":
        cov, shrinkage = ledoit_wolf_jax(residuals_jax)
        return cov, {"shrinkage": float(shrinkage)}
    if method == "oas":
        cov, shrinkage = oas_jax(residuals_jax)
        return cov, {"shrinkage": float(shrinkage)}
    if method == "empirical":
        return empirical_covariance_jax(residuals_jax), {"ddof": 0}
    raise ValueError(f"JAX does not support method={method!r}.")


def _jax_covariance(
    traces: NDArray,
    *,
    method: str,
    device,
    max_gpu_memory_mb: float = 4096.0,
    chunk_size: int = 64,
) -> tuple[NDArray[np.float64], dict]:
    """JAX covariance with automatic CPU fallback on GPU OOM.

    Tries float64 direct → float32 chunked → CPU, in that order.
    """
    if not _jax_ready():
        raise RuntimeError("JAX backend unavailable.")

    n_traces, n_samples = traces.shape
    residuals = traces - np.mean(traces, axis=0, keepdims=True)

    if device is None:
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        device = gpus[0] if gpus else jax.devices("cpu")[0]

    is_gpu = getattr(device, "platform", "cpu") == "gpu"
    info: dict[str, Any] = {
        "backend": "jax",
        "device": str(device),
        "device_type": getattr(device, "platform", "cpu"),
    }

    # pick float32 or float64 based on available GPU memory
    use_chunked = False
    use_f32 = False
    if is_gpu:
        mem_f64 = _estimate_covariance_memory_mb(n_traces, n_samples, use_f64=True)
        mem_f32 = _estimate_covariance_memory_mb(n_traces, n_samples, use_f64=False)
        free_mb = _get_free_gpu_memory_mb()
        budget = min(float(max_gpu_memory_mb), free_mb) if free_mb else float(max_gpu_memory_mb)

        info["estimated_memory_f64_mb"] = round(mem_f64, 1)
        if free_mb:
            info["free_gpu_memory_mb"] = round(free_mb, 1)
        info["gpu_budget_mb"] = round(budget, 1)

        if mem_f64 > budget:
            use_f32 = True
            use_chunked = True
            info["memory_strategy"] = "float32_chunked_gpu" if mem_f32 <= budget else "float32_chunked_gpu_forced"
            if mem_f32 > budget:
                info["estimated_memory_f32_mb"] = round(mem_f32, 1)
        else:
            info["memory_strategy"] = "float64_direct_gpu"
    else:
        info["memory_strategy"] = "cpu_direct"

    jax_dtype = jnp.float32 if use_f32 else jnp.float64

    # compute — with automatic CPU retry on OOM
    t0 = time.perf_counter()
    try:
        if use_chunked:
            gram, norm4 = _compute_gram_on_device(residuals, device, is_gpu, jax_dtype, chunk_size)
            cov_jax, details = _apply_shrinkage_to_gram(gram, n_traces, method, norm4)
        else:
            try:
                residuals_jax = jax.device_put(
                    np.asarray(residuals, dtype=np.float32 if use_f32 else np.float64), device
                )
                cov_jax, details = _compute_cov_direct(residuals_jax, method)
            except Exception as exc:
                if is_gpu and _is_oom_error(exc):
                    # direct OOM → retry with float32 chunked
                    info["direct_gpu_oom"] = str(exc)
                    info["memory_strategy"] = "float32_chunked_gpu_retry"
                    gram, norm4 = _compute_gram_on_device(residuals, device, is_gpu, jnp.float32, chunk_size)
                    cov_jax, details = _apply_shrinkage_to_gram(gram, n_traces, method, norm4)
                else:
                    raise

        ncm = _symmetrize(np.asarray(cov_jax, dtype=np.float64))

    except Exception as exc:
        if is_gpu and _is_oom_error(exc):
            # GPU OOM even chunked → fall back to CPU
            info["gpu_oom"] = str(exc)
            device = jax.devices("cpu")[0]
            info.update({"device": str(device), "device_type": "cpu",
                          "memory_strategy": "cpu_fallback_after_gpu_oom"})
            residuals_jax = jax.device_put(np.asarray(residuals, dtype=np.float64), device)
            cov_jax, details = _compute_cov_direct(residuals_jax, method)
            ncm = _symmetrize(np.asarray(cov_jax, dtype=np.float64))
        else:
            raise

    info["time"] = float(time.perf_counter() - t0)
    info.update(details)
    return ncm, info


# --- sklearn covariance backend ---

def _sklearn_covariance(
    traces: NDArray[np.float64],
    *,
    method: str,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    residuals = traces - np.mean(traces, axis=0, keepdims=True)
    t0 = time.perf_counter()
    info: dict[str, Any] = {"backend": "sklearn", "device": "cpu", "device_type": "cpu"}

    if method == "empirical":
        covariance = _empirical_covariance(traces)
        info["ddof"] = 0

    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        model = LedoitWolf().fit(residuals)
        covariance = np.asarray(model.covariance_, dtype=np.float64)
        info["shrinkage"] = float(model.shrinkage_)

    elif method == "oas":
        from sklearn.covariance import OAS
        model = OAS().fit(residuals)
        covariance = np.asarray(model.covariance_, dtype=np.float64)
        info["shrinkage"] = float(model.shrinkage_)

    elif method == "graphical_lasso":
        from sklearn.covariance import GraphicalLassoCV
        max_samples = int(kwargs.get("max_samples_gl", 2000))
        fit_residuals = residuals
        if fit_residuals.shape[0] > max_samples:
            idx = np.random.default_rng(42).choice(fit_residuals.shape[0], size=max_samples, replace=False)
            fit_residuals = fit_residuals[idx]
            info["downsampled_traces"] = int(fit_residuals.shape[0])
            info["original_traces"] = int(residuals.shape[0])
        cv = int(kwargs.get("cv", 3))
        n_alphas = int(kwargs.get("n_alphas", 2))
        n_refinements = int(kwargs.get("n_refinements", 10))
        max_iter = int(kwargs.get("max_iter", 100))
        tol = float(kwargs.get("tol", 1e-4))
        fitted = GraphicalLassoCV(
            cv=cv, alphas=n_alphas, n_refinements=n_refinements,
            max_iter=max_iter, mode="cd", n_jobs=-1, tol=tol, verbose=False,
        ).fit(fit_residuals)
        covariance = np.asarray(fitted.covariance_, dtype=np.float64)
        info.update({
            "alpha": float(fitted.alpha_), "cv_folds": cv,
            "n_alphas": n_alphas, "n_refinements": n_refinements,
            "max_iter": max_iter, "tol": tol,
        })
        n_iter = getattr(fitted, "n_iter_", None)
        if n_iter is not None:
            arr = np.asarray(n_iter)
            if arr.size > 0:
                info["n_iter"] = int(np.max(arr))
    else:
        raise ValueError(f"Unsupported method={method!r}.")

    info["time"] = float(time.perf_counter() - t0)
    return _symmetrize(covariance), info


# --- Public API ---

def compute_noise_covariance_matrix(
    corrected_traces: ArrayLike,
    method: str = "ledoit_wolf",
    backend: str = "auto",
    device: Any | None = None,
    max_gpu_memory_mb: float = 4096.0,
    chunk_size: int = 64,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Estimate the noise covariance matrix (NCM) from a batch of corrected traces.

    Parameters
    ----------
    corrected_traces:
        Array of shape ``(n_traces, n_samples)``.
    method:
        Estimator: ``"ledoit_wolf"`` (default), ``"oas"``, ``"empirical"``,
        or ``"graphical_lasso"`` (sklearn only).
    backend:
        ``"auto"`` tries JAX then sklearn; ``"jax"`` or ``"sklearn"`` to force one.
    device:
        JAX device (``jax.devices("gpu")[0]`` etc.).  ``None`` auto-selects GPU.
    max_gpu_memory_mb:
        Soft cap on GPU memory usage before switching to chunked / float32 mode.
    chunk_size:
        Number of traces per chunk for the GPU chunked path.

    Returns
    -------
    ncm:
        Symmetric NCM of shape ``(n_samples, n_samples)``, dtype float64.
    info:
        Dictionary with backend, timing, shrinkage, and memory details.
    """
    traces = validate_2d_array(corrected_traces, name="corrected_traces")
    n_traces, n_samples = traces.shape
    if n_traces < 2:
        raise ValueError("corrected_traces must contain at least 2 traces.")

    method_key = (method or "ledoit_wolf").strip().lower()
    if method_key not in {"empirical", "ledoit_wolf", "oas", "graphical_lasso"}:
        raise ValueError("method must be one of {'empirical','ledoit_wolf','oas','graphical_lasso'}.")

    backend_key = _resolve_backend(backend)
    # Legacy kwarg support.
    legacy_use_jax = kwargs.pop("use_jax", None)
    if legacy_use_jax is not None:
        backend_key = "jax" if bool(legacy_use_jax) else "sklearn"

    info: dict[str, Any] = {
        "method": method_key,
        "backend": backend_key,
        "n_traces": int(n_traces),
        "n_samples": int(n_samples),
        "under_determined": bool(n_traces < n_samples),
    }

    # graphical_lasso is sklearn-only.
    if method_key == "graphical_lasso" and backend_key == "jax":
        backend_key = "sklearn"
        info["fallback_reason"] = "graphical_lasso is sklearn-only."

    # --- Try JAX ---
    if backend_key in {"auto", "jax"} and method_key in {"empirical", "ledoit_wolf", "oas"}:
        try:
            ncm, jax_info = _jax_covariance(
                traces, method=method_key, device=device,
                max_gpu_memory_mb=max_gpu_memory_mb, chunk_size=chunk_size,
            )
            info.update(jax_info)
            return _symmetrize(ncm), info
        except Exception as exc:
            if backend_key == "jax":
                raise RuntimeError(f"JAX backend failed for method={method_key}: {exc}") from exc
            info["jax_fallback_reason"] = str(exc)

    # --- Try sklearn ---
    try:
        ncm, sk_info = _sklearn_covariance(traces, method=method_key, **kwargs)
        info.update(sk_info)
    except Exception as exc:
        # Last resort: plain empirical covariance on CPU.
        ncm = _symmetrize(_empirical_covariance(traces))
        info.update({"method": "empirical", "backend": "numpy",
                      "fallback_reason": str(exc), "ddof": 0})

    if ncm.shape != (n_samples, n_samples):
        raise RuntimeError(f"Invalid covariance shape: {ncm.shape}.")
    return _symmetrize(np.asarray(ncm, dtype=np.float64)), info


def compute_precision_matrix(
    ncm: ArrayLike,
    method: str = "direct",
    backend: str = "auto",
    device: Any | None = None,
    regularization: float = 1e-10,
) -> NDArray[np.float64]:
    """Invert the NCM to get the precision matrix.

    Tries JAX Cholesky first, falls back to NumPy. Switches to pinv if ill-conditioned.
    """
    matrix = _symmetrize(validate_square_matrix(ncm, name="ncm"))
    n = matrix.shape[0]
    method_key = (method or "direct").strip().lower()
    backend_key = _resolve_backend(backend)
    if method_key not in {"direct", "pinv", "cholesky"}:
        raise ValueError("method must be one of {'direct','pinv','cholesky'}.")

    # --- JAX path ---
    if backend_key in {"auto", "jax"} and method_key in {"direct", "cholesky"} and _jax_ready():
        try:
            cov_jax = jnp.asarray(matrix, dtype=jnp.float64)
            if device is not None:
                cov_jax = jax.device_put(cov_jax, device)
            precision = precision_matrix_jax(cov_jax, regularization=max(0.0, regularization))
            precision_np = _symmetrize(np.asarray(precision, dtype=np.float64))
            if np.all(np.isfinite(precision_np)):
                return precision_np
        except Exception:
            if backend_key == "jax":
                raise

    # --- NumPy path ---
    matrix_reg = matrix + max(0.0, regularization) * np.eye(n, dtype=np.float64)
    try:
        cond = float(np.linalg.cond(matrix_reg))
    except np.linalg.LinAlgError:
        cond = np.inf

    try:
        if method_key == "pinv":
            precision_np = np.linalg.pinv(matrix_reg)
        elif method_key == "cholesky":
            chol = np.linalg.cholesky(matrix_reg)
            ident = np.eye(n, dtype=np.float64)
            precision_np = np.linalg.solve(chol.T, np.linalg.solve(chol, ident))
        else:
            precision_np = np.linalg.inv(matrix_reg)
    except Exception:
        precision_np = np.linalg.pinv(matrix_reg)

    precision_np = _symmetrize(np.asarray(precision_np, dtype=np.float64))
    if not np.all(np.isfinite(precision_np)):
        precision_np = _symmetrize(np.linalg.pinv(matrix_reg))

    # Fall back to pinv for ill-conditioned matrices.
    residual = np.linalg.norm(matrix_reg @ precision_np - np.eye(n), ord="fro") / max(n, 1)
    if not np.isfinite(cond) or cond > 1e12 or residual > 1e-5:
        precision_np = _symmetrize(np.linalg.pinv(matrix_reg))

    return precision_np


def compute_covariance_diagnostics(
    ncm: ArrayLike,
    precision_matrix: ArrayLike,
) -> dict[str, dict[str, Any]]:
    """Return eigenvalues, rank, condition number and sparsity for both matrices.

    Useful for checking whether the NCM is well-conditioned before using the
    precision matrix for inference.
    """
    ncm_arr = _symmetrize(validate_square_matrix(ncm, name="ncm"))
    precision_arr = _symmetrize(validate_square_matrix(precision_matrix, name="precision_matrix"))
    if ncm_arr.shape != precision_arr.shape:
        raise ValueError(
            f"ncm and precision_matrix must have the same shape, "
            f"got {ncm_arr.shape} and {precision_arr.shape}."
        )

    ncm_eigs = _eigenvalues_symmetric(ncm_arr)
    prec_eigs = _eigenvalues_symmetric(precision_arr)

    def _sparsity(arr):
        abs_arr = np.abs(arr)
        threshold = 1e-10 * max(1.0, float(np.max(abs_arr)))
        return float(np.mean(abs_arr <= threshold))

    return {
        "ncm": {
            "condition_number": float(np.linalg.cond(ncm_arr)),
            "eigenvalues": ncm_eigs,
            "rank": int(np.linalg.matrix_rank(ncm_arr)),
            "trace": float(np.trace(ncm_arr)),
            "frobenius_norm": float(np.linalg.norm(ncm_arr, ord="fro")),
            "sparsity": _sparsity(ncm_arr),
        },
        "precision": {
            "condition_number": float(np.linalg.cond(precision_arr)),
            "eigenvalues": prec_eigs,
            "rank": int(np.linalg.matrix_rank(precision_arr)),
            "sparsity": _sparsity(precision_arr),
        },
    }


def compute_combined_ncm(
    corrected_traces: ArrayLike,
    reference_traces: ArrayLike,
    method: str = "ledoit_wolf",
    backend: str = "auto",
    device: Any | None = None,
    max_gpu_memory_mb: float = 4096.0,
    chunk_size: int = 64,
    transfer_regularization: float = 1e-6,
) -> tuple[NDArray, NDArray, NDArray, dict]:
    """Combined NCM accounting for both sample and reference noise.

    Estimates h(t) between mean traces, simulates reference noise through it,
    then sums both contributions. Returns (ncm_total, ncm_sample, ncm_ref_simulated, info).
    """
    corr = validate_2d_array(corrected_traces, "corrected_traces")
    ref = validate_2d_array(reference_traces, "reference_traces")
    if corr.shape[1] != ref.shape[1]:
        raise ValueError(
            f"corrected_traces and reference_traces must share n_samples, "
            f"got {corr.shape[1]} vs {ref.shape[1]}."
        )
    n_samples = corr.shape[1]
    mean_corr = np.mean(corr, axis=0)
    mean_ref = np.mean(ref, axis=0)

    # Estimate h(t) and simulate reference traces through it.
    simulated_np, jax_fail_reason = _simulate_reference_traces(
        mean_corr, mean_ref, ref, n_samples, backend, device, transfer_regularization
    )

    common_kw = dict(
        method=method, backend=backend, device=device,
        max_gpu_memory_mb=max_gpu_memory_mb, chunk_size=chunk_size,
    )
    ncm_sample, info_sample = compute_noise_covariance_matrix(corr, **common_kw)
    ncm_ref_sim, info_ref = compute_noise_covariance_matrix(simulated_np, **common_kw)
    ncm_total = _symmetrize(ncm_sample + ncm_ref_sim)

    info = {
        "combined": True,
        "transfer_regularization": transfer_regularization,
        "sample": info_sample,
        "reference": info_ref,
    }
    if jax_fail_reason is not None:
        info["jax_transfer_failed"] = jax_fail_reason

    return ncm_total, ncm_sample, ncm_ref_sim, info


def _simulate_reference_traces(
    mean_corr: NDArray,
    mean_ref: NDArray,
    ref: NDArray,
    n_samples: int,
    backend: str,
    device,
    transfer_regularization: float,
) -> tuple[NDArray, str | None]:
    """Simulate reference traces through h(t). Returns (traces, jax_fail_reason or None)."""
    if _jax_ready() and backend in {"auto", "jax"}:
        try:
            from .jax_covariance import transfer_impulse_response_jax, simulate_ref_traces_jax
            if device is None:
                gpus = [d for d in jax.devices() if d.platform == "gpu"]
                device = gpus[0] if gpus else jax.devices("cpu")[0]
            mc = jax.device_put(jnp.asarray(mean_corr, dtype=jnp.float64), device)
            mr = jax.device_put(jnp.asarray(mean_ref, dtype=jnp.float64), device)
            rj = jax.device_put(jnp.asarray(ref, dtype=jnp.float64), device)
            h = transfer_impulse_response_jax(mc, mr, transfer_regularization)
            return np.asarray(simulate_ref_traces_jax(h, rj), dtype=np.float64), None
        except Exception as exc:
            jax_fail_reason = str(exc)  # fall through to NumPy path
    else:
        jax_fail_reason = None

    # NumPy fallback: same math, no GPU.
    S_c = np.fft.rfft(mean_corr)
    S_r = np.fft.rfft(mean_ref)
    reg = transfer_regularization * np.max(np.abs(S_r))
    h_np = np.fft.irfft(S_c / (S_r + reg), n=n_samples)
    simulated_np = np.array(
        [np.convolve(h_np, ref[i])[:n_samples] for i in range(ref.shape[0])],
        dtype=np.float64,
    )
    return simulated_np, jax_fail_reason

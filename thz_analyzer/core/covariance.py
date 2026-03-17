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
    from numba import jit
except Exception:
    jit = None

try:
    from .jax_ops import (
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


if jit is not None:

    @jit(nopython=True, cache=True)
    def _inv_nla_jit(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.linalg.inv(matrix)

else:

    def _inv_nla_jit(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.linalg.inv(matrix)


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
        for x in (
            jax,
            jnp,
            empirical_covariance_jax,
            ledoit_wolf_jax,
            oas_jax,
            precision_matrix_jax,
            _ledoit_wolf_from_cov_jax,
            _oas_from_cov_jax,
        )
    )


def _eigenvalues_symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = _symmetrize(np.asarray(matrix, dtype=np.float64))
    if jnp is not None:
        try:
            return np.asarray(jnp.linalg.eigvalsh(jnp.asarray(matrix)), dtype=np.float64)
        except Exception:
            return np.linalg.eigvalsh(matrix)
    return np.linalg.eigvalsh(matrix)


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in msg
        or "Out of memory" in msg
        or "CUDA_ERROR_OUT_OF_MEMORY" in msg
    )


def _get_free_gpu_memory_mb() -> float | None:
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                value = line.strip()
                if not value:
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
    except Exception:
        pass
    if jax is not None:
        try:
            gpus = [d for d in jax.devices() if d.platform == "gpu"]
            if gpus:
                stats = gpus[0].memory_stats()
                limit = stats.get("bytes_limit")
                in_use = stats.get("bytes_in_use")
                if limit is None or in_use is None:
                    return None
                limit_f = float(limit)
                in_use_f = float(in_use)
                if limit_f <= 0.0:
                    return None
                free = max(0.0, limit_f - in_use_f)
                return free / 1e6
        except Exception:
            pass
    return None


def _estimate_covariance_memory_mb(n_traces: int, n_samples: int, use_f64: bool = True) -> float:
    """Estimate the GPU memory needed for covariance work."""
    bpe = 8 if use_f64 else 4
    return 2.5 * (n_traces * n_samples + n_samples * n_samples) * bpe / 1e6


def _jax_gram_chunked(residuals_jax, chunk_size: int = 64):
    n_traces = residuals_jax.shape[0]
    acc = jnp.zeros(
        (residuals_jax.shape[1], residuals_jax.shape[1]),
        dtype=residuals_jax.dtype,
    )
    for start in range(0, n_traces, chunk_size):
        chunk = residuals_jax[start : start + chunk_size]
        acc = acc + chunk.T @ chunk
    return acc


def _jax_covariance(
    traces,
    *,
    method,
    device,
    max_gpu_memory_mb=4096.0,
    chunk_size=64,
):
    if not _jax_ready():
        raise RuntimeError("JAX backend unavailable.")

    n_traces, n_samples = traces.shape
    residuals = traces - np.mean(traces, axis=0, keepdims=True)

    if device is None:
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        target_device = gpus[0] if gpus else jax.devices("cpu")[0]
    else:
        target_device = device

    is_gpu = getattr(target_device, "platform", "cpu") == "gpu"
    info = {
        "backend": "jax",
        "device": str(target_device),
        "device_type": getattr(target_device, "platform", "cpu"),
    }

    def _compute_direct_cov(residuals_arr):
        if method == "ledoit_wolf":
            cov_local, shrink_local = ledoit_wolf_jax(residuals_arr)  # type: ignore[misc]
            return cov_local, {"shrinkage": float(shrink_local)}
        if method == "oas":
            cov_local, shrink_local = oas_jax(residuals_arr)  # type: ignore[misc]
            return cov_local, {"shrinkage": float(shrink_local)}
        if method == "empirical":
            cov_local = empirical_covariance_jax(residuals_arr)  # type: ignore[misc]
            return cov_local, {"ddof": 0}
        raise ValueError(f"JAX does not support method={method!r}.")

    def _compute_chunked_cov(dtype, initial_chunk_size: int):
        n = float(n_traces)
        chunk_size_used = max(1, int(initial_chunk_size))
        np_dtype = np.float32 if dtype == jnp.float32 else np.float64

        while True:
            try:
                if is_gpu:
                    gram = jnp.zeros((n_samples, n_samples), dtype=dtype)
                    for start in range(0, n_traces, chunk_size_used):
                        chunk_np = np.asarray(residuals[start : start + chunk_size_used], dtype=np_dtype)
                        chunk_dev = jax.device_put(chunk_np, target_device)
                        gram = gram + chunk_dev.T @ chunk_dev
                else:
                    residuals_arr = _put_residuals(dtype, target_device)
                    gram = _jax_gram_chunked(residuals_arr, chunk_size=chunk_size_used)
                break
            except Exception as exc:
                if is_gpu and _is_oom_error(exc) and chunk_size_used > 1:
                    chunk_size_used = max(1, chunk_size_used // 2)
                    continue
                raise

        s = gram / n

        if method == "empirical":
            cov_local = s
            details_local = {"ddof": 0, "chunked": True}
        elif method == "ledoit_wolf":
            cov_local, shrinkage = _ledoit_wolf_from_cov_jax(s, n)  # type: ignore[misc]
            details_local = {"shrinkage": float(shrinkage), "chunked": True}
        elif method == "oas":
            cov_local, shrinkage = _oas_from_cov_jax(s, n)  # type: ignore[misc]
            details_local = {"shrinkage": float(shrinkage), "chunked": True}
        else:
            raise ValueError(f"JAX chunked mode does not support method={method!r}.")

        details_local["chunk_size_used"] = int(chunk_size_used)
        return cov_local, details_local

    def _put_residuals(dtype, dst_device):
        np_dtype = np.float32 if dtype == jnp.float32 else np.float64
        return jax.device_put(np.asarray(residuals, dtype=np_dtype), dst_device)

    use_chunked = False
    use_f32 = False

    if is_gpu:
        mem_f64 = _estimate_covariance_memory_mb(n_traces, n_samples, True)
        mem_f32 = _estimate_covariance_memory_mb(n_traces, n_samples, False)
        free_mb = _get_free_gpu_memory_mb()
        budget = float(max_gpu_memory_mb)
        if free_mb is not None and free_mb > 0.0:
            budget = min(budget, free_mb)

        info["estimated_memory_f64_mb"] = round(mem_f64, 1)
        if free_mb is not None and free_mb > 0.0:
            info["free_gpu_memory_mb"] = round(free_mb, 1)
        info["gpu_budget_mb"] = round(budget, 1)

        if mem_f64 > budget:
            use_f32 = True
            use_chunked = True
            if mem_f32 <= budget:
                info["memory_strategy"] = "float32_chunked_gpu"
            else:
                info["estimated_memory_f32_mb"] = round(mem_f32, 1)
                info["memory_strategy"] = "float32_chunked_gpu_forced"
        else:
            info["memory_strategy"] = "float64_direct_gpu"
    else:
        info["memory_strategy"] = "cpu_direct"

    jax_dtype = jnp.float32 if use_f32 else jnp.float64
    residuals_jax = None
    if not use_chunked:
        try:
            residuals_jax = _put_residuals(jax_dtype, target_device)
        except Exception as exc:
            if is_gpu and _is_oom_error(exc):
                info["residual_allocation_oom"] = str(exc)
                jax_dtype = jnp.float32
                use_chunked = True
                info["memory_strategy"] = "float32_chunked_gpu_retry_on_allocation"
            else:
                raise

    t0 = time.perf_counter()
    try:
        if use_chunked:
            cov_jax, details = _compute_chunked_cov(jax_dtype, chunk_size)
        else:
            try:
                cov_jax, details = _compute_direct_cov(residuals_jax)
            except Exception as exc:
                if is_gpu and _is_oom_error(exc):
                    info["direct_gpu_oom"] = str(exc)
                    info["memory_strategy"] = "float32_chunked_gpu_retry"
                    jax_dtype = jnp.float32
                    cov_jax, details = _compute_chunked_cov(jax_dtype, chunk_size)
                else:
                    raise
    except Exception as exc:
        if is_gpu and _is_oom_error(exc):
            info["gpu_oom"] = str(exc)
            target_device = jax.devices("cpu")[0]
            info["device"] = str(target_device)
            info["device_type"] = "cpu"
            info["memory_strategy"] = "cpu_fallback_after_gpu_oom"
            residuals_jax = _put_residuals(jnp.float64, target_device)
            cov_jax, details = _compute_direct_cov(residuals_jax)
        else:
            raise
    try:
        ncm = _symmetrize(np.asarray(cov_jax, dtype=np.float64))
    except Exception as exc:
        if is_gpu and _is_oom_error(exc):
            info["materialization_gpu_oom"] = str(exc)
            target_device = jax.devices("cpu")[0]
            info["device"] = str(target_device)
            info["device_type"] = "cpu"
            info["memory_strategy"] = "cpu_fallback_after_materialization_oom"
            residuals_jax = _put_residuals(jnp.float64, target_device)
            cov_jax, details = _compute_direct_cov(residuals_jax)
            ncm = _symmetrize(np.asarray(cov_jax, dtype=np.float64))
        else:
            raise

    info["time"] = float(time.perf_counter() - t0)
    info.update(details)
    return ncm, info


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

        max_samples_gl = int(kwargs.get("max_samples_gl", 2000))
        fit_residuals = residuals
        if fit_residuals.shape[0] > max_samples_gl:
            rng = np.random.default_rng(42)
            idx = rng.choice(fit_residuals.shape[0], size=max_samples_gl, replace=False)
            fit_residuals = fit_residuals[idx]
            info["downsampled_traces"] = int(fit_residuals.shape[0])
            info["original_traces"] = int(residuals.shape[0])

        cv = int(kwargs.get("cv", 3))
        n_alphas = int(kwargs.get("n_alphas", 2))
        n_refinements = int(kwargs.get("n_refinements", 10))
        max_iter = int(kwargs.get("max_iter", 100))
        tol = float(kwargs.get("tol", 1e-4))
        model = GraphicalLassoCV(
            cv=cv,
            alphas=n_alphas,
            n_refinements=n_refinements,
            max_iter=max_iter,
            mode="cd",
            n_jobs=-1,
            tol=tol,
            verbose=False,
        )
        fitted = model.fit(fit_residuals)
        covariance = np.asarray(fitted.covariance_, dtype=np.float64)
        info.update(
            {
                "alpha": float(fitted.alpha_),
                "cv_folds": int(cv),
                "n_alphas": int(n_alphas),
                "n_refinements": int(n_refinements),
                "max_iter": int(max_iter),
                "tol": float(tol),
            }
        )
        n_iter = getattr(fitted, "n_iter_", None)
        if n_iter is not None:
            n_iter_arr = np.asarray(n_iter)
            if n_iter_arr.size > 0:
                info["n_iter"] = int(np.max(n_iter_arr))
    else:
        raise ValueError(f"Unsupported method={method!r}.")

    info["time"] = float(time.perf_counter() - t0)
    return _symmetrize(covariance), info


def compute_noise_covariance_matrix(
    corrected_traces: ArrayLike,
    method: str = "ledoit_wolf",
    backend: str = "auto",
    device: Any | None = None,
    max_gpu_memory_mb: float = 4096.0,
    chunk_size: int = 64,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    traces = validate_2d_array(corrected_traces, name="corrected_traces")
    n_traces, n_samples = traces.shape
    if n_traces < 2:
        raise ValueError("corrected_traces must contain at least 2 traces.")

    method_key = (method or "ledoit_wolf").strip().lower()
    if method_key not in {"empirical", "ledoit_wolf", "oas", "graphical_lasso"}:
        raise ValueError("method must be one of {'empirical','ledoit_wolf','oas','graphical_lasso'}.")

    backend_key = _resolve_backend(backend)
    legacy_use_jax = kwargs.pop("use_jax", None)
    if legacy_use_jax is not None:
        backend_key = "jax" if bool(legacy_use_jax) else "sklearn"

    info: dict[str, Any] = {
        "method": method_key,
        "requested_method": method_key,
        "backend": backend_key,
        "requested_backend": backend_key,
        "n_traces": int(n_traces),
        "n_samples": int(n_samples),
        "under_determined": bool(n_traces < n_samples),
        "backend_covariance_final_scale": True,
        "explicit_final_normalization": False,
    }

    if method_key == "graphical_lasso" and backend_key == "jax":
        backend_key = "sklearn"
        info["fallback"] = "sklearn"
        info["fallback_reason"] = "graphical_lasso is sklearn-only."

    if backend_key in {"auto", "jax"} and method_key in {"empirical", "ledoit_wolf", "oas"}:
        try:
            ncm, jax_info = _jax_covariance(
                traces,
                method=method_key,
                device=device,
                max_gpu_memory_mb=max_gpu_memory_mb,
                chunk_size=chunk_size,
            )
            info.update(jax_info)
            if ncm.shape != (n_samples, n_samples):
                raise RuntimeError(f"Invalid JAX covariance shape: {ncm.shape}.")
            return ncm, info
        except Exception as exc:
            info["jax_failed"] = True
            info["jax_failure_reason"] = str(exc)
            if backend_key == "jax" and _is_oom_error(exc):
                retry_budget = min(float(max_gpu_memory_mb), 1024.0)
                retry_chunk_size = max(1, int(chunk_size) // 2)
                try:
                    ncm, jax_info = _jax_covariance(
                        traces,
                        method=method_key,
                        device=device,
                        max_gpu_memory_mb=retry_budget,
                        chunk_size=retry_chunk_size,
                    )
                    info["jax_retry_after_oom"] = True
                    info["jax_retry_budget_mb"] = float(retry_budget)
                    info["jax_retry_chunk_size"] = int(retry_chunk_size)
                    info.update(jax_info)
                    if ncm.shape != (n_samples, n_samples):
                        raise RuntimeError(f"Invalid JAX covariance shape: {ncm.shape}.")
                    return ncm, info
                except Exception as retry_exc:
                    info["jax_retry_failed"] = True
                    info["jax_retry_failure_reason"] = str(retry_exc)
                    raise RuntimeError(f"JAX backend failed for method={method_key}: {retry_exc}") from retry_exc
            if backend_key == "jax":
                raise RuntimeError(f"JAX backend failed for method={method_key}: {exc}") from exc
            info["fallback"] = "sklearn"
            info["fallback_reason"] = f"JAX unavailable/failed: {exc}"

    try:
        ncm, sk_info = _sklearn_covariance(traces, method=method_key, **kwargs)
        info.update(sk_info)
    except Exception as exc:
        ncm = _symmetrize(_empirical_covariance(traces))
        info.update({"method": "empirical", "backend": "numpy", "fallback": "empirical", "fallback_reason": str(exc), "ddof": 0})

    if ncm.shape != (n_samples, n_samples):
        raise RuntimeError(f"Invalid covariance shape returned by estimator: {ncm.shape}.")
    return _symmetrize(np.asarray(ncm, dtype=np.float64)), info


def compute_precision_matrix(
    ncm: ArrayLike,
    method: str = "direct",
    backend: str = "auto",
    device: Any | None = None,
    regularization: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute a precision matrix."""
    matrix = _symmetrize(validate_square_matrix(ncm, name="ncm"))
    n = matrix.shape[0]
    method_key = (method or "direct").strip().lower()
    backend_key = _resolve_backend(backend)
    if method_key not in {"direct", "pinv", "cholesky"}:
        raise ValueError("method must be one of {'direct','pinv','cholesky'}.")

    if backend_key in {"auto", "jax"} and method_key in {"direct", "cholesky"} and _jax_ready():
        try:
            cov_jax = jnp.asarray(matrix, dtype=jnp.float64)  # type: ignore[union-attr]
            if device is not None:
                cov_jax = jax.device_put(cov_jax, device)  # type: ignore[union-attr]
            precision = precision_matrix_jax(cov_jax, regularization=float(max(0.0, regularization)))  # type: ignore[misc]
            precision_np = _symmetrize(np.asarray(precision, dtype=np.float64))
            if np.all(np.isfinite(precision_np)):
                return precision_np
        except Exception:
            if backend_key == "jax":
                raise

    matrix_reg = matrix + float(max(0.0, regularization)) * np.eye(n, dtype=np.float64)
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
            precision_np = _inv_nla_jit(np.asarray(matrix_reg, dtype=np.float64))
    except Exception:
        precision_np = np.linalg.pinv(matrix_reg)

    precision_np = _symmetrize(np.asarray(precision_np, dtype=np.float64))
    if not np.all(np.isfinite(precision_np)):
        precision_np = _symmetrize(np.asarray(np.linalg.pinv(matrix_reg), dtype=np.float64))

    ident = np.eye(n, dtype=np.float64)
    residual = np.linalg.norm(matrix_reg @ precision_np - ident, ord="fro") / float(max(n, 1))
    if (not np.isfinite(cond)) or cond > 1e12 or residual > 1e-5:
        precision_np = _symmetrize(np.asarray(np.linalg.pinv(matrix_reg), dtype=np.float64))
    return precision_np


def compute_covariance_diagnostics(
    ncm: ArrayLike,
    precision_matrix: ArrayLike,
) -> dict[str, dict[str, Any]]:
    ncm_arr = _symmetrize(validate_square_matrix(ncm, name="ncm"))
    precision_arr = _symmetrize(validate_square_matrix(precision_matrix, name="precision_matrix"))
    if ncm_arr.shape != precision_arr.shape:
        raise ValueError(f"ncm and precision_matrix must have same shape, got {ncm_arr.shape} and {precision_arr.shape}.")

    ncm_eigs = _eigenvalues_symmetric(ncm_arr)
    prec_eigs = _eigenvalues_symmetric(precision_arr)
    ncm_abs = np.abs(ncm_arr)
    ncm_scale = float(np.max(ncm_abs)) if ncm_abs.size else 1.0
    ncm_sparsity = float(np.mean(ncm_abs <= (1e-10 * max(1.0, ncm_scale))))

    prec_abs = np.abs(precision_arr)
    prec_scale = float(np.max(prec_abs)) if prec_abs.size else 1.0
    prec_sparsity = float(np.mean(prec_abs <= (1e-10 * max(1.0, prec_scale))))

    return {
        "ncm": {
            "condition_number": float(np.linalg.cond(ncm_arr)),
            "eigenvalues": ncm_eigs,
            "rank": int(np.linalg.matrix_rank(ncm_arr)),
            "trace": float(np.trace(ncm_arr)),
            "frobenius_norm": float(np.linalg.norm(ncm_arr, ord="fro")),
            "sparsity": ncm_sparsity,
        },
        "precision": {
            "condition_number": float(np.linalg.cond(precision_arr)),
            "eigenvalues": prec_eigs,
            "rank": int(np.linalg.matrix_rank(precision_arr)),
            "sparsity": prec_sparsity,
        },
    }


def compute_combined_ncm(
    corrected_traces,
    reference_traces,
    method: str = "ledoit_wolf",
    backend: str = "auto",
    device=None,
    max_gpu_memory_mb: float = 4096.0,
    chunk_size: int = 64,
    transfer_regularization: float = 1e-6,
):
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

    # Simulate the reference traces through h(t).
    use_jax = _jax_ready() and backend in {"auto", "jax"}
    jax_fail_reason = None

    if use_jax:
        try:
            from .jax_ops import transfer_impulse_response_jax, simulate_ref_traces_jax

            if device is None:
                gpus = [d for d in jax.devices() if d.platform == "gpu"]
                target_device = gpus[0] if gpus else jax.devices("cpu")[0]
            else:
                target_device = device

            mc = jax.device_put(jnp.asarray(mean_corr, dtype=jnp.float64), target_device)
            mr = jax.device_put(jnp.asarray(mean_ref, dtype=jnp.float64), target_device)
            rj = jax.device_put(jnp.asarray(ref, dtype=jnp.float64), target_device)

            h = transfer_impulse_response_jax(mc, mr, transfer_regularization)
            simulated_np = np.asarray(simulate_ref_traces_jax(h, rj), dtype=np.float64)
        except Exception as exc:
            use_jax = False
            jax_fail_reason = str(exc)

    if not use_jax:
        S_c = np.fft.rfft(mean_corr)
        S_r = np.fft.rfft(mean_ref)
        reg = transfer_regularization * np.max(np.abs(S_r))
        h_np = np.fft.irfft(S_c / (S_r + reg), n=n_samples)
        simulated_np = np.array(
            [np.convolve(h_np, ref[i])[:n_samples] for i in range(ref.shape[0])],
            dtype=np.float64,
        )

    common_kw = dict(
        method=method,
        backend=backend,
        device=device,
        max_gpu_memory_mb=max_gpu_memory_mb,
        chunk_size=chunk_size,
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

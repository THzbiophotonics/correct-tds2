"""JAX covariance estimators and precision matrix, all JIT-compiled.

Input: residuals of shape (n_traces, n_samples).
Output: (covariance_matrix, shrinkage_coefficient).
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import jit, vmap

__all__ = [
    "empirical_covariance_jax",
    "ledoit_wolf_jax",
    "oas_jax",
    "precision_matrix_jax",
    "transfer_impulse_response_jax",
    "simulate_ref_traces_jax",
]


@jit
def empirical_covariance_jax(residuals: jax.Array) -> jax.Array:
    """Return the empirical covariance matrix (shape n_samples × n_samples)."""
    n = residuals.shape[0]
    return (residuals.T @ residuals) / n


@jit
def _ledoit_wolf_from_cov_jax(
    sample_cov: jax.Array,
    n_observations: int | jax.Array,
    norm4: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Ledoit-Wolf shrinkage toward ``mu * I``.

    norm4 = sum_k ||x_k||^4, required to estimate the noise term exactly.
    """
    n = jnp.asarray(n_observations, dtype=sample_cov.dtype)
    p = sample_cov.shape[0]
    trace_s = jnp.trace(sample_cov)
    mean_variance = trace_s / p
    shrinkage_target = mean_variance * jnp.eye(p, dtype=sample_cov.dtype)
    delta = jnp.sum((sample_cov - shrinkage_target) ** 2)
    # tr(S^2) == ||S||_F^2 for symmetric S
    trace_s2 = jnp.sum(sample_cov ** 2)
    # beta = (1/n^2) * sum_k ||x_k x_k^T - S||^2 = (norm4 - n*tr(S^2)) / n^2
    beta = jnp.maximum(0.0, (norm4 - n * trace_s2) / (n * n))
    shrinkage = jnp.clip(beta / (delta + 1e-12), 0.0, 1.0)
    cov = (1.0 - shrinkage) * sample_cov + shrinkage * shrinkage_target
    return 0.5 * (cov + cov.T), shrinkage


@jit
def _oas_from_cov_jax(
    sample_cov: jax.Array,
    n_observations: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply OAS shrinkage toward ``mu * I`` given a sample covariance."""
    n = jnp.asarray(n_observations, dtype=sample_cov.dtype)
    p = sample_cov.shape[0]
    trace_s = jnp.trace(sample_cov)
    mean_variance = trace_s / p
    shrinkage_target = mean_variance * jnp.eye(p, dtype=sample_cov.dtype)
    trace_s2 = jnp.trace(sample_cov @ sample_cov)
    numerator = (1.0 - 2.0 / p) * trace_s2 + trace_s**2
    denominator = (n + 1.0 - 2.0 / p) * (trace_s2 - trace_s**2 / p) + 1e-12
    shrinkage = jnp.clip(numerator / denominator, 0.0, 1.0)
    cov = (1.0 - shrinkage) * sample_cov + shrinkage * shrinkage_target
    return 0.5 * (cov + cov.T), shrinkage


@jit
def ledoit_wolf_jax(residuals: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(covariance, shrinkage)`` using the Ledoit-Wolf estimator."""
    sample_cov = empirical_covariance_jax(residuals)
    norm4 = jnp.sum(jnp.sum(residuals ** 2, axis=1) ** 2)
    return _ledoit_wolf_from_cov_jax(sample_cov, residuals.shape[0], norm4)


@jit
def oas_jax(residuals: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(covariance, shrinkage)`` using the OAS estimator."""
    sample_cov = empirical_covariance_jax(residuals)
    return _oas_from_cov_jax(sample_cov, residuals.shape[0])


@jit
def precision_matrix_jax(
    cov: jax.Array,
    regularization: float = 1e-10,
) -> jax.Array:
    """Return the regularized precision matrix via Cholesky decomposition."""
    cov_reg = cov + regularization * jnp.eye(cov.shape[0])
    lmat = jnp.linalg.cholesky(cov_reg)
    ymat = jsp_linalg.solve_triangular(lmat, jnp.eye(cov.shape[0]), lower=True)
    return jsp_linalg.solve_triangular(lmat.T, ymat, lower=False)


@jit
def transfer_impulse_response_jax(
    mean_corrected: jax.Array,
    mean_reference: jax.Array,
    regularization: float = 1e-6,
) -> jax.Array:
    """Estimate h(t) so that corrected ≈ h * reference (Wiener deconvolution)."""
    S_corr = jnp.fft.rfft(mean_corrected)
    S_ref = jnp.fft.rfft(mean_reference)
    reg = regularization * jnp.max(jnp.abs(S_ref))
    H = S_corr / (S_ref + reg)
    return jnp.fft.irfft(H, n=mean_corrected.shape[0])


@jit
def _convolve_single(h: jax.Array, ref_trace: jax.Array) -> jax.Array:
    """Convolve one reference trace with the impulse response (linear conv.)."""
    n = h.shape[0]
    n2 = 2 * n
    return jnp.fft.irfft(jnp.fft.rfft(h, n=n2) * jnp.fft.rfft(ref_trace, n=n2), n=n2)[:n]


@jit
def simulate_ref_traces_jax(h: jax.Array, ref_traces: jax.Array) -> jax.Array:
    """Convolve every reference trace with the impulse response h(t)."""
    return vmap(_convolve_single, in_axes=(None, 0))(h, ref_traces)

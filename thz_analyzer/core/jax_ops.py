from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import grad, jit, vmap

__all__ = [
    "rfft_angular_freqs",
    "squash_to_bounds",
    "LASER_FREP_HZ",
    "compute_superresolution_npadded",
    "zero_pad_single",
    "zero_pad_batch",
    "apply_single_correction",
    "apply_batch_corrections",
    "trace_loss",
    "batch_losses",
    "batch_gradients",
    "empirical_covariance_jax",
    "ledoit_wolf_jax",
    "oas_jax",
    "precision_matrix_jax",
    "transfer_impulse_response_jax",
    "linear_convolve_single_jax",
    "simulate_ref_traces_jax",
    "make_optax_optimizer",
    "optax_step",
    "optax_train_step",
    "optax_train_block",
    "infer_initial_lr",
]


@jit
def rfft_angular_freqs(time_axis: jnp.ndarray) -> jnp.ndarray:
    """Return RFFT angular frequencies."""
    dt = time_axis[1] - time_axis[0]
    n = time_axis.shape[0]
    freqs = jnp.fft.rfftfreq(n, d=dt)
    return 2.0 * jnp.pi * freqs


@jit
def squash_to_bounds(
    params_raw: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> jnp.ndarray:
    """Map raw parameters into their bounds."""
    return lower + (upper - lower) * jax.nn.sigmoid(params_raw)


LASER_FREP_HZ: float = 99_991_499.600


def compute_superresolution_npadded(dt: float, frep_hz: float = LASER_FREP_HZ) -> int:
    """Return the padded length for superresolution."""
    return int(round(1.0 / (frep_hz * dt)))


@partial(jit, static_argnums=(1,))
def zero_pad_single(trace: jnp.ndarray, n_padded: int) -> jnp.ndarray:
    """Pad one trace with trailing zeros."""
    return jnp.pad(trace, (0, n_padded - trace.shape[0]))


zero_pad_batch = jit(vmap(zero_pad_single, in_axes=(0, None)), static_argnums=(1,))


@jit
def apply_single_correction(
    pulse: jnp.ndarray,
    time_axis: jnp.ndarray,
    omega: jnp.ndarray,
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Apply delay, amplitude and dilation to one trace."""
    delay, amp_factor, dil_factor = params

    # Shift in frequency space.
    z = jnp.exp(1j * omega * delay)
    pulse_shifted = jnp.fft.irfft(z * jnp.fft.rfft(pulse), n=pulse.shape[0])

    # Differentiate for dilation.
    dt = time_axis[1] - time_axis[0]

    dx = jnp.concatenate(
        [
            jnp.array([(pulse_shifted[1] - pulse_shifted[0]) / dt]),
            (pulse_shifted[2:] - pulse_shifted[:-2]) / (2.0 * dt),
            jnp.array([(pulse_shifted[-1] - pulse_shifted[-2]) / dt]),
        ]
    )

    pulse_dilated = pulse_shifted - (dil_factor * time_axis) * dx
    return (1.0 - amp_factor) * pulse_dilated


_apply_batch_vmap = vmap(
    apply_single_correction,
    in_axes=(0, None, None, 0),
)


@jit
def apply_batch_corrections(
    pulses: jnp.ndarray,
    time_axis: jnp.ndarray,
    omega: jnp.ndarray,
    params_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Apply corrections to a batch of traces."""
    return _apply_batch_vmap(pulses, time_axis, omega, params_matrix)


@jit
def trace_loss(
    params: jnp.ndarray,
    pulse: jnp.ndarray,
    reference: jnp.ndarray,
    time_axis: jnp.ndarray,
    omega: jnp.ndarray,
) -> jnp.ndarray:
    """Return the normalized L2 loss for one trace."""
    corrected = apply_single_correction(pulse, time_axis, omega, params)

    numerator = jnp.linalg.norm(reference - corrected)
    denominator = jnp.linalg.norm(reference) + 1e-12

    return numerator / denominator


def _bounded_trace_loss(
    params_raw: jnp.ndarray,
    pulse: jnp.ndarray,
    reference: jnp.ndarray,
    time_axis: jnp.ndarray,
    omega: jnp.ndarray,
    lower_bounds: jnp.ndarray,
    upper_bounds: jnp.ndarray,
) -> jnp.ndarray:
    """Loss with bounded parameters (internal)."""
    params = squash_to_bounds(params_raw, lower_bounds, upper_bounds)
    return trace_loss(params, pulse, reference, time_axis, omega)


_bounded_loss_grad = grad(_bounded_trace_loss, argnums=0)

batch_losses = jit(
    vmap(
        _bounded_trace_loss,
        in_axes=(0, 0, None, None, None, None, None),
    )
)

batch_gradients = jit(
    vmap(
        _bounded_loss_grad,
        in_axes=(0, 0, None, None, None, None, None),
    )
)


@jit
def empirical_covariance_jax(residuals: jnp.ndarray) -> jnp.ndarray:
    """Return the empirical covariance matrix."""
    n = residuals.shape[0]
    return (residuals.T @ residuals) / n


@jit
def _ledoit_wolf_from_cov_jax(
    sample_cov: jnp.ndarray,
    n_observations,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Covariance-only Ledoit-Wolf-style shrinkage toward ``mu * I``."""
    n = jnp.asarray(n_observations, dtype=sample_cov.dtype)
    p = sample_cov.shape[0]
    trace_s = jnp.trace(sample_cov)
    mean_variance = trace_s / p
    shrinkage_target = mean_variance * jnp.eye(p, dtype=sample_cov.dtype)
    delta_hat = jnp.sum((sample_cov - shrinkage_target) ** 2)
    beta_hat = (jnp.sum(sample_cov**2) - trace_s**2 / p) / n
    shrinkage = jnp.clip(beta_hat / (delta_hat + 1e-12), 0.0, 1.0)
    cov = (1.0 - shrinkage) * sample_cov + shrinkage * shrinkage_target
    cov = 0.5 * (cov + cov.T)
    return cov, shrinkage


@jit
def _oas_from_cov_jax(
    sample_cov: jnp.ndarray,
    n_observations,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """OAS shrinkage toward ``mu * I`` from an empirical covariance matrix."""
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
    cov = 0.5 * (cov + cov.T)
    return cov, shrinkage


@jit
def ledoit_wolf_jax(residuals: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the Ledoit-Wolf covariance estimate."""
    sample_cov = empirical_covariance_jax(residuals)
    return _ledoit_wolf_from_cov_jax(sample_cov, residuals.shape[0])


@jit
def oas_jax(residuals: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return the OAS covariance estimate."""
    sample_cov = empirical_covariance_jax(residuals)
    return _oas_from_cov_jax(sample_cov, residuals.shape[0])


@jit
def precision_matrix_jax(
    cov: jnp.ndarray,
    regularization: float = 1e-10,
) -> jnp.ndarray:
    """Return the regularized inverse covariance matrix."""
    cov_reg = cov + regularization * jnp.eye(cov.shape[0])
    lmat = jnp.linalg.cholesky(cov_reg)
    ymat = jsp_linalg.solve_triangular(lmat, jnp.eye(cov.shape[0]), lower=True)
    precision = jsp_linalg.solve_triangular(lmat.T, ymat, lower=False)
    return precision


@jit
def transfer_impulse_response_jax(
    mean_corrected: jnp.ndarray,
    mean_reference: jnp.ndarray,
    regularization: float = 1e-6,
) -> jnp.ndarray:
    """Estimate the impulse response from the mean traces."""
    S_corr = jnp.fft.rfft(mean_corrected)
    S_ref = jnp.fft.rfft(mean_reference)
    reg = regularization * jnp.max(jnp.abs(S_ref))
    H = S_corr / (S_ref + reg)
    return jnp.fft.irfft(H, n=mean_corrected.shape[0])


@jit
def linear_convolve_single_jax(
    h: jnp.ndarray,
    ref_trace: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve one reference trace with the impulse response."""
    n = h.shape[0]
    n2 = 2 * n
    H_full = jnp.fft.rfft(h, n=n2)
    R_full = jnp.fft.rfft(ref_trace, n=n2)
    conv_full = jnp.fft.irfft(H_full * R_full, n=n2)
    return conv_full[:n]


_simulate_ref_traces_vmap = vmap(linear_convolve_single_jax, in_axes=(None, 0))


@jit
def simulate_ref_traces_jax(
    h: jnp.ndarray,
    ref_traces: jnp.ndarray,
) -> jnp.ndarray:
    """Convolve all reference traces with the impulse response."""
    return _simulate_ref_traces_vmap(h, ref_traces)


def make_optax_optimizer(init_lr: float, decay_steps: int, schedule_type: str = "cosine"):
    import optax
    schedule_key = (schedule_type or "cosine").lower()
    if schedule_key == "cosine":
        schedule = optax.cosine_decay_schedule(init_value=init_lr, decay_steps=decay_steps, alpha=0.0)
    elif schedule_key == "exp":
        schedule = optax.exponential_decay(init_value=init_lr, transition_steps=max(1, decay_steps // 4), decay_rate=0.5)
    elif schedule_key == "piecewise":
        schedule = optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales={decay_steps // 3: 0.3, 2 * decay_steps // 3: 0.1})
    else:
        raise ValueError("schedule_type must be 'cosine', 'exp', or 'piecewise'")
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )
    return tx, schedule


@partial(jit, static_argnums=(0,))
def optax_step(tx, params, opt_state, grads):
    import optax
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


@partial(jit, static_argnums=(0,))
def optax_train_step(
    tx,
    params,
    opt_state,
    pulses,
    reference,
    time_axis,
    angular_freqs,
    lower_bounds,
    upper_bounds,
):
    grads = batch_gradients(
        params,
        pulses,
        reference,
        time_axis,
        angular_freqs,
        lower_bounds,
        upper_bounds,
    )
    return optax_step(tx, params, opt_state, grads)


@partial(jit, static_argnums=(0, 9))
def optax_train_block(
    tx,
    params,
    opt_state,
    pulses,
    reference,
    time_axis,
    angular_freqs,
    lower_bounds,
    upper_bounds,
    steps,
):
    def _body(_, carry):
        curr_params, curr_state = carry
        next_params, next_state = optax_train_step(
            tx,
            curr_params,
            curr_state,
            pulses,
            reference,
            time_axis,
            angular_freqs,
            lower_bounds,
            upper_bounds,
        )
        return next_params, next_state

    return jax.lax.fori_loop(0, steps, _body, (params, opt_state))


def infer_initial_lr(parameter_matrix, pulses, reference, time_axis, angular_freqs, lower_bounds, upper_bounds, target_step=1e-2, eps=1e-8, min_lr=5e-3):
    grads = batch_gradients(parameter_matrix, pulses, reference, time_axis, angular_freqs, lower_bounds, upper_bounds)
    grad_norm = jnp.linalg.norm(grads)
    if grad_norm <= eps:
        return float(min_lr)
    lr_raw = float(target_step / grad_norm)
    return max(lr_raw, min_lr)

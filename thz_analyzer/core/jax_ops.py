"""JAX signal-processing primitives for the THz correction pipeline.

Covers frequency helpers, zero-padding, per-trace corrections (delay/amplitude/dilation),
and batch loss + gradient computation. See jax_covariance for NCM, optimizer for training.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.typing import ArrayLike

__all__ = [
    "LASER_FREP_HZ",
    "rfft_angular_freqs",
    "squash_to_bounds",
    "compute_superresolution_npadded",
    "zero_pad_single",
    "zero_pad_batch",
    "apply_single_correction",
    "apply_batch_corrections",
    "trace_loss",
    "batch_losses",
    "batch_gradients",
]

# Laser rep. frequency — used to compute the superresolution padding length.
LASER_FREP_HZ: float = 99_991_499.600


@jit
def rfft_angular_freqs(time_axis: jax.Array) -> jax.Array:
    """Return the angular frequencies (rad/s) for a real FFT of ``time_axis``."""
    dt = time_axis[1] - time_axis[0]
    n = time_axis.shape[0]
    return 2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=dt)


@jit
def squash_to_bounds(
    params_raw: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    """Map unconstrained ``params_raw`` into ``[lower, upper]`` via sigmoid."""
    return lower + (upper - lower) * jax.nn.sigmoid(params_raw)


def compute_superresolution_npadded(dt: float, frep_hz: float = LASER_FREP_HZ) -> int:
    """Number of samples that fills exactly one laser period (enables superresolution)."""
    return int(round(1.0 / (frep_hz * dt)))


@partial(jit, static_argnums=(1,))
def zero_pad_single(trace: jax.Array, n_padded: int) -> jax.Array:
    """Pad one trace with trailing zeros to length ``n_padded``."""
    return jnp.pad(trace, (0, n_padded - trace.shape[0]))


zero_pad_batch = jit(vmap(zero_pad_single, in_axes=(0, None)), static_argnums=(1,))
"""Batched version of zero_pad_single."""


@jit
def apply_single_correction(
    pulse: jax.Array,
    time_axis: jax.Array,
    omega: jax.Array,
    params: jax.Array,
) -> jax.Array:
    """Apply delay, amplitude, and dilation corrections to one trace.

    params = [delay, amp_factor, dil_factor].
    """
    delay, amp_factor, dil_factor = params

    # time shift in frequency domain (Fourier shift theorem)
    pulse_shifted = jnp.fft.irfft(
        jnp.exp(1j * omega * delay) * jnp.fft.rfft(pulse),
        n=pulse.shape[0],
    )

    # Matches np.gradient(pulse, dt) convention used by Correct-TDS.
    dt = time_axis[1] - time_axis[0]
    dx = jnp.concatenate([
        jnp.array([(pulse_shifted[1] - pulse_shifted[0]) / dt]),
        (pulse_shifted[2:] - pulse_shifted[:-2]) / (2.0 * dt),
        jnp.array([(pulse_shifted[-1] - pulse_shifted[-2]) / dt]),
    ])

    pulse_dilated = pulse_shifted - (dil_factor * time_axis) * dx
    return (1.0 - amp_factor) * pulse_dilated


apply_batch_corrections = jit(
    vmap(apply_single_correction, in_axes=(0, None, None, 0))
)
"""Batched version of apply_single_correction."""


@jit
def trace_loss(
    params: jax.Array,
    pulse: jax.Array,
    reference: jax.Array,
    time_axis: jax.Array,
    omega: jax.Array,
) -> jax.Array:
    """Normalised L2 loss between the corrected pulse and the reference."""
    corrected = apply_single_correction(pulse, time_axis, omega, params)
    return jnp.linalg.norm(reference - corrected) / (jnp.linalg.norm(reference) + 1e-12)


def _bounded_trace_loss(
    params_raw: jax.Array,
    pulse: jax.Array,
    reference: jax.Array,
    time_axis: jax.Array,
    omega: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
) -> jax.Array:
    """Loss with parameters squashed into their physical bounds (for autograd)."""
    params = squash_to_bounds(params_raw, lower_bounds, upper_bounds)
    return trace_loss(params, pulse, reference, time_axis, omega)


batch_losses = jit(
    vmap(_bounded_trace_loss, in_axes=(0, 0, None, None, None, None, None))
)
"""Per-trace bounded loss over a batch."""

batch_gradients = jit(
    vmap(grad(_bounded_trace_loss, argnums=0), in_axes=(0, 0, None, None, None, None, None))
)
"""Per-trace gradient of the bounded loss over a batch."""

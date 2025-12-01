from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

__all__ = [
    "apply_corrections",
    "apply_corrections_batch",
    "adam_batch_step",
    "squash_to_bounds",
    "trace_loss",
    "batched_gradients",
    "batched_losses",
]


@jit
def apply_corrections(pulse, t, w, params):
    """Apply delay, amplitude, and dilation corrections to a signal."""
    delay, a, dil_a = params
    Z = jnp.exp(1j * w * delay)
    x_delayed = jnp.fft.irfft(Z * jnp.fft.rfft(pulse))
    dt = t[1] - t[0]
    dx = jnp.concatenate(
        [
            jnp.array([0.0]),
            (x_delayed[2:] - x_delayed[:-2]) / (2 * dt),
            jnp.array([0.0]),
        ]
    )
    x_dil = x_delayed - (dil_a * t) * dx
    return (1.0 - a) * x_dil


_apply_corrections_batch = jit(
    vmap(
        lambda signal, params, t_axis, omega: apply_corrections(
            signal, t_axis, omega, params
        ),
        in_axes=(0, 0, None, None),
    )
)


def apply_corrections_batch(pulses, t, w, params_matrix):
    """Apply corrections to a batch of signals in one pass."""
    return _apply_corrections_batch(pulses, params_matrix, t, w)


@jit
def adam_batch_step(u, m, v, g, i, lr):
    """Vectorized Adam update."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** i)
    vhat = v / (1 - b2 ** i)
    return u - lr * mhat / (jnp.sqrt(vhat) + eps), m, v


@jit
def squash_to_bounds(u, lo, hi):
    """Project unconstrained parameters back inside [lo, hi] via sigmoid."""
    return lo + (hi - lo) * jax.nn.sigmoid(u)


@jit
def trace_loss(params, pulse, ref, t, w):
    """Normalized loss for a corrected trace."""
    y = apply_corrections(pulse, t, w, params)
    num = jnp.linalg.norm(ref - y)
    den = jnp.linalg.norm(ref) + 1e-12
    return num / den


def _bounded_trace_loss(parameter_vector, signal_trace, reference, time_axis, angular_freqs, lower_bounds, upper_bounds):
    bounded = squash_to_bounds(parameter_vector, lower_bounds, upper_bounds)
    return trace_loss(bounded, signal_trace, reference, time_axis, angular_freqs)


_bounded_trace_loss_grad = grad(_bounded_trace_loss)

batched_gradients = jit(
    vmap(
        _bounded_trace_loss_grad,
        in_axes=(0, 0, None, None, None, None, None),
    )
)

batched_losses = jit(
    vmap(
        _bounded_trace_loss,
        in_axes=(0, 0, None, None, None, None, None),
    )
)

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

__all__ = [
    "preferred_device",
    "resolve_device",
    "rfft_omega",
    "apply_corrections",
    "apply_corrections_batch",
    "adam_batch_step",
    "squash_to_bounds",
    "trace_loss",
    "batched_gradients",
    "batched_losses",
]


def preferred_device(prefer_gpu: bool = True):
    """Return a preferred JAX device (GPU if available, otherwise CPU)."""
    if jax is None:
        return None
    if prefer_gpu:
        gpus = [device for device in jax.devices() if device.platform == "gpu"]
        if gpus:
            return gpus[0]
    cpu_devices = jax.devices("cpu")
    return cpu_devices[0] if cpu_devices else None


def resolve_device(preference: str = "cpu"):
    """
    Return a JAX device matching the requested preference along with a flag
    indicating whether the exact request was satisfied.
    """
    if jax is None:
        raise RuntimeError("JAX is not installed: cannot select a device.")
    requested = (preference or "cpu").lower()
    matching = [device for device in jax.devices() if device.platform == requested]
    if matching:
        return matching[0], True
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("No JAX CPU device available.")
    return cpu_devices[0], requested == "cpu"


def rfft_omega(time_axis):
    """Compute angular frequencies corresponding to the real FFT bins."""
    dt = float(time_axis[1] - time_axis[0])
    freqs = np.fft.rfftfreq(len(time_axis), d=dt)
    return 2.0 * np.pi * freqs


@jit
def apply_corrections(pulse, t, w, params):
    """Apply time shift, amplitude scaling, and time dilation to a pulse."""
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
    """Run apply_corrections on every pulse in the batch using shared axes."""
    return _apply_corrections_batch(pulses, params_matrix, t, w)


@jit
def adam_batch_step(u, m, v, g, i, lr):
    """Single Adam update step applied elementwise to vector batches."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** i)
    vhat = v / (1 - b2 ** i)
    return u - lr * mhat / (jnp.sqrt(vhat) + eps), m, v


@jit
def squash_to_bounds(u, lo, hi):
    """Squash unconstrained parameters into [lo, hi] using a sigmoid."""
    return lo + (hi - lo) * jax.nn.sigmoid(u)


@jit
def trace_loss(params, pulse, ref, t, w):
    """Measure how far a corrected pulse is from the reference in L2 norm."""
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

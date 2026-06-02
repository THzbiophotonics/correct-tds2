from typing import Any, Optional

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from .jax_ops import (
    apply_batch_corrections,
    batch_gradients,
    batch_losses,
    rfft_angular_freqs,
    zero_pad_batch,
    compute_superresolution_npadded,
    LASER_FREP_HZ,
)
from .filters import build_frequency_window

__all__ = ["CorrectionModel"]


class CorrectionModel:
    """Apply JAX corrections on a batch of traces."""

    def __init__(
        self,
        time_axis: ArrayLike,
        device: Optional[Any] = None,
        superresolution: bool = False,
        frep_hz: float = LASER_FREP_HZ,
        filter_low: bool = False,
        filter_high: bool = False,
        freq_start: float = 0.0,
        freq_end: float = 1e13,
        filter_sharpness: float = 1.0,
    ):
        """Pre-compute and upload time axis, angular frequencies, and frequency window to device."""
        self.device = self._resolve_device(device)

        time_jnp = jnp.asarray(time_axis)
        self.n_original = int(time_jnp.shape[0])
        dt = float(time_jnp[1] - time_jnp[0])
        self.superresolution = superresolution

        if superresolution:
            self.n_padded = compute_superresolution_npadded(dt, frep_hz)
            time_for_model = jnp.arange(self.n_padded) * dt
            freqwindow_np = build_frequency_window(
                self.n_padded,
                dt,
                filter_low,
                filter_high,
                freq_start,
                freq_end,
                filter_sharpness,
            )
        else:
            self.n_padded = self.n_original
            time_for_model = time_jnp
            freqwindow_np = build_frequency_window(
                self.n_original,
                dt,
                filter_low,
                filter_high,
                freq_start,
                freq_end,
                filter_sharpness,
            )

        self.time_axis = time_for_model
        self.omega = rfft_angular_freqs(self.time_axis)

        self.freqwindow = jax.device_put(
            jnp.asarray(freqwindow_np, dtype=jnp.float32),
            self.device,
        )

    @staticmethod
    def _resolve_device(device: Optional[Any]):
        """Return the requested JAX device."""
        if device is None:
            gpus = [d for d in jax.devices() if d.platform == "gpu"]
            return gpus[0] if gpus else jax.devices("cpu")[0]
        if isinstance(device, str):
            pref = device.strip().lower()
            matches = [d for d in jax.devices() if d.platform == pref]
            if matches:
                return matches[0]
            return jax.devices("cpu")[0]
        return device

    def prepare_inputs(
        self,
        pulses: ArrayLike,
        reference: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        """Pad and filter the inputs."""
        with jax.default_device(self.device):
            if self.superresolution:
                pulses_padded = zero_pad_batch(jnp.asarray(pulses), self.n_padded)
                ref_padded = jnp.pad(jnp.asarray(reference), (0, self.n_padded - self.n_original))
            else:
                pulses_padded = jnp.asarray(pulses)
                ref_padded = jnp.asarray(reference)

            S_p = jnp.fft.rfft(pulses_padded, axis=-1) * self.freqwindow[None, :]
            S_r = jnp.fft.rfft(ref_padded) * self.freqwindow
            pulses_filtered = jnp.fft.irfft(S_p, n=self.n_padded, axis=-1)
            ref_filtered = jnp.fft.irfft(S_r, n=self.n_padded)
            return pulses_filtered, ref_filtered

    def apply(
        self,
        pulses: ArrayLike,
        params: ArrayLike,
    ) -> jax.Array:
        """Apply the correction to a batch of pulses with the given parameters."""
        pulses_jax = jnp.asarray(pulses)
        params_jax = jnp.asarray(params)

        with jax.default_device(self.device):
            if self.superresolution:
                pulses_padded = zero_pad_batch(pulses_jax, self.n_padded)
            else:
                pulses_padded = pulses_jax
            S_filtered = jnp.fft.rfft(pulses_padded, axis=-1) * self.freqwindow[None, :]
            pulses_filtered = jnp.fft.irfft(S_filtered, n=self.n_padded, axis=-1)

            return apply_batch_corrections(
                pulses_filtered,
                self.time_axis,
                self.omega,
                params_jax,
            )

    def loss(
        self,
        params: ArrayLike,
        pulses: ArrayLike,
        reference: ArrayLike,
        bounds: tuple[ArrayLike, ArrayLike],
    ) -> jax.Array:
        """Normalised L2 loss in the time domain after applying frequency-windowed corrections."""
        lower, upper = bounds
        pulses_filtered, ref_filtered = self.prepare_inputs(pulses, reference)
        with jax.default_device(self.device):
            return batch_losses(
                jnp.asarray(params),
                pulses_filtered,
                ref_filtered,
                self.time_axis,
                self.omega,
                jnp.asarray(lower),
                jnp.asarray(upper),
            )

    def gradients(
        self,
        params: ArrayLike,
        pulses: ArrayLike,
        reference: ArrayLike,
        bounds: tuple[ArrayLike, ArrayLike],
    ) -> jax.Array:
        """Exact gradients of the loss via JAX autodiff, one gradient vector per pulse."""
        lower, upper = bounds
        pulses_filtered, ref_filtered = self.prepare_inputs(pulses, reference)
        with jax.default_device(self.device):
            return batch_gradients(
                jnp.asarray(params),
                pulses_filtered,
                ref_filtered,
                self.time_axis,
                self.omega,
                jnp.asarray(lower),
                jnp.asarray(upper),
            )

    @property
    def output_slice(self) -> slice:
        return slice(None, self.n_original)

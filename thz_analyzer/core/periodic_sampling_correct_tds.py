"""Periodic sampling correction utilities for Correct-TDS."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

__all__ = ["periodic_sampling_correct_tds"]


def _wrap_phase(phi: float) -> float:
    """Wrap a phase angle inside [-pi, pi]."""
    two_pi = 2.0 * math.pi
    return ((phi + math.pi) % two_pi) - math.pi


def _fft_error(signal: np.ndarray, index_limit: int) -> float:
    """Return the high-frequency error metric defined by Correct-TDS."""
    spectrum = np.fft.rfft(signal)
    if index_limit <= 0:
        tail = spectrum
    elif index_limit >= spectrum.shape[0]:
        return 0.0
    else:
        tail = spectrum[index_limit:]
    if tail.size == 0:
        return 0.0
    return float(np.sum(np.abs(tail)))


def periodic_sampling_correct_tds(
    mean_signal: np.ndarray,
    time_axis: np.ndarray,
    freq_limit_thz: float,
    random_trials: int = 128,
    coordinate_rounds: int = 40,
) -> Dict[str, object]:
    """
    Estimate and apply the periodic sampling correction used by Correct-TDS.

    Args:
        mean_signal: 1-D array containing the (already filtered) mean signal.
        time_axis: 1-D array with the same length as `mean_signal`.
        freq_limit_thz: User-provided frequency (THz) only used to pick the
            FFT index above which the residual energy is minimized.
        random_trials: Number of random initial guesses before local search.
        coordinate_rounds: Number of coordinate-descent refinement rounds.

    Returns:
        A dictionary with the corrected mean, the correction waveform, the
        optimized parameters, and diagnostic metrics.
    """

    mean_signal = np.asarray(mean_signal, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)
    if mean_signal.ndim != 1 or time_axis.ndim != 1:
        raise ValueError("mean_signal and time_axis must be 1-D arrays.")
    if mean_signal.shape != time_axis.shape:
        raise ValueError("mean_signal and time_axis must share the same length.")
    if mean_signal.size < 4:
        raise ValueError("Need at least 4 samples for periodic correction.")

    dt = float(time_axis[1] - time_axis[0])
    if dt == 0.0:
        raise ValueError("Time axis has zero spacing; cannot compute gradient.")

    gradient = np.gradient(mean_signal, dt)
    freqs = np.fft.rfftfreq(mean_signal.size, d=dt)
    freq_limit_hz = max(float(freq_limit_thz) * 1e12, 0.0)
    index_limit = int(np.searchsorted(freqs, freq_limit_hz))
    if index_limit < 0:
        index_limit = 0
    if index_limit > freqs.size:
        index_limit = freqs.size

    amp_span = max(5.0 * abs(dt), 1e-15)
    amp_bounds = (-amp_span, amp_span)
    omega_min = 0.0
    omega_max = 2.0 * math.pi * (freqs[-1] if freqs.size > 0 else (0.5 / max(abs(dt), 1e-15)))
    phase_bounds = (-math.pi, math.pi)

    def evaluate(params: np.ndarray) -> float:
        amp = float(np.clip(params[0], *amp_bounds))
        omega = float(np.clip(params[1], omega_min, omega_max))
        phi = _wrap_phase(float(params[2]))
        ct = amp * np.cos(omega * time_axis + phi)
        corrected = mean_signal - gradient * ct
        return _fft_error(corrected, index_limit)

    rng = np.random.default_rng(0)
    params = np.array([0.0, 0.0, 0.0], dtype=float)
    best_error = evaluate(params)

    if index_limit < freqs.size and freqs.size > 0:
        fft_mean = np.fft.rfft(mean_signal)
        tail = np.abs(fft_mean[index_limit:]) if index_limit < fft_mean.size else np.array([])
        if tail.size > 0 and np.any(np.isfinite(tail)):
            dominant = index_limit + int(np.argmax(tail))
            dom_freq = freqs[dominant]
            params[1] = 2.0 * math.pi * dom_freq
            best_error = evaluate(params)

    for _ in range(max(1, random_trials)):
        candidate = np.array(
            [
                rng.uniform(*amp_bounds),
                rng.uniform(omega_min, omega_max),
                rng.uniform(*phase_bounds),
            ],
            dtype=float,
        )
        current_error = evaluate(candidate)
        if current_error < best_error:
            best_error = current_error
            params = candidate

    step = np.array(
        [
            0.25 * (amp_bounds[1] - amp_bounds[0]),
            0.1 * (omega_max - omega_min),
            math.pi / 4.0,
        ],
        dtype=float,
    )
    min_step = np.array(
        [
            max(amp_span * 1e-4, 1e-18),
            max((omega_max - omega_min) * 1e-4, 1e-6),
            1e-4,
        ],
        dtype=float,
    )

    for _ in range(max(1, coordinate_rounds)):
        improved = False
        for axis in range(3):
            for direction in (-1.0, 1.0):
                trial = params.copy()
                trial[axis] += direction * step[axis]
                if axis == 0:
                    trial[axis] = float(np.clip(trial[axis], *amp_bounds))
                elif axis == 1:
                    trial[axis] = float(np.clip(trial[axis], omega_min, omega_max))
                else:
                    trial[axis] = _wrap_phase(trial[axis])
                curr_error = evaluate(trial)
                if curr_error + 1e-18 < best_error:
                    params = trial
                    best_error = curr_error
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if np.all(step <= min_step):
                break

    amp_best, omega_best, phi_best = params
    omega_best = float(np.clip(omega_best, omega_min, omega_max))
    amp_best = float(np.clip(amp_best, *amp_bounds))
    phi_best = _wrap_phase(float(phi_best))

    ct = amp_best * np.cos(omega_best * time_axis + phi_best)
    correction = gradient * ct
    corrected_signal = mean_signal - correction
    residual_error = _fft_error(corrected_signal, index_limit)

    return {
        "corrected_signal": corrected_signal,
        "correction_waveform": correction,
        "ct": ct,
        "params": {
            "A": amp_best,
            "omega": omega_best,
            "phi": phi_best,
        },
        "error": residual_error,
        "freq_limit_hz": freq_limit_hz,
        "index_freq_limit": index_limit,
    }

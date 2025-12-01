import numpy as np

__all__ = [
    "_compute_mask",
    "apply_frequency_filter",
    "_compute_time_mask",
    "apply_time_filter",
]


def _compute_mask(freqs_np, filter_low, filter_high, freq_start, freq_end, sharpness):
    """
    Build the frequency-domain mask using smooth logistic transitions.

    Args:
        freqs_np (ndarray): Frequency axis.
        filter_low (bool): Whether to attenuate frequencies below ``freq_start``.
        filter_high (bool): Whether to attenuate frequencies above ``freq_end``.
        freq_start (float): Start frequency (Hz) for the transition.
        freq_end (float): End frequency (Hz) for the transition.
        sharpness (float): Logistic slope that controls how sharp the mask is.

    Returns:
        ndarray: Smooth mask between 0 and 1.
    """
    mask = np.ones_like(freqs_np)
    if bool(filter_low):
        mask *= 1.0 / (1.0 + np.exp(-(freqs_np - freq_start) * sharpness / 1e11))
    if bool(filter_high):
        mask *= 1.0 / (1.0 + np.exp((freqs_np - freq_end) * sharpness / 1e11))
    return mask


def apply_frequency_filter(freqs, spectrum, filter_low, filter_high, freq_start, freq_end, sharpness):
    """
    Apply the configured frequency filter to a spectrum.
    """
    freqs_np = np.asarray(freqs)
    spectrum_np = np.asarray(spectrum)
    mask = _compute_mask(freqs_np, filter_low, filter_high, freq_start, freq_end, sharpness)
    return spectrum_np * mask


def _compute_time_mask(t_s, filter_low, filter_high, t_start, t_end, sharpness):
    """
    Build a time-domain mask with smooth edges and hard zeros outside.

    - If ``filter_low`` is True, samples strictly before ``t_start`` are set to 0,
      with a smooth logistic rise around ``t_start`` (controlled by ``sharpness``).
    - If ``filter_high`` is True, samples strictly after ``t_end`` are set to 0,
      with a smooth logistic fall around ``t_end``.
    """
    t = np.asarray(t_s)
    mask = np.ones_like(t, dtype=float)
    scale = 1e-12  # slope scaling (seconds)

    if filter_low:
        edge_low = 1.0 / (1.0 + np.exp(-(t - t_start) * sharpness / scale))
        mask *= edge_low
        mask[t < t_start] = 0.0  # enforce zeros before start

    if filter_high:
        edge_high = 1.0 / (1.0 + np.exp((t - t_end) * sharpness / scale))
        mask *= edge_high
        mask[t > t_end] = 0.0  # enforce zeros after end

    return mask


def apply_time_filter(t_s, signals, filter_low, filter_high, t_start, t_end, sharpness):
    """
    Apply the time-domain mask to 1D or 2D signals.
    """
    sig = np.asarray(signals)
    mask = _compute_time_mask(t_s, filter_low, filter_high, t_start, t_end, sharpness)
    if sig.ndim == 1:
        return sig * mask
    if sig.ndim == 2:
        return sig * mask[None, :]
    raise ValueError("signals must be 1D or 2D")

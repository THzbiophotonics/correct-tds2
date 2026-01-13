import numpy as np

__all__ = [
    "_compute_mask",
    "apply_frequency_filter",
    "_compute_time_mask",
    "apply_time_filter",
    "fft_mag_correct_tds",
]


def _compute_mask(freqs, low_cut, high_cut, freq_start, freq_end, sharpness):
    """Build a smooth mask that fades frequencies in or out around the cutoffs."""
    mask = np.ones_like(freqs)
    if low_cut:
        mask *= 1.0 / (1.0 + np.exp(-(freqs - freq_start) * sharpness / 1e11))
    if high_cut:
        mask *= 1.0 / (1.0 + np.exp((freqs - freq_end) * sharpness / 1e11))
    return mask


def apply_frequency_filter(freqs, spectrum, filter_low, filter_high, freq_start, freq_end, sharpness):
    """Multiply a spectrum by the current mask so only in-band content remains."""
    freqs_np = np.asarray(freqs)
    spectrum_np = np.asarray(spectrum)
    mask = _compute_mask(freqs_np, filter_low, filter_high, freq_start, freq_end, sharpness)
    return spectrum_np * mask


def _compute_time_mask(t_s, filter_low, filter_high, t_start, t_end, sharpness):
    """Create a time-domain mask that fades in/out around t_start and t_end."""
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
    """Apply the time-domain mask to 1D or 2D signals to zero-out unwanted tails."""
    sig = np.asarray(signals)
    mask = _compute_time_mask(t_s, filter_low, filter_high, t_start, t_end, sharpness)
    if sig.ndim == 1:
        return sig * mask
    if sig.ndim == 2:
        return sig * mask[None, :]
    raise ValueError("signals must be 1D or 2D")


def fft_mag_correct_tds(signal, axis=-1):
    """Return |rfft(signal)|, i.e., the unscaled magnitude of the real FFT."""
    return np.abs(np.fft.rfft(signal, axis=axis))

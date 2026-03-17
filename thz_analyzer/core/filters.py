import numpy as np

__all__ = [
    "_compute_mask",
    "apply_frequency_filter",
    "_compute_time_mask",
    "apply_time_filter",
    "fft_mag_correct_tds",
    "build_frequency_window",
]


def _compute_mask(freqs, low_cut, high_cut, freq_start, freq_end, sharpness):
    """Build the frequency mask."""
    freqs_np = np.asarray(freqs, dtype=float)
    mask = np.ones_like(freqs_np, dtype=float)
    sharp = max(float(sharpness), 1e-12)

    if low_cut and float(freq_start) > 0.0:
        step_low = float(freq_start) / sharp
        mask *= 0.5 + 0.5 * np.tanh((freqs_np - float(freq_start)) / max(step_low, 1e-30))

    if high_cut and float(freq_end) > 0.0:
        step_high = float(freq_end) / sharp
        mask *= 0.5 - 0.5 * np.tanh((freqs_np - float(freq_end)) / max(step_high, 1e-30))

    return mask


def apply_frequency_filter(freqs, spectrum, filter_low, filter_high, freq_start, freq_end, sharpness):
    """Apply the frequency mask to a spectrum."""
    freqs_np = np.asarray(freqs)
    spectrum_np = np.asarray(spectrum)
    mask = _compute_mask(freqs_np, filter_low, filter_high, freq_start, freq_end, sharpness)
    return spectrum_np * mask


def _compute_time_mask(t_s, filter_low, filter_high, t_start, t_end, sharpness):
    """Build the time mask."""
    t = np.asarray(t_s)
    mask = np.ones_like(t, dtype=float)
    scale = 1e-12

    if filter_low:
        edge_low = 1.0 / (1.0 + np.exp(-(t - t_start) * sharpness / scale))
        mask *= edge_low
        mask[t < t_start] = 0.0

    if filter_high:
        edge_high = 1.0 / (1.0 + np.exp((t - t_end) * sharpness / scale))
        mask *= edge_high
        mask[t > t_end] = 0.0

    return mask


def apply_time_filter(t_s, signals, filter_low, filter_high, t_start, t_end, sharpness):
    """Apply the time mask to one or many signals."""
    sig = np.asarray(signals)
    mask = _compute_time_mask(t_s, filter_low, filter_high, t_start, t_end, sharpness)
    if sig.ndim == 1:
        return sig * mask
    if sig.ndim == 2:
        return sig * mask[None, :]
    raise ValueError("signals must be 1D or 2D")


def fft_mag_correct_tds(signal, axis=-1):
    """Return the real FFT magnitude."""
    return np.abs(np.fft.rfft(signal, axis=axis))


def build_frequency_window(
    n_samples: int,
    dt: float,
    filter_low: bool,
    filter_high: bool,
    freq_start: float,
    freq_end: float,
    sharpness: float,
) -> np.ndarray:
    """Build the FFT window used by the correction code."""
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    window = np.ones(len(freqs), dtype=np.float64)

    if filter_low and freq_start > 0.0:
        step = freq_start / max(sharpness, 1e-12)
        window *= 0.5 + 0.5 * np.tanh((freqs - freq_start) / step)

    if filter_high and freq_end > 0.0:
        step = freq_end / max(sharpness, 1e-12)
        window *= 0.5 - 0.5 * np.tanh((freqs - freq_end) / step)

    return window

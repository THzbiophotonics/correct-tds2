import numpy as np

__all__ = [
    "build_frequency_mask",
    "build_frequency_window",
    "apply_frequency_filter",
    "build_time_mask",
    "apply_time_filter",
    "fft_mag_correct_tds",
]


def build_frequency_mask(
    freqs: np.ndarray,
    filter_low: bool,
    filter_high: bool,
    freq_start: float,
    freq_end: float,
    sharpness: float,
) -> np.ndarray:
    """Build a soft frequency mask from a frequency array.

    Uses a tanh-based rolloff so the transition is smooth (no ringing).
    ``sharpness`` controls how steep the edges are: larger = sharper.
    """
    freqs = np.asarray(freqs, dtype=float)
    mask = np.ones_like(freqs)
    step = max(float(sharpness), 1e-12)

    if filter_low and float(freq_start) > 0.0:
        mask *= 0.5 + 0.5 * np.tanh((freqs - freq_start) / (freq_start / step))

    if filter_high and float(freq_end) > 0.0:
        mask *= 0.5 - 0.5 * np.tanh((freqs - freq_end) / (freq_end / step))

    return mask


def build_frequency_window(
    n_samples: int,
    dt: float,
    filter_low: bool,
    filter_high: bool,
    freq_start: float,
    freq_end: float,
    sharpness: float,
) -> np.ndarray:
    """Build a frequency mask from sample count and time step.

    Convenience wrapper around :func:`build_frequency_mask` for cases
    where the frequency axis has not been computed yet (e.g. inside
    :class:`~thz_analyzer.core.correction.CorrectionModel`).
    """
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    return build_frequency_mask(freqs, filter_low, filter_high, freq_start, freq_end, sharpness)


def apply_frequency_filter(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    filter_low: bool,
    filter_high: bool,
    freq_start: float,
    freq_end: float,
    sharpness: float,
) -> np.ndarray:
    """Multiply a spectrum by the frequency mask."""
    mask = build_frequency_mask(freqs, filter_low, filter_high, freq_start, freq_end, sharpness)
    return np.asarray(spectrum) * mask


def build_time_mask(
    t: np.ndarray,
    filter_low: bool,
    filter_high: bool,
    t_start: float,
    t_end: float,
    sharpness: float,
) -> np.ndarray:
    """Build a soft time-domain mask.

    Uses a sigmoid rolloff. ``sharpness`` is expressed in ps⁻¹ units
    (a value around 2 gives a transition width of roughly a few ps).
    """
    t = np.asarray(t)
    mask = np.ones_like(t, dtype=float)
    scale = 1e-12

    if filter_low:
        mask *= 1.0 / (1.0 + np.exp(-(t - t_start) * sharpness / scale))
        mask[t < t_start] = 0.0

    if filter_high:
        mask *= 1.0 / (1.0 + np.exp((t - t_end) * sharpness / scale))
        mask[t > t_end] = 0.0

    return mask


def apply_time_filter(
    t: np.ndarray,
    signals: np.ndarray,
    filter_low: bool,
    filter_high: bool,
    t_start: float,
    t_end: float,
    sharpness: float,
) -> np.ndarray:
    """Multiply one signal (1D) or a batch (2D) by the time mask."""
    sig = np.asarray(signals)
    mask = build_time_mask(t, filter_low, filter_high, t_start, t_end, sharpness)
    if sig.ndim == 1:
        return sig * mask
    if sig.ndim == 2:
        return sig * mask[None, :]
    raise ValueError("signals must be 1D or 2D")


def fft_mag_correct_tds(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the real-FFT magnitude along the given axis."""
    return np.abs(np.fft.rfft(signal, axis=axis))

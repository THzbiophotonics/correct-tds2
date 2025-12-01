import holoviews as hv
import numpy as np
import panel as pn
from itertools import cycle
from theme import ALASKA_BLUE, ALASKA_PALETTE, ALASKA_PRIMARY


def _to_np(x):
    try:
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def safe_curve(x, y, xlabel, ylabel, label=None, width=900, height=300, title=None, color=None):
    x = _to_np(x)
    y = _to_np(y)
    curve = hv.Curve((x, y), xlabel, ylabel, label=label).opts(width=width, height=height)
    if color:
        curve = curve.opts(color=color)
    if title:
        curve = curve.opts(title=title)
    return curve

def plot_spectrum(freqs, *spectra, db=False, labels=None, title="Spectrum", ylabel=None):
    curves = []
    labels = labels or [f"Series {i + 1}" for i in range(len(spectra))]
    color_cycle = cycle(ALASKA_PALETTE)
    if db:
        spectra = [20 * np.log10(np.maximum(_to_np(s), 1e-16)) for s in spectra]
        ylabel = ylabel or "E [dB]"
    else:
        spectra = [_to_np(s) for s in spectra]
        ylabel = ylabel or "E"
    for y, lab in zip(spectra, labels):
        curves.append(safe_curve(freqs, y, "Frequency [Hz]", ylabel, label=lab, color=next(color_cycle)))
    overlay = hv.Overlay(curves)
    return overlay.opts(title=title) if title else overlay


def plot_phase(freqs, *phases):
    plots = []
    labels = ["Mean", "Ref", "Corrected"]
    color_cycle = cycle(ALASKA_PALETTE)
    for i, p in enumerate(phases):
        label = labels[i] if i < len(labels) else f"Series {i + 1}"
        plots.append(safe_curve(freqs, p, "Frequency [Hz]", "Phase", label=label, color=next(color_cycle)))
    return hv.Overlay(plots) if len(plots) > 1 else plots[0]



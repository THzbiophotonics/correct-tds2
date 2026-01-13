from __future__ import annotations

from itertools import cycle
from typing import Iterable, Sequence

import numpy as np
from bokeh.events import MouseLeave, MouseMove, Tap
from bokeh.layouts import column, row
from bokeh.models import Band, ColumnDataSource, CustomJS, WheelZoomTool
from bokeh.plotting import figure

from theme import (
    ALASKA_BLUE,
    ALASKA_PALETTE,
    ALASKA_PRIMARY,
    ALASKA_SECONDARY,
)

_DEFAULT_HEIGHT = 300
_DEFAULT_WIDTH = None


def _to_np(values) -> np.ndarray:
    """Convert NumPy/JAX arrays to a plain NumPy ndarray."""
    if values is None:
        return None
    try:
        import jax.numpy as jnp  # type: ignore

        if isinstance(values, jnp.ndarray):
            return np.asarray(values)
    except Exception:
        pass
    return np.asarray(values)


def _ensure_source(x, y, **extras) -> ColumnDataSource:
    data = {"x": _to_np(x), "y": _to_np(y)}
    data.update({key: _to_np(val) for key, val in extras.items()})
    return ColumnDataSource(data)


def _iter_legends(p):
    legends = getattr(p, "legend", None)
    if not legends:
        return []
    try:
        legend_items: Iterable = list(legends)
    except TypeError:
        legend_items = [legends]
    return [legend for legend in legend_items if legend is not None]


def _enable_legend_toggle(p):
    """Ensure clicking legend entries hides/shows the corresponding glyph."""
    for legend in _iter_legends(p):
        try:
            legend.click_policy = "hide"
        except Exception:
            continue


def _set_legend_visibility(p, visible):
    """Safely toggle legend visibility when it exists."""
    for legend in _iter_legends(p):
        try:
            legend.visible = visible
        except Exception:
            continue


def _attach_click_wheel_zoom(p):
    """Wheel zoom enabled only after click to avoid stealing page scroll."""
    toolbar = p.toolbar
    wheel = next((tool for tool in toolbar.tools if isinstance(tool, WheelZoomTool)), None)
    if wheel is None:
        wheel = WheelZoomTool()
        p.add_tools(wheel)
    toolbar.active_scroll = None
    p.js_on_event(
        Tap,
        CustomJS(args=dict(toolbar=toolbar, wz=wheel), code="toolbar.active_scroll = wz;"),
    )
    p.js_on_event(
        MouseLeave,
        CustomJS(args=dict(toolbar=toolbar), code="toolbar.active_scroll = null;"),
    )


def _attach_mouse_coordinate_tracker(p, title, x_label, y_label):
    """Attach a lightweight JS tracker to display cursor coordinates in the title."""
    state = ColumnDataSource({"show": [0]})
    base_title = title or ""
    move_cb = CustomJS(
        args=dict(
            plot=p,
            state=state,
            base_title=base_title,
            x_label=x_label or "x",
            y_label=y_label or "y",
        ),
        code="""
            if (!state.data.show[0]) {
                plot.title.text = base_title;
                return;
            }
            const event = cb_obj;
            const x = event.x;
            const y = event.y;
            if (x == null || y == null || !isFinite(x) || !isFinite(y)) {
                plot.title.text = base_title;
                return;
            }
            const fmt = (val) => {
                if (!isFinite(val)) {
                    return "--";
                }
                const absVal = Math.abs(val);
                if (absVal >= 1e3 || absVal <= 1e-3) {
                    return val.toExponential(2);
                }
                if (absVal >= 1) {
                    return val.toFixed(2);
                }
                return val.toPrecision(3);
            };
            plot.title.text = `${base_title} | ${x_label}=${fmt(x)} | ${y_label}=${fmt(y)}`;
        """,
    )
    leave_cb = CustomJS(
        args=dict(plot=p, state=state, base_title=base_title),
        code="""
            if (state.data.show[0]) {
                plot.title.text = base_title;
            }
        """,
    )
    p.js_on_event(MouseMove, move_cb)
    p.js_on_event(MouseLeave, leave_cb)
    setattr(p, "_cursor_coord_state", state)
    setattr(p, "_cursor_coord_title", base_title)


def set_mouse_coordinate_visibility(plot, enabled):
    """Toggle the custom cursor-coordinate overlay for a Bokeh plot."""
    if plot is None:
        return
    state = getattr(plot, "_cursor_coord_state", None)
    base_title = getattr(plot, "_cursor_coord_title", None)
    if state is None:
        return
    state.data = {"show": [1 if enabled else 0]}
    if not enabled and base_title is not None:
        plot.title.text = base_title
    if enabled and base_title is not None:
        # Ensure we start from a clean title until the next mouse event updates it.
        plot.title.text = base_title


def make_figure(
    title,
    x_label,
    y_label,
    height=_DEFAULT_HEIGHT,
    width=_DEFAULT_WIDTH,
    sizing_mode="stretch_width",
):
    figure_kwargs = dict(
        title=title,
        height=height,
        width=width,
        tools="pan,box_zoom,reset,save",
        active_scroll=None,
    )
    if width is None and sizing_mode:
        figure_kwargs["sizing_mode"] = sizing_mode
    p = figure(**figure_kwargs)
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    _attach_click_wheel_zoom(p)
    _attach_mouse_coordinate_tracker(p, title, x_label, y_label)
    return p


def _add_line(p, x, y, *, label=None, color=None, line_dash="solid", line_width=2):
    if y is None:
        return
    source = _ensure_source(x, y)
    kwargs = {}
    if label is not None:
        kwargs["legend_label"] = str(label)
    p.line(
        "x",
        "y",
        source=source,
        line_color=color,
        line_dash=line_dash,
        line_width=line_width,
        **kwargs,
    )
    _enable_legend_toggle(p)


def _add_scatter(p, x, y, *, label=None, color=None, marker="circle", size=6):
    if y is None:
        return
    source = _ensure_source(x, y)
    glyph_args = dict(
        x="x",
        y="y",
        source=source,
        size=size,
        marker=marker,
    )
    if label is not None:
        glyph_args["legend_label"] = str(label)
    if color is not None:
        glyph_args["line_color"] = color
        glyph_args["fill_color"] = color
    p.scatter(**glyph_args)
    _enable_legend_toggle(p)


def _add_band(p, x, lower, upper, *, color, alpha=0.15):
    if lower is None or upper is None:
        return
    source = ColumnDataSource({"x": _to_np(x), "lower": _to_np(lower), "upper": _to_np(upper)})
    band = Band(
        base="x",
        lower="lower",
        upper="upper",
        source=source,
        level="underlay",
        fill_alpha=alpha,
        fill_color=color,
        line_color=None,
    )
    p.add_layout(band)


def _color_sequence():
    return cycle(ALASKA_PALETTE)


def build_filter_plot(x, y, *, title, x_label, y_label, color=ALASKA_PRIMARY):
    p = make_figure(title, x_label, y_label, height=200)
    _add_line(p, x, y, label=None, color=color)
    _set_legend_visibility(p, False)
    return p


def build_time_domain_plot(
    time_axis,
    mean,
    std=None,
    *,
    reference=None,
    corrected_mean=None,
    corrected_std=None,
    title="Time pulses",
    x_label="Time",
    y_label="Amplitude",
):
    color_cycle = _color_sequence()
    p = make_figure(title, x_label, y_label, height=360, sizing_mode="stretch_both")
    mean_color = ALASKA_PRIMARY
    ref_color = ALASKA_BLUE
    corrected_color = ALASKA_SECONDARY
    _add_line(p, time_axis, mean, label="Mean", color=mean_color)
    if std is not None:
        std = _to_np(std)
        mean_arr = _to_np(mean)
        _add_band(p, time_axis, mean_arr - std, mean_arr + std, color=mean_color)
    if reference is not None:
        _add_line(p, time_axis, reference, label="Reference", color=ref_color, line_dash="dashed")
    if corrected_mean is not None:
        _add_line(p, time_axis, corrected_mean, label="Corrected", color=corrected_color)
        if corrected_std is not None:
            corr = _to_np(corrected_mean)
            std_corr = _to_np(corrected_std)
            _add_band(p, time_axis, corr - std_corr, corr + std_corr, color=corrected_color, alpha=0.1)
    return p


def build_time_std_plot(
    time_axis,
    raw_std,
    *,
    corrected_std=None,
    title="Temporal standard deviation",
    x_label="Time",
    y_label="Std",
):
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, time_axis, raw_std, label="Raw", color=ALASKA_PRIMARY)
    if corrected_std is not None:
        _add_line(p, time_axis, corrected_std, label="Corrected", color=ALASKA_BLUE)
    return p

def build_freq_domain_plot( freqs, mean_spec, *, ref_spec=None, corrected_spec=None, title="Spectra", x_label="Frequency [Hz]", y_label="E", ): 
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both") 
    _add_line(p, freqs, mean_spec, label="Mean", color=ALASKA_PRIMARY) 
    if ref_spec is not None: 
        _add_line(p, freqs, ref_spec, label="Reference", color=ALASKA_BLUE) 
    if corrected_spec is not None: _add_line(p, freqs, corrected_spec, label="Corrected", color=ALASKA_SECONDARY) 
    return p


def build_freq_std_plot(
    freqs,
    raw_std,
    *,
    corrected_std=None,
    title="Spectral standard deviation",
    x_label="Frequency [Hz]",
    y_label="Std",
):
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, freqs, raw_std, label="Raw", color=ALASKA_PRIMARY)
    if corrected_std is not None:
        _add_line(p, freqs, corrected_std, label="Corrected", color=ALASKA_BLUE)
    return p


def build_phase_plot(
    freqs,
    mean_phase,
    *,
    ref_phase=None,
    corrected_phase=None,
    title="Phases",
    x_label="Frequency [Hz]",
    y_label="Phase",
):
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, freqs, mean_phase, label="Mean", color=ALASKA_PRIMARY)
    if ref_phase is not None:
        _add_line(p, freqs, ref_phase, label="Reference", color=ALASKA_BLUE)
    if corrected_phase is not None:
        _add_line(p, freqs, corrected_phase, label="Corrected", color=ALASKA_SECONDARY)
    return p


def build_parameter_plot(
    indices,
    values,
    *,
    reference_index=None,
    title="Parameter",
    y_label="Value",
):
    p = make_figure(title, "Trace index", y_label, height=300, width=450)
    _add_line(p, indices, values, label="Value", color=ALASKA_BLUE)
    _add_scatter(p, indices, values, label=None, color=ALASKA_BLUE, marker="circle", size=5)
    if reference_index is not None:
        ref_idx = int(reference_index)
        ref_val = float(_to_np(values)[ref_idx])
        _add_scatter(
            p,
            [ref_idx],
            [ref_val],
            label="Reference",
            color="red",
            marker="triangle",
            size=12,
        )
    return p


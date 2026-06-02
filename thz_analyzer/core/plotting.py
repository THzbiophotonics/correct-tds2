from __future__ import annotations

from typing import Any, Iterable

# Lazy type alias — `from __future__ import annotations` prevents runtime resolution.
BokehFigure = Any

import holoviews as hv
import numpy as np
from bokeh.events import MouseLeave, MouseMove, Tap
from bokeh.models import Band, ColumnDataSource, CustomJS, HoverTool, Span, WheelZoomTool
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.plotting import figure
try:
    from ..theme import (
        ALASKA_BLUE,
        ALASKA_NAVY,
        ALASKA_PRIMARY,
        ALASKA_SECONDARY,
        ALASKA_YELLOW,
    )
except ImportError:
    from theme import (
        ALASKA_BLUE,
        ALASKA_NAVY,
        ALASKA_PRIMARY,
        ALASKA_SECONDARY,
        ALASKA_YELLOW,
    )

_DEFAULT_HEIGHT = 300
_DEFAULT_WIDTH = None


def _to_np(values) -> np.ndarray:
    """Convert NumPy/JAX arrays to a plain NumPy ndarray."""
    if values is None:
        return None
    try:
        import jax.numpy as jnp  # type: ignore
    except ImportError:
        jnp = None  # type: ignore[assignment]
    if jnp is not None and isinstance(values, jnp.ndarray):  # type: ignore[arg-type]
        return np.asarray(values)
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
    if getattr(p, "_click_wheel_zoom_attached", False):
        p.toolbar.active_scroll = None
        return
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
    setattr(p, "_click_wheel_zoom_attached", True)


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


def set_mouse_coordinate_visibility(plot: BokehFigure | None, enabled: bool) -> None:
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
        # Reset the title until the next mouse move.
        plot.title.text = base_title


def make_figure(
    title: str,
    x_label: str,
    y_label: str,
    height: int = _DEFAULT_HEIGHT,
    width: int | None = _DEFAULT_WIDTH,
    sizing_mode: str = "stretch_width",
) -> BokehFigure:
    figure_kwargs = dict(
        title=title,
        height=height,
        width=width,
        tools="pan,box_zoom,reset,save,hover",
        active_scroll=None,
    )
    if width is None and sizing_mode:
        figure_kwargs["sizing_mode"] = sizing_mode
    p = figure(**figure_kwargs)
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    hover_tool = next((tool for tool in p.tools if isinstance(tool, HoverTool)), None)
    if hover_tool is None:
        hover_tool = HoverTool()
        p.add_tools(hover_tool)
    hover_tool.tooltips = [
        (x_label or "x", "$x{0.000e+00}"),
        (y_label or "y", "$y{0.000e+00}"),
    ]
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


def build_filter_plot(
    x: Any,
    y: Any,
    *,
    title: str,
    x_label: str,
    y_label: str,
    color: str = ALASKA_PRIMARY,
    overlay_x: Any = None,
    overlay_y: Any = None,
    overlay_label: str = "Loaded signal (norm)",
    overlay_color: str = ALASKA_BLUE,
) -> BokehFigure:
    p = make_figure(title, x_label, y_label, height=200)
    _add_line(p, x, y, label="Filter mask", color=color, line_width=2)
    if overlay_x is not None and overlay_y is not None:
        _add_line(
            p,
            overlay_x,
            overlay_y,
            label=overlay_label,
            color=overlay_color,
            line_dash="dashed",
            line_width=2,
        )
        _set_legend_visibility(p, True)
    else:
        _set_legend_visibility(p, False)
    p.y_range.start = 0.0
    p.y_range.end = 1.05
    return p


def build_time_domain_plot(
    time_axis: Any,
    mean: Any,
    std: Any = None,
    *,
    reference: Any = None,
    corrected_mean: Any = None,
    corrected_std: Any = None,
    title: str = "Time pulses",
    x_label: str = "Time",
    y_label: str = "Amplitude",
) -> BokehFigure:
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
    time_axis: Any,
    raw_std: Any,
    *,
    corrected_std: Any = None,
    title: str = "Temporal standard deviation",
    x_label: str = "Time",
    y_label: str = "Std",
) -> BokehFigure:
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, time_axis, raw_std, label="Raw", color=ALASKA_PRIMARY)
    if corrected_std is not None:
        _add_line(p, time_axis, corrected_std, label="Corrected", color=ALASKA_BLUE)
    return p


def build_freq_domain_plot(
    freqs: Any,
    mean_spec: Any,
    *,
    ref_spec: Any = None,
    corrected_spec: Any = None,
    title: str = "Spectra",
    x_label: str = "Frequency [Hz]",
    y_label: str = "E",
) -> BokehFigure:
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, freqs, mean_spec, label="Mean", color=ALASKA_PRIMARY)
    if ref_spec is not None:
        _add_line(p, freqs, ref_spec, label="Reference", color=ALASKA_BLUE)
    if corrected_spec is not None:
        _add_line(p, freqs, corrected_spec, label="Corrected", color=ALASKA_SECONDARY)
    return p


def build_freq_std_plot(
    freqs: Any,
    raw_std: Any,
    *,
    corrected_std: Any = None,
    title: str = "Spectral standard deviation",
    x_label: str = "Frequency [Hz]",
    y_label: str = "Std",
) -> BokehFigure:
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, freqs, raw_std, label="Raw", color=ALASKA_PRIMARY)
    if corrected_std is not None:
        _add_line(p, freqs, corrected_std, label="Corrected", color=ALASKA_BLUE)
    return p


def build_phase_plot(
    freqs: Any,
    mean_phase: Any,
    *,
    ref_phase: Any = None,
    corrected_phase: Any = None,
    title: str = "Phases",
    x_label: str = "Frequency [Hz]",
    y_label: str = "Phase",
) -> BokehFigure:
    p = make_figure(title, x_label, y_label, height=320, sizing_mode="stretch_both")
    _add_line(p, freqs, mean_phase, label="Mean", color=ALASKA_PRIMARY)
    if ref_phase is not None:
        _add_line(p, freqs, ref_phase, label="Reference", color=ALASKA_BLUE)
    if corrected_phase is not None:
        _add_line(p, freqs, corrected_phase, label="Corrected", color=ALASKA_SECONDARY)
    return p


def build_parameter_plot(
    indices: Any,
    values: Any,
    *,
    reference_index: int | None = None,
    title: str = "Parameter",
    y_label: str = "Value",
) -> BokehFigure:
    p = make_figure(title, "Trace index", y_label, height=280, sizing_mode="stretch_both")
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


def build_parameter_histogram(
    values,
    *,
    title="Coefficient a histogram",
    x_label="Count",
    y_label="Coefficient a",
    bins="auto",
):
    """Build a histogram for one correction-parameter distribution."""
    p = make_figure(title, x_label, y_label, height=280, sizing_mode="stretch_both")

    values_np = np.asarray(_to_np(values), dtype=np.float64).ravel()
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return p

    if values_np.size == 1:
        center = float(values_np[0])
        width = max(abs(center) * 0.05, 1e-6)
        edges = np.array([center - width, center + width], dtype=np.float64)
        counts = np.array([1.0], dtype=np.float64)
    else:
        if bins == "auto":
            bin_count = int(np.clip(np.sqrt(values_np.size), 8, 40))
        else:
            bin_count = max(1, int(bins))
        counts, edges = np.histogram(values_np, bins=bin_count)

    source = ColumnDataSource(
        data={
            "left": np.zeros_like(counts, dtype=float),
            "right": counts.astype(float),
            "bottom": edges[:-1],
            "top": edges[1:],
        }
    )
    p.quad(
        top="top",
        bottom="bottom",
        left="left",
        right="right",
        source=source,
        fill_color=ALASKA_BLUE,
        fill_alpha=0.55,
        line_color=ALASKA_NAVY,
        line_width=1,
    )

    mean_value = float(np.mean(values_np))
    median_value = float(np.median(values_np))
    p.add_layout(
        Span(
            location=mean_value,
            dimension="width",
            line_color=ALASKA_PRIMARY,
            line_dash="dashed",
            line_width=2,
        )
    )
    p.add_layout(
        Span(
            location=median_value,
            dimension="width",
            line_color=ALASKA_SECONDARY,
            line_dash="dotdash",
            line_width=2,
        )
    )
    p.x_range.start = 0
    return p


_MATRIX_EPS = 1e-20
_MATRIX_MAX_RENDER_SAMPLES = 512
_ALASKA_SEQ_CMAP = [ALASKA_NAVY, ALASKA_BLUE, ALASKA_YELLOW, ALASKA_SECONDARY]
_ALASKA_DIVERGING_CMAP = [ALASKA_BLUE, "#F7F7F7", ALASKA_PRIMARY]


def _coerce_square_matrix_for_plot(values, *, name: str) -> np.ndarray:
    """Coerce to a square float64 matrix without rescanning huge arrays."""
    matrix = np.asarray(_to_np(values), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {matrix.shape}")
    if matrix.size == 0:
        raise ValueError(f"{name} must not be empty")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got {matrix.shape}")
    return matrix


def _prepare_matrix_for_display(
    matrix: np.ndarray,
    *,
    max_render_samples: int = _MATRIX_MAX_RENDER_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Limit matrix display resolution to keep Panel/Bokeh responsive."""
    n = int(matrix.shape[0])
    max_render = max(8, int(max_render_samples))
    if n <= max_render:
        axis = np.arange(n, dtype=np.float64)
        return np.asarray(matrix), axis, None

    stride = max(1, int(np.ceil(n / max_render)))
    sampled = np.asarray(matrix[::stride, ::stride])
    axis = np.arange(sampled.shape[0], dtype=np.float64) * float(stride)
    note = f"x{stride} {sampled.shape[0]}/{n}"
    return sampled, axis, note


def _transform_matrix_for_scale(
    matrix: np.ndarray,
    *,
    scale: str,
    base_label: str,
) -> tuple[np.ndarray, str]:
    """Apply the selected matrix display scale."""
    scale_key = (scale or "log").lower()
    if scale_key == "linear":
        return matrix, base_label
    if scale_key == "log":
        return np.log10(np.abs(matrix) + _MATRIX_EPS), f"log10|{base_label}|"
    if scale_key == "symlog":
        return np.sign(matrix) * np.log10(np.abs(matrix) + 1.0), f"symlog({base_label})"
    raise ValueError("scale must be one of {'linear', 'log', 'symlog'}.")


def _build_matrix_heatmap(
    matrix: np.ndarray,
    *,
    scale: str,
    title: str,
    base_label: str,
    cmap,
) -> hv.Image:
    """Internal helper to build an interactive heatmap with Alaska styling."""
    matrix, axis, display_note = _prepare_matrix_for_display(matrix)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{base_label} contains NaN or inf in displayed samples")
    abs_matrix = np.abs(matrix)
    near_zero_threshold = 1e-10 * max(1.0, float(np.max(abs_matrix)))
    display_sparsity = float(np.mean(abs_matrix <= near_zero_threshold))
    title = f"{title} ({display_sparsity:.1%} sparse)"
    if display_note:
        title = f"{title} [{display_note}]"

    transformed, clabel = _transform_matrix_for_scale(
        matrix,
        scale=scale,
        base_label=base_label,
    )
    image = hv.Image(
        (axis, axis, transformed),
        kdims=["Sample i", "Sample j"],
        vdims=[clabel],
    ).opts(
        cmap=cmap,
        colorbar=True,
        colorbar_opts={"formatter": PrintfTickFormatter(format="%.2e")},
        aspect="equal",
        frame_width=300,
        frame_height=300,
        responsive=False,
        default_tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        tools=["hover"],
        active_tools=[],
        toolbar="above",
        title=title,
        invert_yaxis=True,  # Keep the matrix origin at the top left.
    )
    return image


def build_ncm_heatmap(
    ncm: Any,
    scale: str = "log",
    title: str = "Noise Covariance Matrix",
) -> hv.Image:
    """Build the NCM heatmap."""
    matrix = _coerce_square_matrix_for_plot(ncm, name="ncm")
    return _build_matrix_heatmap(
        matrix,
        scale=scale,
        title=title,
        base_label="NCM",
        cmap=_ALASKA_SEQ_CMAP,
    )


def build_precision_heatmap(
    precision: Any,
    scale: str = "log",
    title: str = "Precision Matrix",
) -> hv.Image:
    """Build the precision heatmap."""
    matrix = _coerce_square_matrix_for_plot(precision, name="precision")
    return _build_matrix_heatmap(
        matrix,
        scale=scale,
        title=title,
        base_label="Precision",
        cmap=_ALASKA_DIVERGING_CMAP,
    )


def build_matrix_value_histogram(
    matrix: Any,
    *,
    scale: str = "log",
    title: str = "Matrix value distribution",
    base_label: str = "Matrix",
    color: str = ALASKA_BLUE,
) -> hv.Histogram:
    """Build a histogram of displayed matrix values."""
    matrix_np = _coerce_square_matrix_for_plot(matrix, name=base_label.lower())
    displayed_matrix, _, display_note = _prepare_matrix_for_display(matrix_np)
    transformed, value_label = _transform_matrix_for_scale(
        displayed_matrix,
        scale=scale,
        base_label=base_label,
    )
    values = np.asarray(transformed, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float64)

    if np.allclose(values, values[0]):
        edges = np.array([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
        counts = np.array([values.size], dtype=np.float64)
    else:
        bins = int(np.clip(np.sqrt(values.size), 16, 80))
        counts, edges = np.histogram(values, bins=bins)

    hist_title = title
    if display_note:
        hist_title = f"{hist_title} [{display_note}]"

    return hv.Histogram(
        (edges, counts),
        kdims=[value_label],
        vdims=["Count"],
    ).opts(
        frame_width=320,
        frame_height=220,
        responsive=False,
        color=color,
        alpha=0.85,
        line_color=ALASKA_NAVY,
        line_width=1,
        default_tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        tools=["hover"],
        active_tools=[],
        toolbar="above",
        title=hist_title,
    )


def _safe_eigvals(values) -> np.ndarray:
    """Return sorted finite eigenvalues (descending), or a safe fallback."""
    arr = np.asarray(values if values is not None else [], dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0], dtype=np.float64)
    return np.sort(arr)[::-1]


def _build_eigen_histogram(
    eigvals: np.ndarray,
    *,
    condition_number: float,
    rank: int,
    title: str,
    color: str,
):
    """Build one eigenvalue histogram with condition-marker overlay."""
    eig_abs = np.abs(eigvals)
    log_eigs = np.log10(eig_abs + _MATRIX_EPS)
    if np.allclose(log_eigs, log_eigs[0]):
        edges = np.array([log_eigs[0] - 0.5, log_eigs[0] + 0.5], dtype=np.float64)
        counts = np.array([log_eigs.size], dtype=np.float64)
    else:
        bins = int(np.clip(np.sqrt(log_eigs.size) * 3, 12, 80))
        counts, edges = np.histogram(log_eigs, bins=bins)
    hist = hv.Histogram(
        (edges, counts),
        kdims=["log10|eigenvalue|"],
        vdims=["Count"],
    ).opts(
        responsive=True,
        color=color,
        alpha=0.85,
        line_color=ALASKA_NAVY,
        line_width=1,
        logy=True,
        default_tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
        tools=["hover"],
        active_tools=[],
        title=f"{title} | rank={int(rank)} | cond={condition_number:.2e}",
    )
    if np.isfinite(condition_number) and condition_number > 0:
        max_abs = float(np.max(eig_abs))
        min_abs_est = max_abs / condition_number
        if np.isfinite(min_abs_est):
            cond_line = hv.VLine(np.log10(min_abs_est + _MATRIX_EPS)).opts(
                color=ALASKA_SECONDARY,
                line_width=2,
                line_dash="dashed",
            )
            return hist * cond_line
    return hist


def build_eigenvalue_plot(diagnostics: dict | None) -> hv.Layout:
    """Build the eigenvalue histograms."""
    diagnostics = diagnostics or {}
    ncm_diag = diagnostics.get("ncm", {}) if isinstance(diagnostics, dict) else {}
    prec_diag = diagnostics.get("precision", {}) if isinstance(diagnostics, dict) else {}

    ncm_eigs = _safe_eigvals(ncm_diag.get("eigenvalues"))
    prec_eigs = _safe_eigvals(prec_diag.get("eigenvalues"))

    ncm_cond = float(ncm_diag.get("condition_number", np.nan))
    prec_cond = float(prec_diag.get("condition_number", np.nan))
    ncm_rank = int(ncm_diag.get("rank", np.sum(np.abs(ncm_eigs) > 1e-12)))
    prec_rank = int(prec_diag.get("rank", np.sum(np.abs(prec_eigs) > 1e-12)))

    ncm_hist = _build_eigen_histogram(
        ncm_eigs,
        condition_number=ncm_cond,
        rank=ncm_rank,
        title="NCM eigenvalues",
        color=ALASKA_BLUE,
    )
    prec_hist = _build_eigen_histogram(
        prec_eigs,
        condition_number=prec_cond,
        rank=prec_rank,
        title="Precision eigenvalues",
        color=ALASKA_PRIMARY,
    )
    return (ncm_hist + prec_hist).cols(2)


def build_matrix_comparison(ncm: Any, precision: Any) -> hv.Layout:
    """Build a 2x2 view of the covariance and precision matrices."""
    ncm_arr = _coerce_square_matrix_for_plot(ncm, name="ncm")
    precision_arr = _coerce_square_matrix_for_plot(precision, name="precision")
    if ncm_arr.shape != precision_arr.shape:
        raise ValueError(
            "ncm and precision must have matching shapes, "
            f"got {ncm_arr.shape} and {precision_arr.shape}."
        )
    ncm_linear = build_ncm_heatmap(ncm_arr, scale="linear", title="Noise Covariance Matrix (linear)")
    ncm_log = build_ncm_heatmap(ncm_arr, scale="log", title="Noise Covariance Matrix (log)")
    prec_linear = build_precision_heatmap(precision_arr, scale="linear", title="Precision Matrix (linear)")
    prec_log = build_precision_heatmap(precision_arr, scale="symlog", title="Precision Matrix (symlog)")
    return (ncm_linear + ncm_log + prec_linear + prec_log).cols(2)

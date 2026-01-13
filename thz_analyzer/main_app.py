import json
from pathlib import Path
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import panel as pn
from theme import (
    ALASKA_BLUE,
    ALASKA_PRIMARY,
    THEME_CSS,
)

pn.extension(raw_css=[THEME_CSS], notifications=True)
pn.config.sizing_mode = "stretch_width"
try:
    pn.state.notifications.position = "bottom-right"
except Exception:
    pass

def _rfft_mag_single_sided(x, axis=-1):
    """
    Compute single-sided magnitude spectrum with proper normalization.
    Returns |FFT| with 2/N scaling (single-sided convention).
    DC and Nyquist bins are scaled by 1/N (no factor 2).
    """
    x = np.asarray(x)
    N = x.shape[axis]
    X = np.fft.rfft(x, axis=axis)
    mag = np.abs(X) * (2.0 / N)
    # DC and Nyquist don't get the factor 2
    if axis == -1:
        mag[..., 0] /= 2.0
        if N % 2 == 0:
            mag[..., -1] /= 2.0
    else:
        slices_dc = [slice(None)] * mag.ndim
        slices_dc[axis] = 0
        mag[tuple(slices_dc)] /= 2.0
        if N % 2 == 0:
            slices_nyq = [slice(None)] * mag.ndim
            slices_nyq[axis] = -1
            mag[tuple(slices_nyq)] /= 2.0
    return mag

# Core modules
from core.io import load_h5_file, save_results
from core.filters import (
    apply_frequency_filter,
    _compute_mask,
    apply_time_filter,
    _compute_time_mask,
    fft_mag_correct_tds,
)
from core.optimization import (
    apply_corrections_batch,
    adam_batch_step,
    batched_gradients,
    batched_losses,
    resolve_device,
    squash_to_bounds,
)
from core.plotting import (
    build_filter_plot,
    build_freq_domain_plot,
    build_freq_std_plot,
    build_parameter_plot,
    build_phase_plot,
    build_time_domain_plot,
    build_time_std_plot,
    set_mouse_coordinate_visibility,
)
from core.periodic_sampling_correct_tds import periodic_sampling_correct_tds



 
CONFIG_FILE = Path("config_filters.json")

class THzOptimizerApp:
    def __init__(self):
        """Initialize the application with default widgets and parameters."""
        # Application state
        self.current_file = None
        self.time = None
        self.time_orig = None
        self.pulses = None
        self.ref_index = None
        self.ref_pulse = None
        self.ref_pulse_time = None
        self.freqs = None
        self.corrected = None
        self.pulses_time_base = None
        self.optimal_params = None
        self.periodic_diagnostics = None
        self.export_payload = {}

        # User interface widgets
        drive_labels = self._available_drives() or ["C:"]
        initial_directory = str(Path.home())
        default_drive = Path.home().anchor.replace("\\", "").replace("/", "")
        if default_drive not in drive_labels:
            default_drive = drive_labels[0]
        self.file_selector = pn.widgets.FileSelector(
            directory=initial_directory,
            file_pattern="*.h5",
            only_files=True,
        )
        self.drive_selector = pn.widgets.Select(
            name="Drive",
            options=drive_labels,
            value=default_drive,
        )
        self.file_area = pn.Column(self.file_selector)
        self.btn_analyze = pn.widgets.Button(name="Analyze (preview)", button_type="primary")
        self.btn_optimize = pn.widgets.Button(name="Optimize (JAX)", button_type="warning", disabled=True)
        self.btn_export = pn.widgets.Button(name="Export results (.txt)", button_type="success", disabled=True)
        self.status = pn.pane.Markdown("No file loaded.")
        self.error_box = pn.pane.Alert("", alert_type="danger", visible=False)
        self.progress = pn.indicators.Progress(value=0, max=100, bar_color="primary", width=900)
        self.log_panel = pn.pane.Markdown(sizing_mode="stretch_width")
        self._std_metric_before = None
        self._std_metric_after = None
        self.extra_metrics = {}
        self._refresh_metric_log()
        # Export message displayed under the Export button
        self.export_msg = pn.pane.Markdown(visible=False)
        self.show_cursor_coords = pn.widgets.Checkbox(name="Show cursor coordinates", value=False)
        self._plots_with_cursor = []

        # Frequency filter configuration
        cfg = self.load_config()
        low_default = bool(cfg.get("filter_low", False))
        high_default = bool(cfg.get("filter_high", False))
        self.filter_low = pn.widgets.Switch(name="Filter lows (< Start)", value=low_default)
        self.filter_high = pn.widgets.Switch(name="Filter highs (> End)", value=high_default)
        self.freq_start = pn.widgets.TextInput(name="Start (Hz)", value=f"{float(cfg.get('freq_start', 0.18e12)):.1e}")
        self.freq_end = pn.widgets.TextInput(name="End (Hz)", value=f"{float(cfg.get('freq_end', 6e12)):.1e}")
        self.sharpness = pn.widgets.FloatInput(name="Sharpness", value=cfg.get("sharpness", 1.0), step=0.1, format="0.0#")
        self.mean_spec_mode = pn.widgets.Select(
            name="Mean spectrum mode",
            options=["FFT(mean)", "mean(FFT)"],
            value="FFT(mean)",
        )
        self.scale_selector = pn.widgets.ToggleGroup(
            name="Scale", options=["Linear", "Log"], behavior="radio", value="Log"
        )
        self.filter_preview = pn.pane.Bokeh(height=180, sizing_mode="stretch_width")
        self.time_filter_preview = pn.pane.Bokeh(height=180, sizing_mode="stretch_width")

        # Correction parameters
        self.cb_delay = pn.widgets.Checkbox(name="Correct delay", value=True)
        self.cb_amplitude = pn.widgets.Checkbox(name="Correct amplitude (a)", value=True)
        self.cb_dilation = pn.widgets.Checkbox(name="Correct dilation (a)", value=False)
        self.limit_delay = pn.widgets.FloatInput(
            name="|delay|max (s)", value=10e-12, step=1e-13
        )
        self.limit_amplitude_a = pn.widgets.FloatInput(name="|amplitude a|max", value=10e-2, step=0.01)
        self.limit_dilation_a = pn.widgets.FloatInput(name="|dilation a|max", value=1e-3, step=0.005)
        # Periodic sampling correction (expert)
        self.cb_periodic = pn.widgets.Checkbox(
            name="Correct periodic sampling?", value=bool(cfg.get("periodic_enable", False))
        )
        self.periodic_freq = pn.widgets.FloatInput(
            name="Frequency [THz]", value=float(cfg.get("periodic_freq_thz", 7.5)), step=0.1, format="0.0#"
        )

        # Optimization parameters
        self.maxiter = pn.widgets.IntInput(name="Iterations (Adam)", value=1000, step=50)
        self.lr = pn.widgets.FloatInput(name="Learning rate", value=0.005, step=0.001)
        self.subsample = pn.widgets.IntSlider(name="Sub-sampling (x)", value=1, start=1, end=8, step=1)
        self.tol = pn.widgets.FloatInput(
            name="Early-stop tolerance", value=3e-5, step=1e-5, width=200
        )

        # Time-domain filtering (applied before the FFT)
        self.tfilter_low = pn.widgets.Switch(name="Filter before tStart", value=bool(cfg.get("tfilter_low", False)))
        self.tfilter_high = pn.widgets.Switch(name="Filter after tEnd", value=bool(cfg.get("tfilter_high", False)))
        self.t_start = pn.widgets.TextInput(name="t Start (s)", value=f"{float(cfg.get('t_start', 0.0)):.1e}")
        self.t_end = pn.widgets.TextInput(name="t End (s)", value=f"{float(cfg.get('t_end', 1e-9)):.1e}")
        self.t_sharpness = pn.widgets.FloatInput(
            name="Time sharpness", value=float(cfg.get("t_sharpness", 2.0)), step=0.1, format="0.0#"
        )

        # Device selection (CPU/GPU)
        self.device_choice = "GPU"
        self.btn_cpu = pn.widgets.Button(name="CPU", button_type="default")
        self.btn_gpu = pn.widgets.Button(name="GPU", button_type="primary")
        self.btn_cpu.on_click(lambda *_: self._select_device("CPU"))
        self.btn_gpu.on_click(lambda *_: self._select_device("GPU"))

        # Plot panes
        responsive_plot_kwargs = dict(sizing_mode="stretch_both", min_height=360)
        spectrum_plot_kwargs = dict(sizing_mode="stretch_both", min_height=320)
        param_plot_kwargs = dict(sizing_mode="stretch_both", min_height=320)
        self.plot_time = pn.pane.Bokeh(**responsive_plot_kwargs)
        self.plot_std_time = pn.pane.Bokeh(**responsive_plot_kwargs)
        self.spectrum_pane = pn.pane.Bokeh(**spectrum_plot_kwargs)
        self.std_spectrum_pane = pn.pane.Bokeh(**spectrum_plot_kwargs)
        self.plot_phase = pn.pane.Bokeh(**responsive_plot_kwargs)
        self.plot_params_delay = pn.pane.Bokeh(**param_plot_kwargs)
        self.plot_params_amp = pn.pane.Bokeh(**param_plot_kwargs)
        self.params_panel = pn.Column(
            self.plot_params_delay,
            self.plot_params_amp,
            sizing_mode="stretch_both",
            min_height=320,
        )

        # Visualization area inspired by the legacy UI (single canvas + buttons)
        self.plot_display_area = pn.Column(
            sizing_mode="stretch_both",
            min_height=420,
            styles={
                "background": "white",
                "border-radius": "10px",
                "box-shadow": "0 2px 10px rgba(15, 23, 42, 0.15)",
                "padding": "15px",
                "min-height": "420px",
                "height": "60vh",
                "max-height": "80vh",
            },
        )
        self.plots = {
            "pulse": (self.plot_time, "Pulse E field", False),
            "pulse_std": (self.plot_std_time, "Std Pulse E field", False),
            "spectrum": (self.spectrum_pane, "E field [dB]", True),
            "spectrum_std": (self.std_spectrum_pane, "Std E field [dB]", True),
            "phase": (self.plot_phase, "Phase", False),
            "params": (self.params_panel, "Correction parameters", False),
        }
        self._plot_order = ["pulse", "pulse_std", "spectrum", "spectrum_std", "phase", "params"]
        toggle_labels = [self.plots[key][1] for key in self._plot_order if key in self.plots]
        default_label = toggle_labels[0] if toggle_labels else None
        self.plot_selector = pn.widgets.ToggleGroup(
            name="Plot selector",
            options=toggle_labels,
            behavior="radio",
            value=default_label,
        )
        self.plot_selector.sizing_mode = "stretch_width"
        self.plot_selector.margin = (10, 0, 5, 0)
        self.scale_selector_row = pn.Row(
            pn.layout.HSpacer(),
            pn.pane.Markdown("Spectrum scale"),
            self.scale_selector,
            sizing_mode="stretch_width",
        )
        self.scale_selector_row.visible = False
        self.visualization_panel = pn.Column(
            pn.pane.Markdown("### Visualization"),
            pn.layout.Spacer(height=5),
            self.plot_display_area,
            pn.layout.Spacer(height=5),
            self.plot_selector,
            self.scale_selector_row,
            sizing_mode="stretch_both",
            min_height=480,
        )

        # Event wiring
        self.file_selector.param.watch(self.on_file_selected, "value")
        self.btn_analyze.on_click(self.preview_analysis)
        self.btn_optimize.on_click(self.run_optimization)
        self.btn_export.on_click(self.export_results)
        for widget in [
            self.filter_low,
            self.filter_high,
            self.freq_start,
            self.freq_end,
            self.sharpness,
            self.tfilter_low,
            self.tfilter_high,
            self.t_start,
            self.t_end,
            self.t_sharpness,
        ]:
            widget.param.watch(self.update_filter_preview, "value")
            widget.param.watch(self.save_config, "value")
        self.scale_selector.param.watch(self.switch_scale, "value")
        if self.plot_selector:
            self.plot_selector.param.watch(self._on_plot_selector_change, "value")
        self.drive_selector.param.watch(self._on_drive_change, "value")
        self.show_cursor_coords.param.watch(self._toggle_cursor_coords, "value")
        # Save config when periodic sampling controls change
        self.cb_periodic.param.watch(self.save_config, "value")
        self.periodic_freq.param.watch(self.save_config, "value")
        if self._plot_order:
            self._select_plot_view(self._plot_order[0], sync_selector=False)

        # Expert options
        self.expert_options = pn.Accordion(
            (
                "Expert options",
                pn.Column(
                    self.cb_delay,
                    self.limit_delay,
                    self.cb_dilation,
                    self.limit_dilation_a,
                    self.cb_periodic,
                    self.periodic_freq,
                ),
            ),
            active=[],
        )

        # Template layout: collapsible file picker + sidebar controls
        self.file_picker = pn.Accordion(
            (
                "Choose a .h5 file",
                pn.Column(
                    pn.Row(self.filter_preview, self.time_filter_preview),
                    self.drive_selector,
                    self.file_area,
                    pn.Row(self.btn_analyze),
                ),
            ),
            active=[],
        )

        tmpl = pn.template.FastListTemplate(
            title="Correct-TDS2",
            theme="default",
            theme_toggle=False,
            sidebar=[
                pn.layout.Divider(),
                pn.pane.Markdown("### Frequency filtering"),
                self.filter_low,
                self.filter_high,
                self.freq_start,
                self.freq_end,
                self.sharpness,
                self.mean_spec_mode,
                pn.layout.Divider(),
                pn.pane.Markdown("### Time filtering"),
                self.tfilter_low,
                self.tfilter_high,
                self.t_start,
                self.t_end,
                self.t_sharpness,
                pn.layout.Divider(),
                pn.pane.Markdown("### Correction"),
                self.cb_amplitude,
                self.limit_amplitude_a,
                self.expert_options,
                pn.layout.Divider(),
                pn.pane.Markdown("### Optimization"),
                pn.Row(self.btn_cpu, self.btn_gpu),
                self.subsample,
                self.maxiter,
                self.lr,
                self.tol,
                pn.Row(self.btn_optimize),
            ],
            main=[
                self.file_picker,
                self.status,
                self.progress,
                self.log_panel,
                pn.Row(self.show_cursor_coords),
                self.error_box,
                self.visualization_panel,
                pn.layout.Divider(),
                pn.Column(pn.Row(self.btn_export), self.export_msg),
            ],
        )
        self.layout = tmpl
        self.update_filter_preview(None)

    def _select_device(self, which: str):
        self.device_choice = "GPU" if str(which).upper() == "GPU" else "CPU"
        # Update button colors
        if self.device_choice == "GPU":
            self.btn_gpu.button_type = "success"
            self.btn_cpu.button_type = "default"
        else:
            self.btn_cpu.button_type = "primary"
            self.btn_gpu.button_type = "default"
        # No extra styling needed: ToggleGroup shows active button pressed

    def load_config(self):
        """Load persisted configuration values from disk."""
        if CONFIG_FILE.exists():
            try:
                return json.loads(CONFIG_FILE.read_text())
            except Exception:
                return {}
        return {}

    def save_config(self, event=None):
        """Write the current configuration to disk."""
        try:
            cfg = dict(
                filter_low=self.filter_low.value,
                filter_high=self.filter_high.value,
                freq_start=float(self.freq_start.value),
                freq_end=float(self.freq_end.value),
                sharpness=self.sharpness.value,
                tfilter_low=self.tfilter_low.value,
                tfilter_high=self.tfilter_high.value,
                t_start=float(self.t_start.value),
                t_end=float(self.t_end.value),
                t_sharpness=float(self.t_sharpness.value),
                periodic_enable=bool(self.cb_periodic.value),
                periodic_freq_thz=float(self.periodic_freq.value),
            )
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        except Exception as e:
            self.show_error(e)

    def _notify(self, message: str, level: str = "info", duration: int = 3000):
        """Show toast notifications when Panel exposes the notification API."""
        try:
            notifications = getattr(pn.state, "notifications", None)
            if notifications is None:
                return
            handler = getattr(notifications, level, None)
            if callable(handler):
                handler(str(message), duration=duration)
            else:
                notifications.info(str(message), duration=duration)
        except Exception:
            pass

    def show_error(self, e: Exception, prefix: str = "Error"):
        """Display an error inside the UI."""
        self.error_box.object = f"{prefix}: {type(e).__name__}\n```\n{traceback.format_exc()}\n```"
        self.error_box.visible = True
        self._notify(f"{prefix}: {type(e).__name__}", level="error")

    def update_filter_preview(self, event):
        """Refresh the frequency/time filter previews."""
        try:
            freqs_np = np.linspace(0, 10e12, 1200)
            mask = _compute_mask(
                freqs_np,
                self.filter_low.value,
                self.filter_high.value,
                float(self.freq_start.value),
                float(self.freq_end.value),
                self.sharpness.value
            )
            freq_fig = build_filter_plot(
                freqs_np,
                mask,
                title="Frequency filter preview",
                x_label="Frequency [Hz]",
                y_label="Transmission",
                color=ALASKA_PRIMARY,
            )
            self._set_plot(self.filter_preview, freq_fig)

            # Use the loaded time axis when available, default to 0-10 ps otherwise
            if self.time is not None:
                t_axis = np.asarray(self.time)
            else:
                t_axis = np.linspace(0, 10e-12, 1200)
            tmask = _compute_time_mask(
                t_axis,
                bool(self.tfilter_low.value),
                bool(self.tfilter_high.value),
                float(self.t_start.value),
                float(self.t_end.value),
                float(self.t_sharpness.value),
            )
            time_fig = build_filter_plot(
                t_axis,
                tmask,
                title="Time filter preview",
                x_label="Time [s]",
                y_label="Transmission",
                color=ALASKA_BLUE,
            )
            self._set_plot(self.time_filter_preview, time_fig)
        except Exception as e:
            self.show_error(e)

    @staticmethod
    def _db_scale(values, floor=1e-12):
        """Convert a spectrum to dB with a numerically stable floor."""
        return 20 * np.log10(np.maximum(values, floor))

    @staticmethod
    def _format_metric_value(value):
        """Format metric entries for the on-screen log."""
        if value is None:
            return "_pending_"
        try:
            val = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(val):
            return "N/A"
        abs_val = abs(val)
        if abs_val >= 1e3 or abs_val <= 1e-3:
            return f"{val:.3e}"
        return f"{val:.3g}"

    def _refresh_metric_log(self):
        """Render the metric log panel with the latest values."""
        before_txt = self._format_metric_value(self._std_metric_before)
        after_txt = self._format_metric_value(self._std_metric_after)
        lines = [
            "### Logs",
            "**Sum of std Pulse E field (sqrt(sum(std^2)))**",
            f"- before correction: {before_txt}",
            f"- after correction: {after_txt}",
        ]
        show_delta = (
            self._std_metric_before is not None
            and self._std_metric_after is not None
            and np.isfinite(self._std_metric_before)
            and np.isfinite(self._std_metric_after)
        )
        if show_delta:
            delta = self._std_metric_after - self._std_metric_before
            lines.append(f"- delta (after - before): {delta:+.4e}")
            if self._std_metric_before != 0:
                rel = (delta / self._std_metric_before) * 100.0
                ratio = self._std_metric_after / self._std_metric_before
                lines.append(f"- relative change: {rel:+.2f}%")
                lines.append(f"- ratio (after / before): {ratio:.3f}")
        if self.extra_metrics:
            lines.append("**Extra metrics**")
            for key in sorted(self.extra_metrics):
                lines.append(f"- {key}: {self._format_metric_value(self.extra_metrics[key])}")
        self.log_panel.object = "\n".join(lines)

    def reset_metrics(self):
        """Reset cached metrics to their initial state."""
        self._std_metric_before = None
        self._std_metric_after = None
        self.extra_metrics.clear()
        self._refresh_metric_log()

    def update_metrics(self, *, before=None, after=None):
        """Update cached std metrics and refresh the log display."""
        if before is not None:
            self._std_metric_before = float(before)
        if after is not None:
            self._std_metric_after = float(after)
        self._refresh_metric_log()

    def _set_metric(self, name, value):
        """Store an auxiliary metric for display."""
        if not name:
            return
        self.extra_metrics[name] = value
        self._refresh_metric_log()

    def _register_plot(self, figure):
        """Keep track of Bokeh figures so cursor overlays can be toggled."""
        if figure is None:
            return None
        if not hasattr(self, "_plots_with_cursor"):
            self._plots_with_cursor = []
        for existing in self._plots_with_cursor:
            if existing is figure:
                set_mouse_coordinate_visibility(figure, bool(self.show_cursor_coords.value))
                return figure
        self._plots_with_cursor.append(figure)
        set_mouse_coordinate_visibility(figure, bool(self.show_cursor_coords.value))
        return figure

    def _set_plot(self, pane, figure):
        """Assign a figure to a Panel pane while registering it for cursor overlays."""
        if pane is None:
            return
        pane.object = figure
        self._register_plot(figure)

    def _toggle_cursor_coords(self, event):
        """Respond to checkbox changes to show/hide cursor coordinates."""
        enabled = bool(event.new) if event else bool(self.show_cursor_coords.value)
        if not hasattr(self, "_plots_with_cursor"):
            return
        for fig in list(self._plots_with_cursor):
            set_mouse_coordinate_visibility(fig, enabled)

    def _on_plot_selector_change(self, event):
        """Update the visualization panel when a different plot button is pressed."""
        if event is None:
            return
        label = event.new
        if not label or not hasattr(self, "plots"):
            return
        for key, (_, cfg_label, _) in self.plots.items():
            if cfg_label == label:
                self._select_plot_view(key, sync_selector=False)
                break

    def _select_plot_view(self, key, *, sync_selector=True):
        """Show a single plot pane in the main visualization area."""
        cfg = self.plots.get(key) if hasattr(self, "plots") else None
        if cfg is None or not hasattr(self, "plot_display_area"):
            return
        pane, label, allow_scale = cfg
        if pane is None:
            return
        self.plot_display_area.objects = [pane]
        show_scale = bool(allow_scale)
        if hasattr(self, "scale_selector_row"):
            self.scale_selector_row.visible = show_scale
        if sync_selector and hasattr(self, "plot_selector"):
            if label and self.plot_selector.value != label:
                self.plot_selector.value = label
        self._active_plot_key = key

    def _filter_config(self):
        """Return the current frequency-filter parameters."""
        return (
            self.filter_low.value,
            self.filter_high.value,
            float(self.freq_start.value),
            float(self.freq_end.value),
            self.sharpness.value,
        )

    def _filter_spectrum(self, freqs, spectrum):
        """Apply the active filter to a spectrum."""
        return apply_frequency_filter(freqs, spectrum, *self._filter_config())

    def _apply_freq_filter_to_pulses(self, pulses_array, freqs):
        """Apply the current frequency mask to a batch of pulses (time domain)."""
        mask = _compute_mask(np.asarray(freqs), *self._filter_config())
        spectrum = np.fft.rfft(pulses_array, axis=1)
        return np.fft.irfft(spectrum * mask, n=pulses_array.shape[1], axis=1)

    def _apply_time_filter_to_pulses(self, pulses_array):
        """Apply the time-domain filter using the current UI configuration."""
        if self.time is None:
            raise ValueError("Time axis is not initialized for filtering.")
        return apply_time_filter(
            self.time,
            pulses_array,
            bool(self.tfilter_low.value),
            bool(self.tfilter_high.value),
            float(self.t_start.value),
            float(self.t_end.value),
            float(self.t_sharpness.value),
        )

    def on_file_selected(self, event):
        """Handle file selection."""
        try:
            if event.new:
                self.current_file = Path(event.new[0])
                self.status.object = f"File selected: `{self.current_file.name}`"
            else:
                self.current_file = None
                self.status.object = "No file selected."
        except Exception as e:
            self.show_error(e)

    def preview_analysis(self, event):
        """Load the selected file, compute preview metrics, and plot diagnostics."""
        try:
            self.error_box.visible = False
            self.progress.value = 0
            self.export_msg.visible = False
            self.export_msg.object = ""
            if not self.current_file:
                self.status.object = "Select an .h5 file first."
                return

            self.reset_metrics()
            self.status.object = "Loading HDF5..."
            self.progress.value = 5
            data = load_h5_file(self.current_file)
            if data is None:
                self.status.object = "Failed to load the file."
                return

            pulses_list, timeaxis = data
            min_len = min(map(len, pulses_list))
            pulses_array = np.vstack([p[:min_len] for p in pulses_list])
            t_orig = timeaxis[:min_len]

            # Convert time axis to seconds when needed
            t = t_orig.astype(float)
            dt_raw = float(t[1] - t[0])
            scale_to_s = 1e-12 if dt_raw > 1e-5 else 1.0
            t_s = t * scale_to_s
            self.time = t_s
            dt_fft = t_s[1] - t_s[0]
            freqs = np.fft.rfftfreq(min_len, d=dt_fft)

            self.progress.value = 15
            self.status.object = (
                f"Pre-processing... dt_raw={dt_raw:g}, scale_to_s={scale_to_s}, "
                f"fmax={freqs.max()/1e12:.2f} THz"
            )

            # Optional time-domain filtering
            pulses_time_filtered = self._apply_time_filter_to_pulses(pulses_array)

            # Reference trace: closest pulse to the mean (normalized dot product)
            mean_pulse_time = pulses_time_filtered.mean(axis=0)
            self.periodic_diagnostics = None
            if bool(self.cb_periodic.value):
                try:
                    freq_limit_thz = float(self.periodic_freq.value)
                except Exception:
                    freq_limit_thz = 0.0
                try:
                    periodic_diag = periodic_sampling_correct_tds(
                        mean_pulse_time,
                        self.time,
                        freq_limit_thz,
                    )
                    correction_wave = np.asarray(periodic_diag.get("correction_waveform"))
                    if correction_wave.shape != mean_pulse_time.shape:
                        raise ValueError("Periodic correction has incompatible shape.")
                    pulses_time_filtered = pulses_time_filtered - correction_wave
                    mean_pulse_time = np.asarray(periodic_diag.get("corrected_signal"))
                    self.periodic_diagnostics = periodic_diag
                    params = periodic_diag.get("params", {})
                    self._set_metric("periodic_A", params.get("A"))
                    self._set_metric("periodic_omega", params.get("omega"))
                    self._set_metric("periodic_phi", params.get("phi"))
                    self._set_metric("periodic_error", periodic_diag.get("error"))
                except Exception as err:
                    self.show_error(err, prefix="Periodic sampling correction")

            # Apply frequency-domain filtering before metrics/optimization
            pulses_freq_filtered = self._apply_freq_filter_to_pulses(pulses_time_filtered, freqs)
            mean_pulse_filtered = pulses_freq_filtered.mean(axis=0)
            std_metric_before = float(np.linalg.norm(pulses_freq_filtered.std(axis=0)))
            self.update_metrics(before=std_metric_before)

            proj = pulses_time_filtered @ mean_pulse_time
            norms = np.einsum('ij,ij->i', pulses_time_filtered, pulses_time_filtered)
            ref_idx = int(np.argmin(np.abs(proj / (norms + 1e-30) - 1)))
            ref_pulse_filtered = pulses_freq_filtered[ref_idx]
            ref_pulse_time = pulses_time_filtered[ref_idx]
            self._notify(f"Reference pulse index selected: {ref_idx}", level="info")
            self._set_metric("ref_idx", ref_idx)

            self.progress.value = 25
            self.status.object = "Spectra & phases (preview)..."

            # FFT of all traces (magnitudes without normalization) before applying the mask
            fft_all = np.fft.rfft(pulses_time_filtered, axis=1)
            mag_all = fft_mag_correct_tds(pulses_time_filtered, axis=1)

            # --- Mean spectrum, selectable computation ---
            if self.mean_spec_mode.value == "FFT(mean)":
                mean_spec_raw = fft_mag_correct_tds(mean_pulse_time)
            else:
                mean_spec_raw = mag_all.mean(axis=0)
            ref_spec_raw = fft_mag_correct_tds(ref_pulse_time)

            # Magnitude spectral standard deviation 
            std_spec_raw = np.abs(np.std(fft_all, axis=0))

            # Apply the mask only once for display
            mean_spec_f = self._filter_spectrum(freqs, mean_spec_raw)
            ref_spec_f = self._filter_spectrum(freqs, ref_spec_raw)
            std_spec_f = self._filter_spectrum(freqs, std_spec_raw)

            # Phases
            # Phases computed after time-filter for consistency
            fft_mean = np.fft.rfft(mean_pulse_filtered)
            fft_ref = np.fft.rfft(ref_pulse_filtered)
            mean_phase = np.unwrap(np.angle(fft_mean))
            ref_phase = np.unwrap(np.angle(fft_ref))

            # --- Persist computed state ---
            self.time = t_s
            self.time_orig = t_orig
            self.freqs = freqs
            self.pulses = pulses_freq_filtered
            self.pulses_time_base = pulses_time_filtered
            self.ref_index = ref_idx
            self.ref_pulse = ref_pulse_filtered
            self.ref_pulse_time = ref_pulse_time
            self.corrected = None
            self.export_payload = {}

            # --- Graphiques preview ---
            time_std = pulses_freq_filtered.std(axis=0)
            self._set_plot(
                self.plot_time,
                build_time_domain_plot(
                    t_orig,
                    mean_pulse_filtered,
                    std=time_std,
                    reference=ref_pulse_filtered,
                    title="Time pulses - Mean / Ref",
                    x_label="Time [orig units]",
                    y_label="Amp",
                ),
            )

            self._set_plot(
                self.plot_std_time,
                build_time_std_plot(
                    t_orig,
                    time_std,
                    title="Temporal standard deviation (raw)",
                    x_label="Time [orig units]",
                    y_label="Std",
                ),
            )

            self._spec_lin = self._register_plot(
                build_freq_domain_plot(
                    freqs,
                    mean_spec_f,
                    ref_spec=ref_spec_f,
                    title="Spectra (linear)",
                    y_label="E",
                )
            )

            self._spec_log = self._register_plot(
                build_freq_domain_plot(
                    freqs,
                    self._db_scale(mean_spec_f),
                    ref_spec=self._db_scale(ref_spec_f),
                    title="Spectra (log)",
                    y_label="E [dB]",
                )
            )

            self._std_lin = self._register_plot(
                build_freq_std_plot(
                    freqs,
                    std_spec_f,
                    title="Spectral std dev (linear)",
                    y_label="Std",
                )
            )

            self._std_log = self._register_plot(
                build_freq_std_plot(
                    freqs,
                    self._db_scale(std_spec_f),
                    title="Spectral std dev (log)",
                    y_label="Std [dB]",
                )
            )

            self._set_plot(
                self.plot_phase,
                build_phase_plot(
                    freqs,
                    mean_phase,
                    ref_phase=ref_phase,
                    title="Phases",
                    y_label="Phase",
                ),
            )

            self.progress.value = 40
            self.switch_scale(None)
            self.status.object = f"Preview ready - {pulses_array.shape[0]} traces, ref #{ref_idx}"
            self.btn_optimize.disabled = False
            self.btn_export.disabled = False
            self._notify("Preview completed successfully.", level="success")

        except Exception as e:
            self.show_error(e, prefix="Preview")
            self.status.object = "Error during preview"


    def switch_scale(self, event):
        """Toggle between linear and logarithmic displays."""
        if not hasattr(self, "_spec_lin"):
            return
        try:
            if self.scale_selector.value == "Linear":
                self._set_plot(self.spectrum_pane, self._spec_lin)
                self._set_plot(self.std_spectrum_pane, self._std_lin)
            else:
                self._set_plot(self.spectrum_pane, self._spec_log)
                self._set_plot(self.std_spectrum_pane, self._std_log)
        except Exception as e:
            self.show_error(e)

    def build_bounds(self):
        """Build parameter bounds for the correction step."""
        lo = np.array([0.0, 0.0, 0.0], dtype=float)
        hi = np.array([0.0, 0.0, 0.0], dtype=float)
        if self.cb_delay.value:
            lo[0], hi[0] = -self.limit_delay.value, self.limit_delay.value
        if self.cb_amplitude.value:
            lo[1], hi[1] = -self.limit_amplitude_a.value, self.limit_amplitude_a.value
        if self.cb_dilation.value:
            lo[2], hi[2] = -self.limit_dilation_a.value, self.limit_dilation_a.value
        return jnp.asarray(lo, dtype=jnp.float32), jnp.asarray(hi, dtype=jnp.float32)

    def run_optimization(self, event):
        """Launch the JAX-based correction loop on the current dataset."""
        try:
            if self.pulses is None:
                self.status.object = "Run the preview first."
                return
            self.export_msg.visible = False
            self.export_msg.object = ""

            # Progress tracking initialization
            self.error_box.visible = False
            self.progress.value = 45
            self.status.object = "Initializing optimization (JAX)..."

            # === Automatic device selection (CPU/GPU) ===
            requested_device = self.device_choice
            try:
                computation_device, exact_match = resolve_device(requested_device)
                if exact_match:
                    self.status.object = f"Computing on {computation_device.platform.upper()}"
                else:
                    self.status.object = "GPU unavailable - using CPU"
            except RuntimeError as err:
                self.show_error(err, prefix="Device selection")
                self.status.object = "No JAX device available"
                return

            # === Hyperparameters and subsampling ===
            base_dt = float(self.time[1] - self.time[0])
            effective_dt = base_dt
            subsample_factor = max(1, int(self.subsample.value))
            subsample_slice = slice(None, None, subsample_factor)
            freq_subsampled_dt = effective_dt * subsample_factor
            num_iterations = int(self.maxiter.value)
            learning_rate = float(self.lr.value)
            early_stop_tolerance = float(self.tol.value)

            # === Prepare tensors on the selected device ===
            with jax.default_device(computation_device):
                time_vector = jnp.asarray(self.time, dtype=jnp.float32)
                angular_frequencies = (
                    jnp.fft.rfftfreq(self.time.shape[0], d=effective_dt).astype(jnp.float32) * (2 * jnp.pi)
                )
                reference_pulse = jnp.asarray(self.ref_pulse, dtype=jnp.float32)
                all_pulses = jnp.asarray(self.pulses, dtype=jnp.float32)

                num_traces = all_pulses.shape[0]
                lower_bounds, upper_bounds = self.build_bounds()

                sub_time_vector = time_vector[subsample_slice]
                sub_angular_frequencies = (
                    jnp.fft.rfftfreq(sub_time_vector.shape[0], d=freq_subsampled_dt).astype(jnp.float32) * (2 * jnp.pi)
                )
                subsampled_pulses = all_pulses[:, subsample_slice]
                subsampled_reference = reference_pulse[subsample_slice]

                parameter_matrix = jnp.zeros((num_traces, 3), dtype=jnp.float32)
                adam_momentum = jnp.zeros_like(parameter_matrix)
                adam_velocity = jnp.zeros_like(parameter_matrix)

            previous_mean_loss = np.inf

            # === Vectorized optimization loop ===
            import time as _t
            # JIT warm-up (excluded from timing)
            try:
                warmup_grads = batched_gradients(
                    parameter_matrix,
                    subsampled_pulses,
                    subsampled_reference,
                    sub_time_vector,
                    sub_angular_frequencies,
                    lower_bounds,
                    upper_bounds,
                )
                _ = adam_batch_step(
                    parameter_matrix,
                    adam_momentum,
                    adam_velocity,
                    warmup_grads,
                    jnp.array(1, dtype=jnp.float32),
                    jnp.array(learning_rate, dtype=jnp.float32),
                )
            except Exception:
                pass

            start_timestamp = _t.perf_counter()

            for iteration in range(1, num_iterations + 1):
                gradients_matrix = batched_gradients(
                    parameter_matrix,
                    subsampled_pulses,
                    subsampled_reference,
                    sub_time_vector,
                    sub_angular_frequencies,
                    lower_bounds,
                    upper_bounds,
                )
                parameter_matrix, adam_momentum, adam_velocity = adam_batch_step(
                    parameter_matrix, adam_momentum, adam_velocity,
                    gradients_matrix, jnp.array(iteration, dtype=jnp.float32), learning_rate
                )

                if iteration % 10 == 0:
                    current_losses = batched_losses(
                        parameter_matrix,
                        subsampled_pulses,
                        subsampled_reference,
                        sub_time_vector,
                        sub_angular_frequencies,
                        lower_bounds,
                        upper_bounds,
                    )
                    mean_loss = float(jnp.mean(current_losses))
                    self.progress.value = min(95, int(45 + 45 * (iteration / num_iterations)))
                    self.status.object = f"JAX it {iteration}/{num_iterations} - mean loss {mean_loss:.3e}"

                    # Early-stop condition
                    if abs(previous_mean_loss - mean_loss) < early_stop_tolerance:
                        self.status.object += " Early stop achieved"
                        break
                    previous_mean_loss = mean_loss

            # === Final application of the corrections ===
            optimal_parameters = squash_to_bounds(parameter_matrix, lower_bounds, upper_bounds)
            corrected_pulses = apply_corrections_batch(
                all_pulses,
                time_vector,
                angular_frequencies,
                optimal_parameters,
            )

            compute_end = _t.perf_counter()
            compute_duration = compute_end - start_timestamp

            self.corrected = np.asarray(corrected_pulses)
            self.optimal_params = np.asarray(optimal_parameters)

            # === Update plots ===
            self.update_plots_after_correction()

            total_end = _t.perf_counter()
            plot_duration = total_end - compute_end
            total_duration = compute_duration + plot_duration

            # === Success message ===
            self.status.object = (
                f"Optimization finished on {computation_device.device_kind.upper()} - "
                f"compute {compute_duration:.2f}s, plots {plot_duration:.2f}s, total {total_duration:.2f}s"
            )
            self.progress.value = 100
            self.btn_export.disabled = False
            self._notify("Optimization finished successfully.", level="success")

        except Exception as e:
            self.show_error(e, prefix="Optimization")
            self.status.object = "Error during optimization"


    def update_plots_after_correction(self):
        """Refresh every plot using the newly corrected pulses."""
        try:
            if self.corrected is None:
                return
            pulses_array = self.pulses_time_base if self.pulses_time_base is not None else self.pulses
            ref_array = self.ref_pulse_time if self.ref_pulse_time is not None else self.ref_pulse
            t_orig = self.time_orig
            freqs = self.freqs
            if self.pulses_time_base is not None:
                pulses_tf = pulses_array
                ref_tf = ref_array
            else:
                pulses_tf = self._apply_time_filter_to_pulses(pulses_array)
                ref_tf = self._apply_time_filter_to_pulses(ref_array)
            corrected_tf = self._apply_time_filter_to_pulses(self.corrected)
            pulses_freq = self._apply_freq_filter_to_pulses(pulses_tf, freqs)
            corrected_freq = corrected_tf
            ref_freq = self._apply_freq_filter_to_pulses(ref_tf[None, :], freqs)[0]

            mean_raw = pulses_freq.mean(axis=0)
            mean_corr = corrected_freq.mean(axis=0)

            raw_std_time = pulses_freq.std(axis=0)
            corrected_std_time = corrected_freq.std(axis=0)
            before_metric = (
                float(np.linalg.norm(raw_std_time))
                if self._std_metric_before is None
                else self._std_metric_before
            )
            after_metric = float(np.linalg.norm(corrected_std_time))
            self.update_metrics(before=before_metric, after=after_metric)

            self._set_plot(
                self.plot_time,
                build_time_domain_plot(
                    t_orig,
                    mean_raw,
                    std=raw_std_time,
                    reference=ref_freq,
                    corrected_mean=mean_corr,
                    corrected_std=corrected_std_time,
                    title="Time pulses - Mean / Ref / Corrected",
                    x_label="Time [orig units]",
                    y_label="Amp",
                ),
            )

            self._set_plot(
                self.plot_std_time,
                build_time_std_plot(
                    t_orig,
                    raw_std_time,
                    corrected_std=corrected_std_time,
                    title="Temporal std dev",
                    x_label="Time [orig units]",
                    y_label="Std",
                ),
            )

            fft_raw_plot = np.fft.rfft(pulses_freq, axis=1)
            fft_corr_plot = np.fft.rfft(corrected_freq, axis=1)
            mean_spec_raw_f = fft_mag_correct_tds(mean_raw)
            mean_spec_corr_f = fft_mag_correct_tds(mean_corr)
            ref_spec_f = fft_mag_correct_tds(ref_freq)

            std_spec_raw_f = np.abs(np.std(fft_raw_plot, axis=0))
            std_spec_corr_f = np.abs(np.std(fft_corr_plot, axis=0))

            self._spec_lin = self._register_plot(
                build_freq_domain_plot(
                    freqs,
                    mean_spec_raw_f,
                    ref_spec=ref_spec_f,
                    corrected_spec=mean_spec_corr_f,
                    title="Spectra (linear)",
                    y_label="E",
                )
            )
            self._spec_log = self._register_plot(
                build_freq_domain_plot(
                    freqs,
                    self._db_scale(mean_spec_raw_f),
                    ref_spec=self._db_scale(ref_spec_f),
                    corrected_spec=self._db_scale(mean_spec_corr_f),
                    title="Spectra (log)",
                    y_label="E [dB]",
                )
            )

            self._std_lin = self._register_plot(
                build_freq_std_plot(
                    freqs,
                    std_spec_raw_f,
                    corrected_std=std_spec_corr_f,
                    title="Spectral std dev (linear)",
                    y_label="Std",
                )
            )
            self._std_log = self._register_plot(
                build_freq_std_plot(
                    freqs,
                    self._db_scale(std_spec_raw_f),
                    corrected_std=self._db_scale(std_spec_corr_f),
                    title="Spectral std dev (log)",
                    y_label="Std [dB]",
                )
            )

            # Phases
            # Phases computed after time-filter for consistency
            phase_mean_raw = np.unwrap(np.angle(np.fft.rfft(mean_raw)))
            phase_ref = np.unwrap(np.angle(np.fft.rfft(ref_freq)))
            phase_mean_corr = np.unwrap(np.angle(np.fft.rfft(mean_corr)))

            self._set_plot(
                self.plot_phase,
                build_phase_plot(
                    freqs,
                    phase_mean_raw,
                    ref_phase=phase_ref,
                    corrected_phase=phase_mean_corr,
                    title="Phases",
                    y_label="Phase",
                ),
            )
            self.switch_scale(None)

            # --- Correction parameters (Delay and amplitude coef a) ---
            if self.optimal_params is not None and len(self.optimal_params) == pulses_array.shape[0]:
                idx = np.arange(self.optimal_params.shape[0])
                delays = self.optimal_params[:, 0]
                coef_a = self.optimal_params[:, 1]
                self._set_plot(
                    self.plot_params_delay,
                    build_parameter_plot(
                        idx,
                        delays,
                        reference_index=self.ref_index,
                        title="Delay",
                        y_label="Delay [s]",
                    ),
                )
                self._set_plot(
                    self.plot_params_amp,
                    build_parameter_plot(
                        idx,
                        coef_a,
                        reference_index=self.ref_index,
                        title="Coef a - amplitude",
                        y_label="Coefficient a",
                    ),
                )

            # --- Prepared data for export ---
            self.export_payload = {
                "corrected_mean": (t_orig, mean_corr),
                "corrected_std_time": (t_orig, corrected_std_time),
                "corrected_std_freq": (freqs, std_spec_corr_f),
            }

        except Exception as e:
            self.show_error(e, prefix="Plot update")

    def export_results(self, event):
        """Save corrected data and metrics as TXT files."""
        try:
            if not self.export_payload:
                raise ValueError("No corrected data available for export.")
            if self.current_file is None:
                raise ValueError("No source file associated with the export.")
            target_dir = Path(Path(self.current_file).name).with_suffix("")
            written_files = save_results(self.export_payload, target_dir)
            file_names = ", ".join(p.name for p in written_files)
            resolved = target_dir.resolve()
            self.status.object = f"TXT exports saved in {resolved}: {file_names}"
            self.export_msg.object = f"Exported files to `{resolved}`: {file_names}"
            self.export_msg.visible = True
            self._notify("Export completed successfully.", level="success")
        except Exception as e:
            self.export_msg.visible = False
            self.show_error(e, prefix="Export")

    def _available_drives(self):
        """Return a list of available drive labels like ['C:', 'D:'].""" 
        drives = []
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{letter}:/")
            if root.exists():
                drives.append(f"{letter}:")
        return drives

    def _on_drive_change(self, event):
        """When the user selects another drive, recreate a fresh FileSelector."""
        try:
            new_drive = event.new
            if not new_drive:
                return
            new_dir = f"{new_drive}/"
            new_fs = pn.widgets.FileSelector(
                directory=new_dir,
                file_pattern="*.h5",
                only_files=True,
            )
            new_fs.param.watch(self.on_file_selected, "value")
            self.file_selector = new_fs
            self.file_area[0] = new_fs
            self.status.object = f"Browsing drive `{new_drive}` ({new_dir})"
        except Exception as e:
            self.show_error(e, prefix="Drive change")

    def show(self):
        """Return the Panel layout."""
        return self.layout

app = THzOptimizerApp()
app.show().servable()


    

import json
import gc
import os
import sys
from pathlib import Path
import traceback

# --no-preallocate (passed via `panel serve ... --args --no-preallocate`)
# must be handled before `import jax` below - JAX only reads
# XLA_PYTHON_CLIENT_PREALLOCATE at import time, it can't be toggled once
# the module is already loaded.
if "--no-preallocate" in sys.argv:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# JAX configuration — must happen before any JAX computation.
# x64: enables float64 (required for numerically stable NCM inversion).
# persistent_cache: saves XLA compilations to disk so restarts skip recompilation.
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax_xla_thz"))

import jax.numpy as jnp
import numpy as np
import panel as pn
from theme import (
    ALASKA_BLUE,
    ALASKA_PRIMARY,
    THEME_CSS,
)
from core.io import PulseDataset, load_h5_file, save_results
from core.batching import BatchLinker
from core.estimator import CovarianceEstimator
from core.filters import (
    apply_frequency_filter,
    build_frequency_mask,
    apply_time_filter,
    build_time_mask,
    fft_mag_correct_tds,
)
from core.jax_ops import (
    apply_batch_corrections,
    batch_gradients,
    batch_losses,
    compute_superresolution_npadded,
    LASER_FREP_HZ,
    rfft_angular_freqs,
    squash_to_bounds,
)
from core.optimizer import (
    infer_initial_lr,
    make_optax_optimizer,
    optax_step,
    optax_train_block,
)
from core.correction import CorrectionModel
from core.covariance import (
    compute_precision_matrix,
    compute_covariance_diagnostics,
)
from core.plotting import (
    build_filter_plot,
    build_freq_domain_plot,
    build_freq_std_plot,
    build_ncm_heatmap,
    build_precision_heatmap,
    build_matrix_value_histogram,
    build_eigenvalue_plot,
    build_parameter_plot,
    build_parameter_histogram,
    build_phase_plot,
    build_time_domain_plot,
    build_time_std_plot,
    set_mouse_coordinate_visibility,
)
from core.periodic_sampling_correct_tds import periodic_sampling_correct_tds

pn.extension(raw_css=[THEME_CSS], notifications=True)
pn.config.sizing_mode = "stretch_width"
notifications = getattr(pn.state, "notifications", None)
if notifications is not None:
    notifications.position = "bottom-right"
CONFIG_FILE = Path("config_filters.json")


def _warmup_jax() -> None:
    """Trigger JAX startup with tiny arrays so the first real computation is fast."""
    try:
        _t = jnp.linspace(0.0, 1e-12, 16)
        _p = jnp.zeros((4, 3))
        _pulses = jnp.zeros((4, 16))
        _ref = jnp.zeros(16)
        _om = rfft_angular_freqs(_t)
        _lo = jnp.array([-1e-12, -0.1, -0.1])
        _hi = jnp.array([1e-12, 0.1, 0.1])
        batch_gradients(_p, _pulses, _ref, _t, _om, _lo, _hi).block_until_ready()
    except Exception:
        pass  # warmup is best-effort; never crash the app startup


_warmup_jax()

class THzOptimizerApp:
    @staticmethod
    def _make_h5_selector(directory: str):
        return pn.widgets.FileSelector(
            directory=directory,
            file_pattern="*.h5",
            only_files=True,
        )

    def _build_file_widgets(self):
        """Create file selection widgets."""
        drive_labels = self._available_drives() or ["C:"]
        initial_directory = str(Path.home())
        default_drive = Path.home().anchor.replace("\\", "").replace("/", "")
        if default_drive not in drive_labels:
            default_drive = drive_labels[0]

        self.file_selector = self._make_h5_selector(initial_directory)
        self.reference_file_selector = self._make_h5_selector(initial_directory)
        self.drive_selector = pn.widgets.Select(
            name="Drive",
            options=drive_labels,
            value=default_drive,
        )
        self.sample_file_display = pn.pane.Markdown("**Sample selected:** _none_")
        self.reference_file_display = pn.pane.Markdown("**Reference selected:** _none_")
        self.sample_selector_panel = pn.Column(
            self.file_selector,
            self.sample_file_display,
            sizing_mode="stretch_width",
        )
        self.reference_selector_panel = pn.Column(
            self.reference_file_selector,
            self.reference_file_display,
            sizing_mode="stretch_width",
        )
        self.batch_file_selector = self._make_h5_selector(initial_directory)
        self.batch_selected_display = pn.pane.Markdown("**Batch files:** _none_")
        self.btn_batch_autolink = pn.widgets.Button(name="Auto-link sample/ref", button_type="primary")
        self.batch_mapping_editor = pn.widgets.TextAreaInput(
            name="Batch mapping (`sample ==> ref`)",
            value="",
            placeholder=(
                "# One job per line: sample_path ==> reference_path(optional)\n"
                "# Reference-only line allowed: ==> ref_path"
            ),
            height=220,
            sizing_mode="stretch_width",
        )
        self.btn_batch_run = pn.widgets.Button(name="Run batch (sequential)", button_type="warning")
        self.batch_status = pn.pane.Markdown("Batch idle.")
        self.batch_panel = pn.Column(
            pn.pane.Markdown(
                "Select multiple `.h5` files, auto-link by similar names, then edit if needed."
            ),
            self.batch_file_selector,
            self.batch_selected_display,
            pn.Row(self.btn_batch_autolink, self.btn_batch_run),
            self.batch_mapping_editor,
            self.batch_status,
            sizing_mode="stretch_width",
        )
        self.file_area = pn.Tabs(
            ("Sample file (.h5)", self.sample_selector_panel),
            ("Reference file — optional", self.reference_selector_panel),
            ("Batch mode", self.batch_panel),
            dynamic=True,
        )

    def _build_control_widgets(self):
        """Create control buttons and status."""
        self.btn_analyze = pn.widgets.Button(name="Analyze (preview)", button_type="primary")
        self.btn_optimize = pn.widgets.Button(name="Optimize (JAX)", button_type="warning", disabled=True)
        self.btn_export = pn.widgets.Button(name="Export results (.txt)", button_type="success", disabled=True)
        self.status = pn.pane.Markdown("No file loaded.")
        self.error_box = pn.pane.Alert("", alert_type="danger", visible=False)
        self.progress = pn.indicators.Progress(value=0, max=100, bar_color="primary", width=900)
        self.log_panel = pn.pane.Markdown(sizing_mode="stretch_width")
        self.export_msg = pn.pane.Markdown(visible=False)
        self.show_cursor_coords = pn.widgets.Checkbox(name="Show cursor coordinates", value=False)

        self._std_metric_before = None
        self._std_metric_after = None
        self.extra_metrics = {}
        self._refresh_metric_log()
        self._plots_with_cursor = set()

    def _build_filter_widgets(self):
        """Create frequency and time filter widgets."""
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
        self.superresolution_toggle = pn.widgets.Toggle(
            name="⚡ Superresolution mode",
            value=False,
            width=200,
        )
        self.superresolution_info = pn.pane.Markdown("", width=350)

        # Time-domain filtering (applied before the FFT)
        self.tfilter_low = pn.widgets.Switch(name="Filter before tStart", value=bool(cfg.get("tfilter_low", False)))
        self.tfilter_high = pn.widgets.Switch(name="Filter after tEnd", value=bool(cfg.get("tfilter_high", False)))
        self.t_start = pn.widgets.TextInput(name="t Start (s)", value=f"{float(cfg.get('t_start', 0.0)):.1e}")
        self.t_end = pn.widgets.TextInput(name="t End (s)", value=f"{float(cfg.get('t_end', 1e-9)):.1e}")
        self.t_sharpness = pn.widgets.FloatInput(
            name="Time sharpness", value=float(cfg.get("t_sharpness", 2.0)), step=0.1, format="0.0#"
        )
        return cfg

    def _build_ncm_widgets(self, cfg=None):
        """Create NCM computation widgets."""
        cfg = cfg or {}
        ncm_time_mode = str(cfg.get("ncm_time_sampling_mode", "Max"))
        if ncm_time_mode not in ("Max", "Custom", "Sans"):
            ncm_time_mode = "Max"
        try:
            ncm_max_samples_cfg = int(cfg.get("ncm_max_samples", 800))
        except (TypeError, ValueError):
            ncm_max_samples_cfg = 800
        ncm_max_samples_cfg = max(32, ncm_max_samples_cfg)

        self.ncm_method = pn.widgets.Select(
            name="NCM Method",
            options=["ledoit_wolf", "oas", "graphical_lasso", "empirical"],
            value="ledoit_wolf",
        )
        self.ncm_backend = pn.widgets.RadioButtonGroup(
            name="NCM Backend",
            options=["JAX (GPU)", "sklearn (CPU)", "Auto"],
            value="Auto",
            button_type="primary",
        )
        self.ncm_method_info = pn.pane.Markdown(
            """
### NCM Estimation Methods

**Backend Selection**
- **JAX (GPU)**: fastest on large matrices (recommended default)
- **sklearn (CPU)**: robust CPU fallback and Graphical Lasso support
- **Auto**: tries JAX first, then sklearn fallback

**Methods**
1. **Ledoit-Wolf** (recommended): best speed/accuracy tradeoff for n <= p
2. **OAS**: shrinkage assuming Gaussian residuals
3. **Empirical**: fastest but unstable when n_traces << n_samples
4. **Graphical Lasso**: sklearn-only, usually slower

**Reference:** Denakpo et al., *IEEE Transactions on Instrumentation and Measurement*, 2025.
""",
            sizing_mode="stretch_width",
        )
        self.ncm_progress = pn.indicators.LoadingSpinner(
            value=False,
            size=25,
            name="Computing NCM...",
        )
        self.ncm_time_sampling_mode = pn.widgets.RadioButtonGroup(
            name="NCM time samples",
            options=["Max", "Custom", "Sans"],
            value=ncm_time_mode,
            button_type="default",
        )
        self.ncm_max_samples = pn.widgets.IntInput(
            name="NCM max time samples",
            value=ncm_max_samples_cfg,
            step=100,
        )
        self.ncm_compute_diagnostics = pn.widgets.Checkbox(
            name="Compute NCM diagnostics",
            value=False,
        )
        self.ncm_backend_display = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.ncm_info_display = pn.pane.Markdown("", sizing_mode="stretch_width")
        self._ncm_combined_info_text = ""
        self.ncm_combined_info = pn.pane.Markdown(
            pn.bind(lambda _: self._ncm_combined_info_text, self.ncm_info_display.param.object),
            sizing_mode="stretch_width",
        )
        self.ncm_settings_panel = pn.Column(
            pn.pane.Markdown("### Noise Covariance Matrix Settings"),
            self.ncm_backend,
            self.ncm_method,
            self.ncm_method_info,
            self.ncm_time_sampling_mode,
            self.ncm_max_samples,
            self.ncm_compute_diagnostics,
            sizing_mode="stretch_width",
        )
        self._on_ncm_sampling_mode_change()

    def _build_correction_widgets(self, cfg):
        """Create correction parameter widgets."""
        periodic_mode_cfg = str(cfg.get("periodic_mode", "CPU"))
        periodic_mode_norm = periodic_mode_cfg.strip().lower()
        if periodic_mode_norm == "strict":
            periodic_mode_cfg = "Strict"
        elif periodic_mode_norm == "fast":
            periodic_mode_cfg = "Fast"
        else:
            periodic_mode_cfg = "CPU"

        self.cb_delay = pn.widgets.Checkbox(name="Correct delay", value=True)
        self.cb_amplitude = pn.widgets.Checkbox(name="Correct amplitude (a)", value=True)
        self.cb_dilation = pn.widgets.Checkbox(name="Correct dilation (a)", value=False)
        self.limit_delay = pn.widgets.FloatInput(
            name="|delay|max (s)", value=10e-12, step=1e-13
        )
        self.limit_amplitude_a = pn.widgets.FloatInput(name="|amplitude a|max", value=10e-2, step=0.01)
        self.limit_dilation_a = pn.widgets.FloatInput(name="|dilation a|max", value=1e-3, step=0.005)
        self.cb_periodic = pn.widgets.Checkbox(
            name="Correct periodic sampling?", value=bool(cfg.get("periodic_enable", False))
        )
        self.periodic_mode = pn.widgets.RadioButtonGroup(
            name="Periodic mode",
            options=["CPU", "Fast", "Strict"],
            value=periodic_mode_cfg,
            button_type="default",
        )
        self.periodic_freq = pn.widgets.FloatInput(
            name="Frequency [THz]", value=float(cfg.get("periodic_freq_thz", 7.5)), step=0.1, format="0.0#"
        )

    def _build_optimization_widgets(self, cfg=None):
        """Create optimization widgets."""
        cfg = cfg or {}
        trace_mode = str(cfg.get("opt_trace_mode", "All"))
        if trace_mode not in ("All", "Limit"):
            trace_mode = "All"
        try:
            trace_count = int(cfg.get("opt_trace_count", 1000))
        except (TypeError, ValueError):
            trace_count = 1000
        trace_count = max(1, trace_count)
        legacy_mode = bool(cfg.get("legacy_plot_mode", False))
        self.maxiter = pn.widgets.IntInput(name="Iterations (Adam)", value=1000, step=50)
        self.lr = pn.widgets.FloatInput(
            name="Learning rate (auto)",
            value=0.0,
            step=0.001,
            format="0.0#",
            disabled=True,
        )
        self.scheduler_type = pn.widgets.Select(
            name="LR scheduler",
            options=["cosine", "exp", "piecewise"],
            value="cosine",
        )
        self.opt_trace_mode = pn.widgets.RadioButtonGroup(
            name="Traces to optimize",
            options=["All", "Limit"],
            value=trace_mode,
            button_type="default",
        )
        self.opt_trace_count = pn.widgets.IntInput(
            name="Trace count (if Limit)",
            value=trace_count,
            step=100,
            start=1,
        )
        self.legacy_plot_mode = pn.widgets.Checkbox(
            name="Legacy comparison mode (no corrected re-filter)",
            value=legacy_mode,
        )
        self.tol = pn.widgets.FloatInput(
            name="Early-stop tolerance (auto)",
            value=0.0,
            step=1e-5,
            format="0.0e",
            disabled=True,
        )

        self.device_choice = "GPU"
        self.btn_cpu = pn.widgets.Button(name="CPU", button_type="default")
        self.btn_gpu = pn.widgets.Button(name="GPU", button_type="primary")
        self.btn_cpu.on_click(lambda *_: self._select_device("CPU"))
        self.btn_gpu.on_click(lambda *_: self._select_device("GPU"))
        self._on_trace_mode_change()

    def _build_matrix_widgets(self):
        """Create matrix analysis widgets."""
        self.matrix_view_selector = pn.widgets.RadioButtonGroup(
            name="Matrix View",
            options=["NCM", "Precision", "Both"],
            value="Both",
            button_type="primary",
        )
        self.matrix_scale_selector = pn.widgets.RadioButtonGroup(
            name="Color Scale",
            options=["Linear", "Log", "SymLog"],
            value="Log",
            button_type="default",
        )
        self.show_eigenvalues = pn.widgets.Checkbox(
            name="Show Eigenvalue Distribution",
            value=True,
        )
        self.show_matrix_histograms = pn.widgets.Checkbox(
            name="Show Matrix Histograms",
            value=True,
        )
        self.matrix_diagnostics_area = pn.Column(sizing_mode="stretch_width")
        self.matrix_plot_area = pn.Column(
            sizing_mode="stretch_width",
            styles={"gap": "18px"},
        )
        self.matrix_analysis_view = pn.Column(
            pn.pane.Markdown("### Matrix Analysis"),
            pn.Row(
                self.matrix_view_selector,
                self.matrix_scale_selector,
                self.show_eigenvalues,
                self.show_matrix_histograms,
                sizing_mode="stretch_width",
            ),
            self.matrix_diagnostics_area,
            self.matrix_plot_area,
            sizing_mode="stretch_width",
        )

    @staticmethod
    def _plot_card_style(*, min_height: str, height: str, max_height: str, overflow: str) -> dict:
        """Return shared card style for visualization containers."""
        return {
            "background": "white",
            "border-radius": "10px",
            "box-shadow": "0 2px 10px rgba(15, 23, 42, 0.15)",
            "padding": "15px",
            "min-height": min_height,
            "height": height,
            "max-height": max_height,
            "overflow": overflow,
        }

    def _build_visualization_widgets(self):
        """Create plot panes and visualization containers."""
        responsive_plot_kwargs = dict(sizing_mode="stretch_both", min_height=360)
        spectrum_plot_kwargs = dict(sizing_mode="stretch_both", min_height=320)
        param_plot_kwargs = dict(sizing_mode="stretch_both", min_height=280, min_width=0)
        self.plot_time = pn.pane.Bokeh(**responsive_plot_kwargs)
        self.plot_std_time = pn.pane.Bokeh(**responsive_plot_kwargs)
        self.spectrum_pane = pn.pane.Bokeh(**spectrum_plot_kwargs)
        self.std_spectrum_pane = pn.pane.Bokeh(**spectrum_plot_kwargs)
        self.plot_phase = pn.pane.Bokeh(**responsive_plot_kwargs)
        self.plot_params_delay = pn.pane.Bokeh(**param_plot_kwargs)
        self.plot_params_delay_hist = pn.pane.Bokeh(**param_plot_kwargs)
        self.plot_params_amp = pn.pane.Bokeh(**param_plot_kwargs)
        self.plot_params_amp_hist = pn.pane.Bokeh(**param_plot_kwargs)
        self.params_delay_row = pn.Row(
            self.plot_params_delay,
            self.plot_params_delay_hist,
            sizing_mode="stretch_both",
            min_height=300,
            styles={"gap": "12px"},
        )
        self.params_amp_row = pn.Row(
            self.plot_params_amp,
            self.plot_params_amp_hist,
            sizing_mode="stretch_both",
            min_height=300,
            styles={"gap": "12px"},
        )
        self.params_panel = pn.Column(
            self.params_delay_row,
            self.params_amp_row,
            sizing_mode="stretch_both",
            min_height=620,
        )

        self._plot_display_styles_default = self._plot_card_style(
            min_height="45vh",
            height="62vh",
            max_height="85vh",
            overflow="hidden",
        )
        self._plot_display_styles_matrix = self._plot_card_style(
            min_height="70vh",
            height="calc(100vh - 240px)",
            max_height="none",
            overflow="auto",
        )
        self._plot_display_styles_params = dict(self._plot_display_styles_matrix)
        self.plot_display_area = pn.Column(
            sizing_mode="stretch_both",
            styles=dict(self._plot_display_styles_default),
        )
        self.plots = {
            "pulse": (self.plot_time, "Pulse E field", False),
            "pulse_std": (self.plot_std_time, "Std Pulse E field", False),
            "spectrum": (self.spectrum_pane, "E field [dB]", True),
            "spectrum_std": (self.std_spectrum_pane, "Std E field [dB]", True),
            "phase": (self.plot_phase, "Phase", False),
            "params": (self.params_panel, "Correction parameters", False),
            "matrix": (self.matrix_analysis_view, "Matrix Analysis", False),
        }
        self._plot_order = ["pulse", "pulse_std", "spectrum", "spectrum_std", "phase", "params", "matrix"]
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

    def _wire_events(self):
        """Connect widget callbacks."""
        self.file_selector.param.watch(self.on_file_selected, "value")
        self.reference_file_selector.param.watch(self.on_reference_file_selected, "value")
        self.batch_file_selector.param.watch(self._on_batch_files_selected, "value")
        self.btn_analyze.on_click(self.preview_analysis)
        self.btn_optimize.on_click(self.run_optimization)
        self.btn_export.on_click(self.export_results)
        self.btn_batch_autolink.on_click(self.auto_link_batch_pairs)
        self.btn_batch_run.on_click(self.run_batch_processing)
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
        self.matrix_view_selector.param.watch(self._on_matrix_control_change, "value")
        self.matrix_scale_selector.param.watch(self._on_matrix_control_change, "value")
        self.show_eigenvalues.param.watch(self._on_matrix_control_change, "value")
        self.show_matrix_histograms.param.watch(self._on_matrix_control_change, "value")
        self.cb_periodic.param.watch(self.save_config, "value")
        self.periodic_mode.param.watch(self.save_config, "value")
        self.periodic_freq.param.watch(self.save_config, "value")
        self.ncm_time_sampling_mode.param.watch(self._on_ncm_sampling_mode_change, "value")
        self.ncm_time_sampling_mode.param.watch(self.save_config, "value")
        self.ncm_max_samples.param.watch(self.save_config, "value")
        self.opt_trace_mode.param.watch(self._on_trace_mode_change, "value")
        self.opt_trace_mode.param.watch(self.save_config, "value")
        self.opt_trace_count.param.watch(self.save_config, "value")
        self.legacy_plot_mode.param.watch(self.save_config, "value")
        self.superresolution_toggle.param.watch(self._update_superres_info, "value")
        self._update_superres_info()
        if self._plot_order:
            self._select_plot_view(self._plot_order[0], sync_selector=False)

    def _build_layout(self):
        """Build template layout."""
        self.expert_options = pn.Accordion(
            (
                "Expert options",
                pn.Column(
                    self.cb_delay,
                    self.limit_delay,
                    self.cb_dilation,
                    self.limit_dilation_a,
                    self.cb_periodic,
                    self.periodic_mode,
                    self.periodic_freq,
                ),
            ),
            active=[],
        )

        self.file_picker = pn.Accordion(
            (
                "Choose a .h5 file",
                pn.Column(
                    pn.Row(self.time_filter_preview, self.filter_preview),
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
                pn.Row(self.superresolution_toggle, self.superresolution_info),
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
                self.opt_trace_mode,
                self.opt_trace_count,
                self.legacy_plot_mode,
                self.maxiter,
                self.lr,
                self.scheduler_type,
                self.ncm_settings_panel,
                self.ncm_progress,
                self.ncm_backend_display,
                self.ncm_info_display,
                self.ncm_combined_info,
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

    def __init__(self):
        """Initialize application."""
        self.current_file = None
        self._active_reference_file = None
        self._batch_file_pool = []
        self.time = None
        self.time_orig = None
        self.pulses_raw = None
        self.mean_pulse_raw = None
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
        self.ncm = None
        self.ncm_info = {}
        self.ncm_sample = None
        self.ncm_ref_sim = None
        self.precision_matrix = None
        self.matrix_diagnostics = None
        self._model = None
        self._superresolution_n_original = None

        self._build_file_widgets()
        self._build_control_widgets()
        cfg = self._build_filter_widgets()
        self._build_ncm_widgets(cfg)
        self._build_correction_widgets(cfg)
        self._build_optimization_widgets(cfg)
        self._build_matrix_widgets()
        self._build_visualization_widgets()

        self._wire_events()
        self._build_layout()
        self._refresh_matrix_analysis_views()
        self.update_filter_preview(None)

    def _select_device(self, which: str):
        self.device_choice = "GPU" if str(which).upper() == "GPU" else "CPU"
        if self.device_choice == "GPU":
            self.btn_gpu.button_type = "success"
            self.btn_cpu.button_type = "default"
        else:
            self.btn_cpu.button_type = "primary"
            self.btn_gpu.button_type = "default"

    def _on_trace_mode_change(self, event=None):
        """Enable/disable fixed trace-count input depending on optimization mode."""
        if not hasattr(self, "opt_trace_mode") or not hasattr(self, "opt_trace_count"):
            return
        mode = str(self.opt_trace_mode.value)
        self.opt_trace_count.disabled = mode != "Limit"

    def _on_ncm_sampling_mode_change(self, event=None):
        """Enable/disable custom NCM sample cap based on selected mode."""
        if not hasattr(self, "ncm_time_sampling_mode") or not hasattr(self, "ncm_max_samples"):
            return
        mode = str(self.ncm_time_sampling_mode.value)
        self.ncm_max_samples.disabled = mode != "Custom"
        disable_ncm_controls = mode == "Sans"
        if hasattr(self, "ncm_backend"):
            self.ncm_backend.disabled = disable_ncm_controls
        if hasattr(self, "ncm_method"):
            self.ncm_method.disabled = disable_ncm_controls
        if hasattr(self, "ncm_compute_diagnostics"):
            self.ncm_compute_diagnostics.disabled = disable_ncm_controls

    def _update_superres_info(self, event=None):
        """Update superresolution info pane when the toggle state changes."""
        if not getattr(self, "superresolution_toggle", None):
            return
        if not self.superresolution_toggle.value:
            self.superresolution_info.object = ""
            return
        time_axis = getattr(self, "time", None)
        if time_axis is None:
            self.superresolution_info.object = "_Load a file first_"
            return
        time_np = np.asarray(time_axis)
        if time_np.size < 2:
            self.superresolution_info.object = "_Load a file first_"
            return
        dt_s = float(time_np[1] - time_np[0])
        n_orig = len(time_np)
        n_padded = compute_superresolution_npadded(dt_s)
        factor = n_padded // n_orig
        self.superresolution_info.object = (
            f"**{n_orig} → {n_padded} samples** (×{factor} resolution boost, "
            f"Δf = {LASER_FREP_HZ/1e6:.3f} MHz)"
        )

    @staticmethod
    def _is_resource_exhausted_error(err: Exception) -> bool:
        """Return True when an exception message indicates a device OOM/resource exhaustion."""
        msg = str(err)
        return (
            "RESOURCE_EXHAUSTED" in msg
            or "Out of memory" in msg
            or "CUDA_ERROR_OUT_OF_MEMORY" in msg
            or "Failed to create cuFFT batched plan" in msg
            or "cufft" in msg.lower()
        )

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
                periodic_mode=str(self.periodic_mode.value) if hasattr(self, "periodic_mode") else "CPU",
                periodic_freq_thz=float(self.periodic_freq.value),
                opt_trace_mode=str(self.opt_trace_mode.value) if hasattr(self, "opt_trace_mode") else "All",
                opt_trace_count=int(self.opt_trace_count.value) if hasattr(self, "opt_trace_count") else 1000,
                legacy_plot_mode=bool(self.legacy_plot_mode.value) if hasattr(self, "legacy_plot_mode") else False,
                ncm_time_sampling_mode=(
                    str(self.ncm_time_sampling_mode.value) if hasattr(self, "ncm_time_sampling_mode") else "Max"
                ),
                ncm_max_samples=int(self.ncm_max_samples.value) if hasattr(self, "ncm_max_samples") else 800,
            )
            CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        except Exception as e:
            self.show_error(e)

    def _notify(self, message: str, level: str = "info", duration: int = 3000):
        """Show toast notifications when Panel exposes the notification API."""
        notifications = getattr(pn.state, "notifications", None)
        if notifications is None:
            return
        handler = getattr(notifications, level, None)
        if callable(handler):
            handler(str(message), duration=duration)
            return
        fallback = getattr(notifications, "info", None)
        if callable(fallback):
            fallback(str(message), duration=duration)

    def _clear_ncm_state(self, *, clear_export_payload: bool = False):
        """Reset NCM-related arrays and optional export cache keys."""
        self.ncm = None
        self.ncm_info = {}
        self.ncm_sample = None
        self.ncm_ref_sim = None
        self.precision_matrix = None
        self.matrix_diagnostics = None
        self._ncm_combined_info_text = ""
        if clear_export_payload and isinstance(self.export_payload, dict):
            for key in ("ncm", "ncm_sample", "ncm_ref_sim", "precision_matrix", "matrix_diagnostics", "ncm_info"):
                self.export_payload.pop(key, None)

    def _set_ncm_backend_pane(self, *, backend: str, device: str, backend_time):
        """Render NCM backend summary pane."""
        self.ncm_backend_display.object = (
            "**NCM Backend:**\n"
            f"- Backend: {backend}\n"
            f"- Device: {device}\n"
            f"- Backend time: {backend_time}"
        )

    def _set_ncm_info_pane(self, lines):
        """Render NCM info pane from bullet lines."""
        body = "\n".join(f"- {line}" for line in lines)
        self.ncm_info_display.object = f"**NCM Computation Info:**\n{body}"

    def _release_previous_optimization_artifacts(self):
        """
        Drop large arrays/plots from the previous optimization run.

        This reduces memory pressure before starting a new optimization and helps
        keep run-to-run timing stable.
        """
        had_previous = any(
            (
                self.corrected is not None,
                self.optimal_params is not None,
                self.ncm is not None,
                self.precision_matrix is not None,
                self.matrix_diagnostics is not None,
                bool(self.export_payload),
            )
        )
        if not had_previous:
            return False

        self.corrected = None
        self.optimal_params = None
        self._clear_ncm_state(clear_export_payload=False)

        self.export_payload = {}

        if hasattr(self, "ncm_progress"):
            self.ncm_progress.value = False
        if hasattr(self, "ncm_backend_display"):
            self.ncm_backend_display.object = ""
        if hasattr(self, "ncm_info_display"):
            self.ncm_info_display.object = ""

        return True

    def show_error(self, e: Exception, prefix: str = "Error"):
        """Display an error inside the UI."""
        self.error_box.object = f"{prefix}: {type(e).__name__}\n```\n{traceback.format_exc()}\n```"
        self.error_box.visible = True
        self._notify(f"{prefix}: {type(e).__name__}", level="error")

    @staticmethod
    def _normalize_preview_curve(values):
        """Normalize preview curves to [0, 1] for mask overlay visualization."""
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size == 0:
            return None
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        vmax = float(np.max(np.abs(arr)))
        if not np.isfinite(vmax) or vmax <= 1e-30:
            return np.zeros_like(arr)
        return np.clip(np.abs(arr) / vmax, 0.0, 1.0)

    def update_filter_preview(self, event):
        """Refresh the frequency/time filter previews."""
        try:
            loaded_freqs = np.asarray(self.freqs) if self.freqs is not None else None
            if loaded_freqs is not None and loaded_freqs.size > 1:
                freqs_np = loaded_freqs
            else:
                freqs_np = np.linspace(0, 10e12, 1200)
            mask_freq = build_frequency_mask(
                freqs_np,
                self.filter_low.value,
                self.filter_high.value,
                float(self.freq_start.value),
                float(self.freq_end.value),
                self.sharpness.value
            )

            loaded_time = np.asarray(self.time) if self.time is not None else None
            mean_raw = np.asarray(self.mean_pulse_raw) if self.mean_pulse_raw is not None else None
            freq_overlay = None
            freq_title = "Frequency filter preview"
            if (
                loaded_time is not None
                and mean_raw is not None
                and loaded_time.size > 1
                and mean_raw.size == loaded_time.size
            ):
                mean_time_filtered = apply_time_filter(
                    loaded_time,
                    mean_raw,
                    bool(self.tfilter_low.value),
                    bool(self.tfilter_high.value),
                    float(self.t_start.value),
                    float(self.t_end.value),
                    float(self.t_sharpness.value),
                )
                mean_spec = fft_mag_correct_tds(mean_time_filtered)
                common = int(min(freqs_np.size, mask_freq.size, mean_spec.size))
                if common > 1:
                    freqs_np = freqs_np[:common]
                    mask_freq = mask_freq[:common]
                    freq_overlay = self._normalize_preview_curve(mean_spec[:common])
                    freq_title = "Frequency filter preview (loaded .h5)"

            freq_fig = build_filter_plot(
                freqs_np,
                mask_freq,
                title=freq_title,
                x_label="Frequency [Hz]",
                y_label="Transmission",
                color=ALASKA_PRIMARY,
                overlay_x=freqs_np if freq_overlay is not None else None,
                overlay_y=freq_overlay,
                overlay_label="Loaded spectrum (norm)",
                overlay_color=ALASKA_BLUE,
            )
            self._set_plot(self.filter_preview, freq_fig)

            # Use the loaded time axis when available, default to 0-10 ps otherwise
            if loaded_time is not None and loaded_time.size > 1:
                t_axis = loaded_time
            else:
                t_axis = np.linspace(0, 10e-12, 1200)
            mask_time = build_time_mask(
                t_axis,
                bool(self.tfilter_low.value),
                bool(self.tfilter_high.value),
                float(self.t_start.value),
                float(self.t_end.value),
                float(self.t_sharpness.value),
            )
            time_overlay = None
            time_title = "Time filter preview"
            if mean_raw is not None and mean_raw.size == t_axis.size:
                time_overlay = self._normalize_preview_curve(mean_raw)
                time_title = "Time filter preview (loaded .h5)"
            time_fig = build_filter_plot(
                t_axis,
                mask_time,
                title=time_title,
                x_label="Time [s]",
                y_label="Transmission",
                color=ALASKA_BLUE,
                overlay_x=t_axis if time_overlay is not None else None,
                overlay_y=time_overlay,
                overlay_label="Loaded mean pulse (norm)",
                overlay_color=ALASKA_PRIMARY,
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
            self._plots_with_cursor = set()
        try:
            self._plots_with_cursor.add(figure)
        except Exception:
            return figure
        set_mouse_coordinate_visibility(figure, bool(self.show_cursor_coords.value))
        return figure

    def _unregister_plot(self, figure):
        """Remove a figure from cursor-tracking registry."""
        if figure is None:
            return
        if not hasattr(self, "_plots_with_cursor"):
            return
        self._plots_with_cursor.discard(figure)

    def _set_plot(self, pane, figure):
        """Assign a figure to a Panel pane while registering it for cursor overlays."""
        if pane is None:
            return
        previous = getattr(pane, "object", None)
        if previous is not None and previous is not figure:
            self._unregister_plot(previous)
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
        if key == "matrix":
            self.plot_display_area.styles = dict(self._plot_display_styles_matrix)
        elif key == "params":
            self.plot_display_area.styles = dict(self._plot_display_styles_params)
        else:
            self.plot_display_area.styles = dict(self._plot_display_styles_default)
        self._active_plot_key = key

    def _on_matrix_control_change(self, event):
        """Refresh matrix analysis views when matrix controls are changed."""
        self._refresh_matrix_analysis_views()

    def _update_matrix_plots(self, view: str, scale: str, show_eigs: bool, show_hists: bool):
        """Build matrix analysis plots according to current UI controls."""
        if self.ncm is None:
            return pn.pane.Markdown("No matrix data. Run optimization first.")

        scale_map = {"linear": "linear", "log": "log", "symlog": "symlog"}
        scale_key = scale_map.get(str(scale).strip().lower(), "log")

        def _pair_layout(heatmap, histogram):
            return pn.Row(
                heatmap,
                histogram,
                sizing_mode="stretch_width",
                min_height=340,
                styles={
                    "align-items": "flex-start",
                    "gap": "18px",
                    "flex-wrap": "nowrap",
                    "overflow-x": "auto",
                    "overflow-y": "hidden",
                },
                margin=(0, 0, 12, 0),
            )

        matrix_panels = []
        if view in ("NCM", "Both"):
            heatmap = build_ncm_heatmap(
                self.ncm,
                scale=scale_key,
                title="Noise Covariance Matrix",
            )
            if show_hists:
                histogram = build_matrix_value_histogram(
                    self.ncm,
                    scale=scale_key,
                    title="NCM value distribution",
                    base_label="NCM",
                )
                matrix_panels.append(_pair_layout(heatmap, histogram))
            else:
                matrix_panels.append(heatmap)

        if view in ("Precision", "Both"):
            if self.precision_matrix is None:
                matrix_panels.append(
                    pn.pane.Markdown("Precision matrix unavailable.")
                )
            else:
                heatmap = build_precision_heatmap(
                    self.precision_matrix,
                    scale=scale_key,
                    title="Precision Matrix",
                )
                if show_hists:
                    histogram = build_matrix_value_histogram(
                        self.precision_matrix,
                        scale=scale_key,
                        title="Precision value distribution",
                        base_label="Precision",
                        color=ALASKA_PRIMARY,
                    )
                    matrix_panels.append(_pair_layout(heatmap, histogram))
                else:
                    matrix_panels.append(heatmap)

        if not matrix_panels:
            return pn.pane.Markdown("No matrix plots available.")

        sections = list(matrix_panels)
        if show_eigs and self.matrix_diagnostics is not None:
            sections.append(build_eigenvalue_plot(self.matrix_diagnostics))

        if len(sections) == 1:
            return sections[0]
        return pn.Column(
            *sections,
            sizing_mode="stretch_width",
            styles={"gap": "18px"},
        )

    def _update_diagnostics_display(self):
        """Render matrix diagnostics as a formatted HTML table."""
        if self.matrix_diagnostics is None:
            return pn.pane.Markdown("No diagnostics available.")

        diagnostics = self.matrix_diagnostics or {}
        ncm_diag = diagnostics.get("ncm", {}) if isinstance(diagnostics, dict) else {}
        prec_diag = diagnostics.get("precision", {}) if isinstance(diagnostics, dict) else {}

        ncm_eigs = np.asarray(ncm_diag.get("eigenvalues", []), dtype=float).ravel()
        ncm_eigs = ncm_eigs[np.isfinite(ncm_eigs)]
        prec_eigs = np.asarray(prec_diag.get("eigenvalues", []), dtype=float).ravel()
        prec_eigs = prec_eigs[np.isfinite(prec_eigs)]

        ncm_max_eig = float(np.max(ncm_eigs)) if ncm_eigs.size else np.nan
        ncm_min_eig = float(np.min(ncm_eigs)) if ncm_eigs.size else np.nan
        prec_max_eig = float(np.max(prec_eigs)) if prec_eigs.size else np.nan
        prec_min_eig = float(np.min(prec_eigs)) if prec_eigs.size else np.nan

        ncm_condition = float(ncm_diag.get("condition_number", np.nan))
        ncm_rank = int(ncm_diag.get("rank", 0))
        ncm_trace = float(ncm_diag.get("trace", np.nan))
        ncm_frob = float(ncm_diag.get("frobenius_norm", np.nan))

        prec_condition = float(prec_diag.get("condition_number", np.nan))
        prec_rank = int(prec_diag.get("rank", 0))
        if "trace" in prec_diag:
            prec_trace = float(prec_diag.get("trace", np.nan))
        elif self.precision_matrix is not None:
            prec_trace = float(np.trace(np.asarray(self.precision_matrix)))
        else:
            prec_trace = np.nan
        if "frobenius_norm" in prec_diag:
            prec_frob = float(prec_diag.get("frobenius_norm", np.nan))
        elif self.precision_matrix is not None:
            prec_frob = float(np.linalg.norm(np.asarray(self.precision_matrix), ord="fro"))
        else:
            prec_frob = np.nan
        ncm_sparsity = float(ncm_diag.get("sparsity", np.nan))
        prec_sparsity = float(prec_diag.get("sparsity", np.nan))

        def _fmt(value):
            try:
                val = float(value)
            except Exception:
                return "N/A"
            if not np.isfinite(val):
                return "N/A"
            return f"{val:.2e}"

        def _fmt_pct(value):
            try:
                val = float(value)
            except Exception:
                return "N/A"
            if not np.isfinite(val):
                return "N/A"
            return f"{val * 100:.1f}%"

        table_html = f"""
        <table style="width:100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: {ALASKA_PRIMARY}; color: white;">
                <th style="padding: 8px; text-align: left;">Metric</th>
                <th style="padding: 8px; text-align: left;">NCM</th>
                <th style="padding: 8px; text-align: left;">Precision</th>
            </tr>
        </thead>
        <tbody>
            <tr><td><b>Condition Number</b></td><td>{_fmt(ncm_condition)}</td><td>{_fmt(prec_condition)}</td></tr>
            <tr><td><b>Rank</b></td><td>{ncm_rank}</td><td>{prec_rank}</td></tr>
            <tr><td><b>Trace</b></td><td>{_fmt(ncm_trace)}</td><td>{_fmt(prec_trace)}</td></tr>
            <tr><td><b>Frobenius Norm</b></td><td>{_fmt(ncm_frob)}</td><td>{_fmt(prec_frob)}</td></tr>
            <tr><td><b>Max Eigenvalue</b></td><td>{_fmt(ncm_max_eig)}</td><td>{_fmt(prec_max_eig)}</td></tr>
            <tr><td><b>Min Eigenvalue</b></td><td>{_fmt(ncm_min_eig)}</td><td>{_fmt(prec_min_eig)}</td></tr>
            <tr><td><b>Sparsity</b></td><td>{_fmt_pct(ncm_sparsity)}</td><td>{_fmt_pct(prec_sparsity)}</td></tr>
        </tbody>
        </table>
        """
        return pn.pane.HTML(table_html, sizing_mode="stretch_width")

    def _refresh_matrix_analysis_views(self):
        """Refresh diagnostics table and matrix plots in the Matrix Analysis tab."""
        if not hasattr(self, "matrix_diagnostics_area") or not hasattr(self, "matrix_plot_area"):
            return
        self.matrix_diagnostics_area.objects = [self._update_diagnostics_display()]
        self.matrix_plot_area.objects = [
            self._update_matrix_plots(
                self.matrix_view_selector.value,
                self.matrix_scale_selector.value,
                bool(self.show_eigenvalues.value),
                bool(self.show_matrix_histograms.value),
            )
        ]

    @staticmethod
    def _estimate_graphical_lasso_runtime(n_samples_used: int, n_traces_used: int) -> str:
        """
        Estimate Graphical Lasso runtime window from matrix shape.

        This heuristic depends on both:
        - feature dimension (`n_samples_used`)
        - number of traces (`n_traces_used`)
        and is used for UI messaging only.
        """
        p = float(max(1, n_samples_used))
        n = float(max(3, n_traces_used))
        complexity = (p / 300.0) ** 2.6 * (n / 1000.0) ** 0.5

        if complexity <= 0.4:
            return "1-3 min"
        if complexity <= 0.8:
            return "3-6 min"
        if complexity <= 1.5:
            return "5-10 min"
        if complexity <= 2.5:
            return "10-20 min"
        if complexity <= 4.0:
            return "20-30 min"
        return "30+ min"

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
        mask = build_frequency_mask(np.asarray(freqs), *self._filter_config())
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

    def _validate_and_preprocess_data(self):
        """Validate loaded data and apply time-domain preprocessing."""
        if self.pulses is None or len(self.pulses) == 0:
            raise ValueError("No data loaded")

        pulses_raw = np.asarray(self.pulses, dtype=float)
        return self._apply_time_filter_to_pulses(pulses_raw)

    def _compute_analysis_metrics(self, filtered_pulses):
        """Compute periodic sampling diagnostics and statistics."""
        mean_pulse = np.mean(filtered_pulses, axis=0)
        self.periodic_diagnostics = None

        std_pulse = np.std(filtered_pulses, axis=0)
        return filtered_pulses, mean_pulse, std_pulse

    def _update_preview_plots(self, mean_pulse, std_pulse):
        """Update preview plots with analyzed data."""
        t_orig = self.time_orig
        freqs = self.freqs
        ref_pulse_filtered = self.ref_pulse
        ref_pulse_time = self.ref_pulse_time

        self._set_plot(
            self.plot_time,
            build_time_domain_plot(
                t_orig,
                mean_pulse,
                std=std_pulse,
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
                std_pulse,
                title="Temporal standard deviation (raw)",
                x_label="Time [orig units]",
                y_label="Std",
            ),
        )

        self._unregister_plot(getattr(self, "_spec_lin", None))
        self._unregister_plot(getattr(self, "_spec_log", None))
        self._unregister_plot(getattr(self, "_std_lin", None))
        self._unregister_plot(getattr(self, "_std_log", None))

        fft_all = np.fft.rfft(self.pulses_time_base, axis=1)
        if self.mean_spec_mode.value == "FFT(mean)":
            mean_spec_raw = fft_mag_correct_tds(mean_pulse)
        else:
            mean_spec_raw = fft_mag_correct_tds(self.pulses_time_base, axis=1).mean(axis=0)
        ref_spec_raw = fft_mag_correct_tds(ref_pulse_time)
        std_spec_raw = np.abs(np.std(fft_all, axis=0))

        mean_spec_f = self._filter_spectrum(freqs, mean_spec_raw)
        ref_spec_f = self._filter_spectrum(freqs, ref_spec_raw)
        std_spec_f = self._filter_spectrum(freqs, std_spec_raw)

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
                raw_std=self._db_scale(std_spec_f),
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

        phase_mean = np.unwrap(np.angle(np.fft.rfft(mean_pulse)))
        phase_ref = np.unwrap(np.angle(np.fft.rfft(ref_pulse_filtered)))
        self._set_plot(
            self.plot_phase,
            build_phase_plot(
                freqs,
                phase_mean,
                ref_phase=phase_ref,
                title="Phases",
                y_label="Phase",
            ),
        )

    def on_file_selected(self, event):
        """Handle file selection."""
        try:
            if event.new:
                self.current_file = Path(event.new[0])
                self._active_reference_file = None
                self.pulses_raw = None
                self.mean_pulse_raw = None
                self.status.object = f"File selected: `{self.current_file.name}`"
                self.sample_file_display.object = f"**Sample selected:** `{self.current_file}`"
                sample_dir = str(self.current_file.parent)
                current_ref_dir = str(getattr(self.reference_file_selector, "directory", ""))
                if sample_dir != current_ref_dir:
                    new_ref_fs = self._make_h5_selector(sample_dir)
                    new_ref_fs.param.watch(self.on_reference_file_selected, "value")
                    self.reference_file_selector = new_ref_fs
                    self.reference_selector_panel[0] = new_ref_fs
                    self.reference_file_display.object = "**Reference selected:** _none_"
            else:
                self.current_file = None
                self._active_reference_file = None
                self.pulses_raw = None
                self.mean_pulse_raw = None
                self.status.object = "No file selected."
                self.sample_file_display.object = "**Sample selected:** _none_"
            self.update_filter_preview(None)
        except Exception as e:
            self.show_error(e)

    def on_reference_file_selected(self, event):
        """Handle optional reference file selection."""
        try:
            if event.new:
                reference_file = Path(event.new[0])
                self._active_reference_file = reference_file
                self.reference_file_display.object = f"**Reference selected:** `{reference_file}`"
            else:
                self._active_reference_file = None
                self.reference_file_display.object = "**Reference selected:** _none_"
        except Exception as e:
            self.show_error(e)

    def preview_analysis(self, event=None, raise_on_error: bool = False):
        """Analyze loaded data and update preview figures."""
        try:
            self.error_box.visible = False
            self.progress.value = 0
            self.export_msg.visible = False
            self.export_msg.object = ""
            if not self.current_file:
                self.status.object = "Select an .h5 file first."
                if raise_on_error:
                    raise ValueError("No sample file selected.")
                return

            self.reset_metrics()
            self.status.object = "Loading HDF5..."
            self.progress.value = 5
            data = load_h5_file(self.current_file)
            if data is None:
                self.status.object = "Failed to load the file."
                if raise_on_error:
                    raise ValueError(f"Failed to load file: {self.current_file}")
                return

            pulses_list, timeaxis = data
            min_len = min(map(len, pulses_list))
            if min_len < 2:
                raise ValueError("Need at least 2 samples per trace.")
            pulses_array = np.vstack([p[:min_len] for p in pulses_list])
            t_orig = timeaxis[:min_len]

            t = t_orig.astype(float)
            dt_raw = float(t[1] - t[0])
            scale_to_s = 1e-12 if dt_raw > 1e-5 else 1.0
            t_s = t * scale_to_s
            freqs = np.fft.rfftfreq(min_len, d=(t_s[1] - t_s[0]))

            self.progress.value = 15
            self.status.object = (
                f"Pre-processing... dt_raw={dt_raw:g}, scale_to_s={scale_to_s}, "
                f"fmax={freqs.max()/1e12:.2f} THz"
            )

            self.time = t_s
            self.time_orig = t_orig
            self.freqs = freqs
            self.pulses = pulses_array
            self.pulses_raw = np.asarray(pulses_array, dtype=float)
            self.mean_pulse_raw = np.mean(self.pulses_raw, axis=0)
            self._update_superres_info()

            pulses_time_filtered = self._validate_and_preprocess_data()
            pulses_time_base, mean_pulse_time, _ = self._compute_analysis_metrics(pulses_time_filtered)

            pulses_freq_filtered = self._apply_freq_filter_to_pulses(pulses_time_base, freqs)
            mean_pulse_filtered = np.mean(pulses_freq_filtered, axis=0)
            time_std = np.std(pulses_freq_filtered, axis=0)

            std_metric_before = float(np.linalg.norm(time_std))
            self.update_metrics(before=std_metric_before)

            proj = pulses_time_base @ mean_pulse_time
            norms = np.einsum("ij,ij->i", pulses_time_base, pulses_time_base)
            ref_idx = int(np.argmin(np.abs(proj / (norms + 1e-30) - 1)))
            ref_pulse_filtered = pulses_freq_filtered[ref_idx]
            ref_pulse_time = pulses_time_base[ref_idx]
            self._notify(f"Reference pulse index selected: {ref_idx}", level="info")
            self._set_metric("ref_idx", ref_idx)

            self.pulses = pulses_freq_filtered
            self.pulses_time_base = pulses_time_base
            self.ref_index = ref_idx
            self.ref_pulse = ref_pulse_filtered
            self.ref_pulse_time = ref_pulse_time
            self.corrected = None
            self.export_payload = {}
            self._clear_ncm_state(clear_export_payload=False)
            self.ncm_progress.value = False
            self.ncm_backend_display.object = ""
            self.ncm_info_display.object = ""
            self._refresh_matrix_analysis_views()

            self.progress.value = 25
            self.status.object = "Spectra & phases (preview)..."
            self._update_preview_plots(mean_pulse_filtered, time_std)
            self.update_filter_preview(None)

            self.progress.value = 40
            self.switch_scale(None)
            self.status.object = f"Preview ready - {pulses_array.shape[0]} traces, ref #{ref_idx}"
            self.btn_optimize.disabled = False
            self.btn_export.disabled = False
            self._notify("Preview completed successfully.", level="success")

        except Exception as e:
            self.show_error(e, prefix="Preview")
            self.status.object = "Error during preview"
            if raise_on_error:
                raise


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

    def _setup_optimization(self):
        """Setup device, bounds, initial parameters."""
        choice = getattr(self, "device_choice", "CPU")
        device_preference = str(getattr(choice, "value", choice))
        if device_preference.lower() == "gpu":
            gpus = [d for d in jax.devices() if d.platform == "gpu"]
            device = gpus[0] if gpus else jax.devices("cpu")[0]
            exact_device = bool(gpus)
        else:
            device = jax.devices("cpu")[0]
            exact_device = True

        if all(hasattr(self, name) for name in ("delay_min", "amp_min", "dil_min", "delay_max", "amp_max", "dil_max")):
            lower_bounds = jnp.array(
                [float(self.delay_min.value), float(self.amp_min.value), float(self.dil_min.value)],
                dtype=jnp.float32,
            )
            upper_bounds = jnp.array(
                [float(self.delay_max.value), float(self.amp_max.value), float(self.dil_max.value)],
                dtype=jnp.float32,
            )
        else:
            lower_bounds, upper_bounds = self.build_bounds()

        n_traces = int(len(self.pulses))
        parameter_matrix = jnp.zeros((n_traces, 3), dtype=jnp.float32)
        return device, exact_device, lower_bounds, upper_bounds, parameter_matrix

    def _subsample_data_for_optimization(self, all_pulses, reference_pulse, time_vector, angular_frequencies):
        """Apply optional trace-count limiting for optimization."""
        n_traces = int(all_pulses.shape[0])

        indices = None
        if hasattr(self, "opt_trace_mode") and str(self.opt_trace_mode.value) == "Limit":
            target_count = max(1, int(getattr(self.opt_trace_count, "value", n_traces)))
            target_count = min(n_traces, target_count)
            if target_count < n_traces:
                indices = jnp.arange(target_count, dtype=jnp.int32)
                self._notify(
                    f"Optimizing first {target_count}/{n_traces} traces (mode=Limit)",
                    level="warning",
                    duration=5000,
                )
                all_pulses = all_pulses[indices]

        self._optimization_trace_indices = indices
        return all_pulses, reference_pulse, time_vector, angular_frequencies, indices

    def _run_optimization_loop(
        self,
        device,
        parameter_matrix,
        subsampled_pulses,
        subsampled_reference,
        sub_time_vector,
        sub_angular_frequencies,
        lower_bounds,
        upper_bounds,
    ):
        """Execute main JAX optimization loop."""
        num_iterations = int(getattr(getattr(self, "max_iterations", None), "value", self.maxiter.value))
        schedule_type = str(getattr(getattr(self, "schedule_type", None), "value", self.scheduler_type.value))

        min_iters_ratio = 0.514  # quality_speed=0.7 equivalent: good convergence, still early-stops
        min_iters = max(1, min(num_iterations, int(round(min_iters_ratio * num_iterations))))
        patience = 5
        tol_rel = 1.1e-3

        self._set_metric("min_iters_gate", int(min_iters))
        self._set_metric("patience_gate", int(patience))
        self._set_metric("tol_rel_base", tol_rel)
        check_every = 20  # fewer GPU-CPU syncs → faster on GPU

        trace_indices = getattr(self, "_optimization_trace_indices", None)
        working_params = parameter_matrix if trace_indices is None else parameter_matrix[trace_indices]
        n_working_traces = int(working_params.shape[0])
        model = getattr(self, "_model", None)
        use_model_superres = bool(model is not None and bool(getattr(model, "superresolution", False)))
        precomputed_objective = False
        if use_model_superres:
            try:
                objective_pulses, objective_reference = model.prepare_inputs(subsampled_pulses, subsampled_reference)
                objective_time = model.time_axis
                objective_omega = model.omega
                precomputed_objective = True
            except Exception as prep_err:
                if not self._is_resource_exhausted_error(prep_err):
                    raise
                objective_pulses = subsampled_pulses
                objective_reference = subsampled_reference
                objective_time = sub_time_vector
                objective_omega = sub_angular_frequencies
                self._notify(
                    "Superresolution precompute disabled (OOM): using chunked on-the-fly filtering",
                    level="warning",
                    duration=5000,
                )
        else:
            objective_pulses = subsampled_pulses
            objective_reference = subsampled_reference
            objective_time = sub_time_vector
            objective_omega = sub_angular_frequencies

        if use_model_superres:
            n_working_samples = int(getattr(model, "n_padded", objective_pulses.shape[1]))
        else:
            n_working_samples = int(objective_pulses.shape[1])
        bytes_per_trace_opt = n_working_samples * 4 * 24
        mb_per_trace_opt = bytes_per_trace_opt / (1024 ** 2)
        chunk_candidate = int(192 / max(mb_per_trace_opt, 1e-6))
        opt_chunk_size = max(1, min(n_working_traces, chunk_candidate))
        use_chunked_opt = opt_chunk_size < n_working_traces

        if use_chunked_opt:
            self._notify(
                f"Optimization gradients chunked: {n_working_traces} traces, chunk={opt_chunk_size}",
                level="warning",
                duration=4500,
            )
        bounds_tuple = (lower_bounds, upper_bounds)

        def _compute_gradients(params_matrix):
            if not use_chunked_opt:
                if use_model_superres and not precomputed_objective:
                    return model.gradients(
                        params_matrix,
                        subsampled_pulses,
                        subsampled_reference,
                        bounds_tuple,
                    )
                return batch_gradients(
                    params_matrix,
                    objective_pulses,
                    objective_reference,
                    objective_time,
                    objective_omega,
                    lower_bounds,
                    upper_bounds,
                )
            grad_chunks = []
            for i in range(0, n_working_traces, opt_chunk_size):
                if use_model_superres and not precomputed_objective:
                    grad_chunks.append(
                        model.gradients(
                            params_matrix[i : i + opt_chunk_size],
                            subsampled_pulses[i : i + opt_chunk_size],
                            subsampled_reference,
                            bounds_tuple,
                        )
                    )
                    continue
                grad_chunks.append(
                    batch_gradients(
                        params_matrix[i : i + opt_chunk_size],
                        objective_pulses[i : i + opt_chunk_size],
                        objective_reference,
                        objective_time,
                        objective_omega,
                        lower_bounds,
                        upper_bounds,
                    )
                )
            return jnp.concatenate(grad_chunks, axis=0)

        def _compute_losses(params_matrix):
            if not use_chunked_opt:
                if use_model_superres and not precomputed_objective:
                    return model.loss(
                        params_matrix,
                        subsampled_pulses,
                        subsampled_reference,
                        bounds_tuple,
                    )
                return batch_losses(
                    params_matrix,
                    objective_pulses,
                    objective_reference,
                    objective_time,
                    objective_omega,
                    lower_bounds,
                    upper_bounds,
                )
            loss_chunks = []
            for i in range(0, n_working_traces, opt_chunk_size):
                if use_model_superres and not precomputed_objective:
                    loss_chunks.append(
                        model.loss(
                            params_matrix[i : i + opt_chunk_size],
                            subsampled_pulses[i : i + opt_chunk_size],
                            subsampled_reference,
                            bounds_tuple,
                        )
                    )
                    continue
                loss_chunks.append(
                    batch_losses(
                        params_matrix[i : i + opt_chunk_size],
                        objective_pulses[i : i + opt_chunk_size],
                        objective_reference,
                        objective_time,
                        objective_omega,
                        lower_bounds,
                        upper_bounds,
                    )
                )
            return jnp.concatenate(loss_chunks, axis=0)

        initial_lr = None
        lr_widget = getattr(self, "lr", None)
        lr_widget_is_manual = bool(lr_widget is not None and not bool(getattr(lr_widget, "disabled", False)))
        current_lr_raw = getattr(lr_widget, "value", None)
        if lr_widget_is_manual and current_lr_raw is not None:
            try:
                current_lr = float(current_lr_raw)
            except (TypeError, ValueError):
                current_lr = None
            if current_lr is not None and np.isfinite(current_lr) and current_lr > 0:
                initial_lr = current_lr
        if initial_lr is None:
            lr_trace_count = min(n_working_traces, max(1, opt_chunk_size))
            try:
                if use_model_superres and not precomputed_objective:
                    lr_grads = model.gradients(
                        working_params[:lr_trace_count],
                        subsampled_pulses[:lr_trace_count],
                        subsampled_reference,
                        bounds_tuple,
                    )
                    grad_norm = float(jnp.linalg.norm(lr_grads))
                    if not np.isfinite(grad_norm) or grad_norm <= 1e-8:
                        initial_lr = 5e-3
                    else:
                        initial_lr = max(float(1e-2 / grad_norm), 5e-3)
                else:
                    initial_lr = float(
                        infer_initial_lr(
                            working_params[:lr_trace_count],
                            objective_pulses[:lr_trace_count],
                            objective_reference,
                            objective_time,
                            objective_omega,
                            lower_bounds,
                            upper_bounds,
                        )
                    )
            except Exception as lr_err:
                if self._is_resource_exhausted_error(lr_err):
                    initial_lr = 5e-3
                    self._notify(
                        "Auto LR fallback (OOM): using 5e-3",
                        level="warning",
                        duration=5000,
                    )
                else:
                    raise
            if hasattr(self, "lr") and hasattr(self.lr, "value"):
                self.lr.value = initial_lr

        tx, schedule = make_optax_optimizer(initial_lr, num_iterations, schedule_type)
        opt_state = tx.init(working_params)
        use_compiled_block = not use_chunked_opt and not (use_model_superres and not precomputed_objective)

        tol_widget_ref = getattr(self, "tol", None)
        tol_widget_is_manual = bool(tol_widget_ref is not None and not bool(getattr(tol_widget_ref, "disabled", False)))
        tol_raw = getattr(tol_widget_ref, "value", None)
        if tol_widget_ref is not None and hasattr(tol_widget_ref, "value") and not tol_widget_is_manual:
            tol_widget_ref.value = 0.0
        if tol_widget_is_manual and tol_raw is not None:
            try:
                tol_widget_value = float(tol_raw)
            except (TypeError, ValueError):
                tol_widget_value = None
            if tol_widget_value is not None and np.isfinite(tol_widget_value) and tol_widget_value > 0:
                tol_rel = tol_widget_value
        prev_check_loss = None
        small_improve_streak = 0
        last_rel_improvement = 0.0
        ref_rel_improvement = None
        auto_tol = None
        final_iteration = num_iterations

        # Flat-landscape detection: when traces are already nearly identical to reference
        # (SNR >> 1), the initial loss is tiny, gradients ≈ noise, and Adam drifts.
        # Reduce min_iters so early stopping can trigger before the drift accumulates.
        _initial_losses = _compute_losses(working_params)
        _initial_mean_loss = float(jnp.mean(_initial_losses))
        if _initial_mean_loss < 0.05:
            flat_min = max(50, num_iterations // 20)
            if flat_min < min_iters:
                min_iters = flat_min
                self._set_metric("flat_landscape_min_iters", int(min_iters))

        def _check_progress(params_matrix, iteration):
            nonlocal prev_check_loss, small_improve_streak, last_rel_improvement
            nonlocal ref_rel_improvement, auto_tol, final_iteration

            current_losses = _compute_losses(params_matrix)
            mean_loss = float(jnp.mean(current_losses))
            self._set_metric("current_loss", mean_loss)

            if prev_check_loss is None:
                prev_check_loss = mean_loss
                small_improve_streak = 0
            else:
                improvement = prev_check_loss - mean_loss
                rel_improvement = improvement / max(abs(prev_check_loss), 1e-12)
                if rel_improvement < 0:
                    rel_improvement = 0.0
                last_rel_improvement = float(rel_improvement)

                if ref_rel_improvement is None and rel_improvement > 0.0:
                    ref_rel_improvement = float(rel_improvement)
                    auto_tol = tol_rel * ref_rel_improvement
                    if hasattr(self, "tol") and hasattr(self.tol, "value"):
                        self.tol.value = auto_tol
                    self._set_metric("auto_tol_rel", auto_tol)
                    self._set_metric("ref_rel_improvement", ref_rel_improvement)

                threshold = auto_tol if auto_tol is not None else tol_rel
                if rel_improvement < threshold:
                    small_improve_streak += 1
                else:
                    small_improve_streak = 0

                self._set_metric("rel_improvement", last_rel_improvement)
                prev_check_loss = mean_loss

                if iteration >= min_iters and small_improve_streak >= patience:
                    final_iteration = iteration
                    self.status.object += (
                        f" | Early stop at it {iteration} "
                        f"(rel_improv={rel_improvement:.2e}, threshold={threshold:.2e})"
                    )
                    return True

            self._set_metric("current_lr", float(schedule(iteration)))
            self.status.object = f"JAX it {iteration}/{num_iterations} - loss {mean_loss:.3e}"
            self.progress.value = min(95, int(45 + 50 * (iteration / max(1, num_iterations))))
            return False

        with jax.default_device(device):
            if use_compiled_block:
                iteration = 0
                while iteration < num_iterations:
                    block_steps = min(check_every, num_iterations - iteration)
                    working_params, opt_state = optax_train_block(
                        tx,
                        working_params,
                        opt_state,
                        objective_pulses,
                        objective_reference,
                        objective_time,
                        objective_omega,
                        lower_bounds,
                        upper_bounds,
                        block_steps,
                    )
                    iteration += block_steps
                    if _check_progress(working_params, iteration):
                        break
            else:
                for iteration in range(1, num_iterations + 1):
                    grads = _compute_gradients(working_params)
                    working_params, opt_state = optax_step(tx, working_params, opt_state, grads)

                    if iteration % check_every == 0 or iteration == num_iterations:
                        if _check_progress(working_params, iteration):
                            break

        self._set_metric("last_rel_improvement", last_rel_improvement)
        self._set_metric("final_iteration", int(final_iteration))

        if trace_indices is not None:
            parameter_matrix = parameter_matrix.at[trace_indices].set(working_params)
            return parameter_matrix
        return working_params

    def _apply_final_correction(
        self,
        parameter_matrix,
        all_pulses,
        time_vector,
        angular_frequencies,
        lower_bounds,
        upper_bounds,
    ):
        """Apply corrections with optimal parameters."""
        optimal_parameters = squash_to_bounds(parameter_matrix, lower_bounds, upper_bounds)
        n_traces = int(all_pulses.shape[0])
        model = getattr(self, "_model", None)
        n_samples = int(
            getattr(model, "n_padded", all_pulses.shape[1])
            if model is not None
            else all_pulses.shape[1]
        )
        bytes_per_trace = n_samples * 8 * 3
        mb_per_trace = bytes_per_trace / (1024 ** 2)
        min_chunk = 1 if bool(model is not None and getattr(model, "superresolution", False)) else 10
        chunk_size = max(min_chunk, min(1000, int(500 / mb_per_trace)))
        n_chunks = max(1, (n_traces + chunk_size - 1) // chunk_size)
        report_every = max(1, n_chunks // 10)
        self._notify(
            f"Processing {n_traces} traces in chunks of {chunk_size} "
            f"(~{chunk_size * mb_per_trace:.1f} MB per chunk)"
        )

        corrected_chunks = []
        for chunk_idx, i in enumerate(range(0, n_traces, chunk_size)):
            pulse_chunk = all_pulses[i : i + chunk_size]
            param_chunk = optimal_parameters[i : i + chunk_size]
            model = getattr(self, "_model", None)
            if model is not None:
                corrected_chunk = model.apply(pulse_chunk, param_chunk)
            else:
                corrected_chunk = apply_batch_corrections(
                    pulse_chunk,
                    time_vector,
                    angular_frequencies,
                    param_chunk,
                )
            corrected_chunks.append(np.asarray(corrected_chunk))
            if chunk_idx % report_every == 0 or chunk_idx == n_chunks - 1:
                if hasattr(self, "progress"):
                    self.progress.value = int(80 + (chunk_idx / n_chunks) * 15)
                if hasattr(self, "status"):
                    self.status.object = f"Applying corrections: chunk {chunk_idx + 1}/{n_chunks}"

        if hasattr(self, "status"):
            self.status.object = "Merging corrected traces..."
        corrected_pulses = np.vstack(corrected_chunks)
        return corrected_pulses, optimal_parameters

    def _compute_ncm_after_optimization(self):
        """Compute NCM on corrected traces."""
        n_orig = getattr(self, "_superresolution_n_original", None)
        if n_orig is not None:
            self.corrected = np.asarray(self.corrected)[:, :n_orig]

        ncm_time_mode = str(getattr(self, "ncm_time_sampling_mode", None) and self.ncm_time_sampling_mode.value)
        if ncm_time_mode == "Sans":
            self._clear_ncm_state(clear_export_payload=True)
            self._set_ncm_backend_pane(backend="Disabled", device="Mode: Sans", backend_time="N/A")
            self._set_ncm_info_pane(["Skipped by user selection (`Sans`)."])
            self.ncm_progress.value = False
            self.status.object = "NCM skipped (mode: Sans)"
            self._refresh_matrix_analysis_views()
            return

        requested_ncm_method = str(self.ncm_method.value)
        backend_map = {"JAX (GPU)": "jax", "sklearn (CPU)": "sklearn", "Auto": "auto"}
        backend = backend_map.get(str(self.ncm_backend.value), "auto")
        self.status.object = f"Preparing NCM with {requested_ncm_method} [{backend}]..."
        self.ncm_progress.value = True
        try:
            corrected_np_full = np.asarray(self.corrected)
            n_traces, n_samples_full = corrected_np_full.shape
            if ncm_time_mode == "Custom":
                ncm_max_samples = max(32, int(self.ncm_max_samples.value)) if hasattr(self, "ncm_max_samples") else 800
            else:
                ncm_time_mode = "Max"
                # n_samples²×8 bytes for float64; cap so the matrix fits in ~500 MB
                _max_safe = max(32, int((500e6 / 8) ** 0.5))  # ≈7906
                ncm_max_samples = min(int(n_samples_full), _max_safe)
            ncm_stride = max(1, int(np.ceil(n_samples_full / ncm_max_samples)))
            corrected_np = corrected_np_full[:, ::ncm_stride]
            estimator = CovarianceEstimator(
                method=requested_ncm_method,
                backend=backend,
            )
            reference_path = getattr(self, "_active_reference_file", None)
            if reference_path is None:
                reference_selection = list(getattr(self.reference_file_selector, "value", []) or [])
                if reference_selection:
                    reference_path = Path(reference_selection[0])
            if reference_path is not None:
                ref_dataset = PulseDataset.from_hdf5(Path(reference_path))
                if ref_dataset is None:
                    raise ValueError("Failed to load reference file.")
                reference_np = np.asarray(ref_dataset.pulses, dtype=np.float64)[:, ::ncm_stride]
                estimator.fit(
                    corrected_np,
                    reference_traces=reference_np,
                    compute_precision=False,
                    compute_diagnostics=False,
                )
            else:
                estimator.fit(
                    corrected_np,
                    compute_precision=False,
                    compute_diagnostics=False,
                )

            self.ncm = estimator.ncm
            self.ncm_info = dict(estimator.info or {})
            self.ncm_sample = estimator.ncm_sample
            self.ncm_ref_sim = estimator.ncm_ref_sim
            self.ncm_info["n_traces_input"] = int(n_traces)
            self.ncm_info["n_samples_input"] = int(n_samples_full)
            self.ncm_info["n_samples_used"] = int(corrected_np.shape[1])
            self.ncm_info["time_decimation"] = int(ncm_stride)
            self.ncm_info["time_sampling_mode"] = ncm_time_mode
            self.ncm_info["reference_file"] = str(reference_path) if reference_path is not None else ""

            if self.ncm_sample is not None:
                self._ncm_combined_info_text = (
                    "**NCM = NCM_sample + NCM_ref_simulée**\n"
                    f"- NCM_sample  ‖F‖ = {np.linalg.norm(self.ncm_sample):.3e}\n"
                    f"- NCM_ref_sim ‖F‖ = {np.linalg.norm(self.ncm_ref_sim):.3e}"
                )
            else:
                self._ncm_combined_info_text = ""

            self.precision_matrix = compute_precision_matrix(self.ncm)
            if bool(getattr(self, "ncm_compute_diagnostics", None) and self.ncm_compute_diagnostics.value):
                self.matrix_diagnostics = compute_covariance_diagnostics(self.ncm, self.precision_matrix)
            else:
                self.matrix_diagnostics = None

            self.export_payload.update(
                {
                    "ncm": self.ncm,
                    "ncm_sample": self.ncm_sample,
                    "ncm_ref_sim": self.ncm_ref_sim,
                    "precision_matrix": self.precision_matrix,
                    "matrix_diagnostics": self.matrix_diagnostics,
                    "ncm_info": self.ncm_info,
                }
            )

            self._set_ncm_backend_pane(
                backend=str(self.ncm_info.get("backend", "N/A")),
                device=str(self.ncm_info.get("device", "N/A")),
                backend_time=self.ncm_info.get("time", "N/A"),
            )
            self._set_ncm_info_pane(
                [
                    f"Method: {self.ncm_info.get('method', requested_ncm_method)}",
                    f"Backend: {self.ncm_info.get('backend', 'N/A')}",
                    (
                        f"Samples: {self.ncm_info.get('n_samples_used', 'N/A')} / "
                        f"{self.ncm_info.get('n_samples_input', 'N/A')}"
                    ),
                    f"Time sampling mode: {self.ncm_info.get('time_sampling_mode', 'N/A')}",
                    f"Time decimation: x{self.ncm_info.get('time_decimation', 'N/A')}",
                    f"Shrinkage: {self.ncm_info.get('shrinkage', 'N/A')}",
                    f"Alpha: {self.ncm_info.get('alpha', 'N/A')}",
                ]
            )
            self.status.object = f"NCM computed ({self.ncm_info.get('method', requested_ncm_method)})"
        except Exception as ncm_err:
            self._clear_ncm_state(clear_export_payload=True)
            self._set_ncm_backend_pane(backend="N/A", device="N/A", backend_time="N/A")
            self._set_ncm_info_pane(
                [
                    f"Failed: {type(ncm_err).__name__}",
                    f"Message: {ncm_err}",
                ]
            )
            self._notify(f"NCM computation skipped: {type(ncm_err).__name__}", level="warning")
        finally:
            self.ncm_progress.value = False
            self._refresh_matrix_analysis_views()

    def _apply_periodic_after_optimization(self, device=None):
        """Apply periodic sampling correction on corrected traces (Correct-TDS order)."""
        if self.corrected is None or not bool(getattr(self, "cb_periodic", None) and self.cb_periodic.value):
            return
        if self.time is None or len(self.time) < 2:
            return

        try:
            freq_limit_thz = float(self.periodic_freq.value)
        except Exception:
            freq_limit_thz = 0.0
        periodic_mode_widget = getattr(self, "periodic_mode", None)
        periodic_mode = str(getattr(periodic_mode_widget, "value", "CPU")).strip().lower()

        try:
            corrected_np = np.asarray(self.corrected, dtype=float)
            mean_corr = np.mean(corrected_np, axis=0)
            periodic_diag = periodic_sampling_correct_tds(
                mean_corr,
                np.asarray(self.time, dtype=float),
                freq_limit_thz,
                mode=periodic_mode,
                device=device,
            )
            ct = np.asarray(periodic_diag.get("ct"), dtype=float)
            if ct.shape != mean_corr.shape:
                raise ValueError("Periodic correction has incompatible shape.")

            dt = float(self.time[1] - self.time[0])
            grad_traces = np.gradient(corrected_np, dt, axis=1)
            corrected_np = corrected_np - grad_traces * ct[None, :]
            self.corrected = corrected_np
            self.periodic_diagnostics = periodic_diag

            params = periodic_diag.get("params", {})
            self._set_metric("periodic_A", params.get("A"))
            self._set_metric("periodic_omega", params.get("omega"))
            self._set_metric("periodic_phi", params.get("phi"))
            self._set_metric("periodic_error", periodic_diag.get("error"))
            self._set_metric("periodic_optimizer", periodic_diag.get("optimizer"))
            self._notify("Periodic sampling correction applied after optimization.", level="info", duration=4500)
        except Exception as err:
            self._notify(
                f"Periodic sampling correction skipped: {type(err).__name__}",
                level="warning",
                duration=5000,
            )

    def run_optimization(self, event=None, raise_on_error: bool = False):
        """Run JAX optimization (orchestrator)."""
        try:
            if self.pulses is None:
                self.status.object = "Run the preview first."
                if raise_on_error:
                    raise ValueError("Preview must run before optimization.")
                return
            self.export_msg.visible = False
            self.export_msg.object = ""
            _ = self._release_previous_optimization_artifacts()
            self.error_box.visible = False
            self.progress.value = 45
            self.status.object = "Initializing optimization (JAX)..."
            import time as _t
            start_time = _t.perf_counter()

            computation_device, exact_match, lower_bounds, upper_bounds, parameter_matrix = self._setup_optimization()
            if exact_match:
                self.status.object = f"Computing on {computation_device.platform.upper()}"
            else:
                self.status.object = "GPU unavailable - using CPU"

            superres = bool(getattr(self, "superresolution_toggle", None) and self.superresolution_toggle.value)
            self._superresolution_n_original = None
            self._model = None
            model_kwargs = dict(
                device=computation_device,
                superresolution=superres,
                filter_low=bool(self.filter_low.value),
                filter_high=bool(self.filter_high.value),
                freq_start=float(self.freq_start.value),
                freq_end=float(self.freq_end.value),
                filter_sharpness=float(self.sharpness.value),
            )

            with jax.default_device(computation_device):
                if superres and self.pulses_time_base is not None:
                    opt_pulses_np = np.asarray(self.pulses_time_base, dtype=np.float32)
                    opt_reference_np = np.asarray(
                        self.ref_pulse_time if self.ref_pulse_time is not None else self.ref_pulse,
                        dtype=np.float32,
                    )
                else:
                    opt_pulses_np = np.asarray(self.pulses, dtype=np.float32)
                    opt_reference_np = np.asarray(self.ref_pulse, dtype=np.float32)

                all_pulses = jnp.asarray(opt_pulses_np, dtype=jnp.float32)
                reference_pulse = jnp.asarray(opt_reference_np, dtype=jnp.float32)
                time_vector = jnp.asarray(self.time, dtype=jnp.float32)
                dt_s = float(np.asarray(time_vector)[1] - np.asarray(time_vector)[0])
                self._superresolution_n_original = len(time_vector) if superres else None
                if self.freqs is not None:
                    angular_frequencies = jnp.asarray(2.0 * jnp.pi * self.freqs, dtype=jnp.float32)
                else:
                    base_dt = dt_s
                    angular_frequencies = (
                        jnp.fft.rfftfreq(self.time.shape[0], d=base_dt).astype(jnp.float32) * (2 * jnp.pi)
                    )

            (
                subsampled_pulses,
                subsampled_reference,
                sub_time_vector,
                sub_angular_frequencies,
                _indices,
            ) = self._subsample_data_for_optimization(
                all_pulses,
                reference_pulse,
                time_vector,
                angular_frequencies,
            )

            self._model = CorrectionModel(
                time_axis=sub_time_vector,
                **model_kwargs,
            ) if superres else None

            parameter_matrix = self._run_optimization_loop(
                computation_device,
                parameter_matrix,
                subsampled_pulses,
                subsampled_reference,
                sub_time_vector,
                sub_angular_frequencies,
                lower_bounds,
                upper_bounds,
            )

            self._model = CorrectionModel(
                time_axis=time_vector,
                **model_kwargs,
            ) if superres else None
            corrected_pulses, optimal_parameters = self._apply_final_correction(
                parameter_matrix,
                all_pulses,
                time_vector,
                angular_frequencies,
                lower_bounds,
                upper_bounds,
            )

            self.corrected = np.asarray(corrected_pulses)
            if superres:
                self.corrected = np.asarray(self.corrected)[:, self._model.output_slice]
                self.time = np.asarray(time_vector)[: self._model.n_original]
                self.freqs = np.fft.rfftfreq(self._model.n_original, d=dt_s)
            self.optimal_params = np.asarray(optimal_parameters)
            self._apply_periodic_after_optimization(device=computation_device)

            self._compute_ncm_after_optimization()

            self.update_plots_after_correction()
            total_duration = _t.perf_counter() - start_time
            ncm_suffix = f" | NCM {self.ncm_method.value}" if self.ncm is not None else ""
            self.status.object = (
                f"Optimization finished on {computation_device.device_kind.upper()} - "
                f"total {total_duration:.2f}s"
                f"{ncm_suffix}"
            )
            self.progress.value = 100
            self.btn_export.disabled = False
            self._notify("Optimization finished successfully.", level="success")

        except Exception as e:
            self.show_error(e, prefix="Optimization")
            self.status.object = "Error during optimization"
            if raise_on_error:
                raise


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
            corrected_freq = np.asarray(self.corrected)
            pulses_freq = self._apply_freq_filter_to_pulses(pulses_tf, freqs)
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

            self._unregister_plot(getattr(self, "_spec_lin", None))
            self._unregister_plot(getattr(self, "_spec_log", None))
            self._unregister_plot(getattr(self, "_std_lin", None))
            self._unregister_plot(getattr(self, "_std_log", None))

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
                    raw_std=self._db_scale(std_spec_raw_f),
                    corrected_std=self._db_scale(std_spec_corr_f),
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
                    self.plot_params_delay_hist,
                    build_parameter_histogram(
                        delays,
                        title="Delay histogram",
                        x_label="Count",
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
                self._set_plot(
                    self.plot_params_amp_hist,
                    build_parameter_histogram(
                        coef_a,
                        title="Coefficient a histogram",
                        x_label="Count",
                        y_label="Coefficient a",
                    ),
                )
            else:
                self._set_plot(self.plot_params_delay, None)
                self._set_plot(self.plot_params_delay_hist, None)
                self._set_plot(self.plot_params_amp, None)
                self._set_plot(self.plot_params_amp_hist, None)

            # --- Prepared data for export ---
            if not isinstance(self.export_payload, dict):
                self.export_payload = {}
            self.export_payload.update(
                {
                    "corrected_mean": (t_orig, mean_corr),
                    "corrected_std_time": (t_orig, corrected_std_time),
                    "corrected_std_freq": (freqs, std_spec_corr_f),
                }
            )

        except Exception as e:
            self.show_error(e, prefix="Plot update")

    @staticmethod
    def _to_builtin_json(value):
        """Convert NumPy/JAX values into JSON-serializable Python types."""
        if isinstance(value, dict):
            return {str(k): THzOptimizerApp._to_builtin_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [THzOptimizerApp._to_builtin_json(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def export_results(self, event=None, raise_on_error: bool = False):
        """Save corrected data, NCM matrices, and diagnostics to disk."""
        try:
            if not self.export_payload:
                raise ValueError("No corrected data available for export.")
            if self.current_file is None:
                raise ValueError("No source file associated with the export.")
            if self.corrected is None:
                raise ValueError("No corrected traces available for export.")
            if self.optimal_params is None:
                raise ValueError("No optimization parameters available for export.")

            target_dir = Path(Path(self.current_file).name).with_suffix("")
            target_dir.mkdir(parents=True, exist_ok=True)

            corrected_mean_tuple = self.export_payload.get("corrected_mean")
            corrected_std_time_tuple = self.export_payload.get("corrected_std_time")
            corrected_std_freq_tuple = self.export_payload.get("corrected_std_freq")

            if corrected_mean_tuple is None or not (isinstance(corrected_mean_tuple, tuple) and len(corrected_mean_tuple) == 2):
                raise ValueError("Missing 'corrected_mean' export series.")
            time_axis, corrected_mean = corrected_mean_tuple

            corrected_std_time = None
            if isinstance(corrected_std_time_tuple, tuple) and len(corrected_std_time_tuple) == 2:
                _, corrected_std_time = corrected_std_time_tuple

            frequency_axis = None
            corrected_std_freq = None
            if isinstance(corrected_std_freq_tuple, tuple) and len(corrected_std_freq_tuple) == 2:
                frequency_axis, corrected_std_freq = corrected_std_freq_tuple

            self.status.object = "Exporting NCM and precision matrix..."
            written_files = save_results(
                output_dir=target_dir,
                time_axis=np.asarray(time_axis),
                corrected_traces=np.asarray(self.corrected),
                optimal_params=np.asarray(self.optimal_params),
                file_prefix=target_dir.name,
                frequency_axis=np.asarray(frequency_axis) if frequency_axis is not None else None,
                corrected_mean=np.asarray(corrected_mean),
                corrected_std_time=(
                    np.asarray(corrected_std_time) if corrected_std_time is not None else None
                ),
                corrected_std_freq=(
                    np.asarray(corrected_std_freq) if corrected_std_freq is not None else None
                ),
                ncm=self.ncm,
                ncm_sample=self.ncm_sample,
                ncm_ref_sim=self.ncm_ref_sim,
                precision_matrix=self.precision_matrix,
                matrix_diagnostics=self.matrix_diagnostics,
                ncm_info=self.ncm_info,
            )

            file_names = ", ".join(p.name for p in written_files)
            resolved = target_dir.resolve()
            self.status.object = f"Exports saved in {resolved}: {file_names}"
            self.export_msg.object = f"Exported files to `{resolved}`: {file_names}"
            self.export_msg.visible = True
            self._notify("Export completed successfully.", level="success")
        except Exception as e:
            self.export_msg.visible = False
            self.show_error(e, prefix="Export")
            if raise_on_error:
                raise

    def _on_batch_files_selected(self, event):
        """Cache selected batch files and update status display."""
        raw_values = list(event.new or [])
        self._batch_file_pool = [Path(p).resolve() for p in raw_values]
        if not self._batch_file_pool:
            self.batch_selected_display.object = "**Batch files:** _none_"
            self.batch_status.object = "Batch idle."
            return
        preview = ", ".join(p.name for p in self._batch_file_pool[:6])
        if len(self._batch_file_pool) > 6:
            preview += f", ... (+{len(self._batch_file_pool) - 6})"
        self.batch_selected_display.object = (
            f"**Batch files:** {len(self._batch_file_pool)} selected ({preview})"
        )
        self.batch_status.object = f"{len(self._batch_file_pool)} file(s) selected. Click auto-link to build pairs."

    def _get_batch_linker(self) -> BatchLinker:
        """Create a batch linker from the current selected file pool."""
        return BatchLinker(getattr(self, "_batch_file_pool", []) or [])

    def auto_link_batch_pairs(self, event=None):
        """Auto-generate an editable batch mapping from selected files."""
        try:
            linker = self._get_batch_linker()
            if not linker.files:
                self.batch_status.object = "Select batch files first."
                return

            jobs, reference_only = linker.auto_link()
            self.batch_mapping_editor.value = BatchLinker.render_mapping_text(jobs, reference_only)

            linked = sum(1 for job in jobs if job.reference is not None)
            unlinked = len(jobs) - linked
            self.batch_status.object = (
                f"Auto-link complete: {len(jobs)} sample(s), "
                f"{linked} linked, {unlinked} without reference, {len(reference_only)} reference-only."
            )
        except Exception as err:
            self.show_error(err, prefix="Batch auto-link")

    def _set_active_paths_for_run(self, sample_path: Path, reference_path):
        """Set current sample/reference paths for one run."""
        self.current_file = Path(sample_path)
        self._active_reference_file = Path(reference_path) if reference_path is not None else None
        self.sample_file_display.object = f"**Sample selected:** `{self.current_file}`"
        if self._active_reference_file is not None:
            self.reference_file_display.object = f"**Reference selected:** `{self._active_reference_file}`"
        else:
            self.reference_file_display.object = "**Reference selected:** _none_"

    def _clear_loaded_state_for_batch(self):
        """Release heavy arrays between sequential batch jobs."""
        self._release_previous_optimization_artifacts()
        self.time = None
        self.time_orig = None
        self.pulses_raw = None
        self.mean_pulse_raw = None
        self.pulses = None
        self.pulses_time_base = None
        self.ref_index = None
        self.ref_pulse = None
        self.ref_pulse_time = None
        self.freqs = None
        self.periodic_diagnostics = None
        self._model = None
        self._superresolution_n_original = None
        self.btn_optimize.disabled = True
        self.btn_export.disabled = True
        gc.collect()
        clear_caches = getattr(jax, "clear_caches", None)
        if callable(clear_caches):
            clear_caches()

    def run_batch_processing(self, event=None):
        """Execute mapping jobs one by one to stay memory-safe."""
        try:
            self.error_box.visible = False
            self.export_msg.visible = False
            self.export_msg.object = ""

            parse_result = BatchLinker.parse_mapping_text(
                str(getattr(self.batch_mapping_editor, "value", "") or ""),
                getattr(self, "_batch_file_pool", []) or [],
            )
            if parse_result.errors:
                msg = "\n".join([f"- {line}" for line in parse_result.errors[:10]])
                if len(parse_result.errors) > 10:
                    msg += f"\n- ... {len(parse_result.errors) - 10} more"
                self.batch_status.object = f"Batch mapping has errors:\n{msg}"
                self.status.object = "Fix batch mapping errors before running."
                return
            if not parse_result.jobs:
                self.batch_status.object = "No sample job found. Add lines like `sample.h5 ==> ref.h5`."
                return

            total = len(parse_result.jobs)
            successes = []
            failures = []
            for index, job in enumerate(parse_result.jobs, start=1):
                self._clear_loaded_state_for_batch()
                self._set_active_paths_for_run(job.sample, job.reference)
                ref_name = job.reference.name if job.reference is not None else "none"
                self.batch_status.object = (
                    f"Running {index}/{total}: sample={job.sample.name}, reference={ref_name}"
                )
                try:
                    self.preview_analysis(None, raise_on_error=True)
                    self.run_optimization(None, raise_on_error=True)
                    self.export_results(None, raise_on_error=True)
                    successes.append(job.sample.name)
                except Exception as run_err:
                    failures.append((job.sample, run_err))
                    self._notify(
                        f"Batch failed for {job.sample.name}: {type(run_err).__name__}",
                        level="warning",
                        duration=5000,
                    )

            summary_lines = [f"Batch finished: {len(successes)}/{total} success."]
            if parse_result.reference_only:
                summary_lines.append(f"Reference-only entries kept: {len(parse_result.reference_only)}")
            if failures:
                summary_lines.append("Failures:")
                for sample_path, err in failures[:10]:
                    summary_lines.append(
                        f"- {sample_path.name}: {type(err).__name__} ({err})"
                    )
                if len(failures) > 10:
                    summary_lines.append(f"- ... {len(failures) - 10} more")
                self.status.object = f"Batch finished with errors ({len(successes)}/{total})."
            else:
                self.status.object = f"Batch completed successfully ({len(successes)}/{total})."
            self.batch_status.object = "\n".join(summary_lines)
        except Exception as err:
            self.show_error(err, prefix="Batch run")
            self.batch_status.object = f"Batch run failed: {type(err).__name__} ({err})"

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
            new_fs = self._make_h5_selector(new_dir)
            new_ref_fs = self._make_h5_selector(new_dir)
            new_batch_fs = self._make_h5_selector(new_dir)
            new_fs.param.watch(self.on_file_selected, "value")
            new_ref_fs.param.watch(self.on_reference_file_selected, "value")
            new_batch_fs.param.watch(self._on_batch_files_selected, "value")
            self.file_selector = new_fs
            self.reference_file_selector = new_ref_fs
            self.batch_file_selector = new_batch_fs
            self.sample_selector_panel[0] = new_fs
            self.reference_selector_panel[0] = new_ref_fs
            self.batch_panel[1] = new_batch_fs
            self.current_file = None
            self._active_reference_file = None
            self._batch_file_pool = []
            self.pulses_raw = None
            self.mean_pulse_raw = None
            self.sample_file_display.object = "**Sample selected:** _none_"
            self.reference_file_display.object = "**Reference selected:** _none_"
            self.batch_selected_display.object = "**Batch files:** _none_"
            self.batch_mapping_editor.value = ""
            self.batch_status.object = "Batch idle."
            self.status.object = f"Browsing drive `{new_drive}` ({new_dir})"
            self.update_filter_preview(None)
        except Exception as e:
            self.show_error(e, prefix="Drive change")

    def show(self):
        """Return the Panel layout."""
        return self.layout

def _make_session_app():
    app = THzOptimizerApp()
    view = app.show()
    setattr(view, "_thz_app_ref", app)

    def _cleanup_session(session_context):
        app._release_previous_optimization_artifacts()

    on_destroy = getattr(pn.state, "on_session_destroyed", None)
    if callable(on_destroy):
        on_destroy(_cleanup_session)
    return view


_make_session_app().servable()

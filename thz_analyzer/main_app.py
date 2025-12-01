import json
from pathlib import Path
import traceback

import holoviews as hv
import jax
import jax.numpy as jnp
import numpy as np
import panel as pn
from theme import (
    ALASKA_BLUE,
    ALASKA_NAVY,
    ALASKA_PRIMARY,
    ALASKA_SECONDARY,
    THEME_CSS,
)

pn.extension(raw_css=[THEME_CSS], notifications=True)
hv.extension("bokeh", theme="caliber")
pn.config.sizing_mode = "stretch_width"
try:
    pn.state.notifications.position = "bottom-right"
except Exception:
    pass

def _rfft_mag_single_sided(x):
    """
    Compute the single-sided normalized magnitude of a real FFT.
    Returns the magnitude with 2/N scaling and half DC/Nyquist bins.
    """

    x = np.asarray(x)
    N = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    mag = np.abs(X) * 2.0
    mag[..., 0] *= 0.5
    if N % 2 == 0:
        mag[..., -1] *= 0.5
    return mag

# Core modules
from core.io import load_h5_file, save_results
from core.filters import apply_frequency_filter, _compute_mask, apply_time_filter, _compute_time_mask
from core.optimization import (
    apply_corrections_batch,
    adam_batch_step,
    batched_gradients,
    batched_losses,
    squash_to_bounds,
)
from core.backend import resolve_device



 
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
        self.freqs = None
        self.corrected = None
        self.optimal_params = None
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
        # Export message displayed under the Export button
        self.export_msg = pn.pane.Markdown(visible=False)

        # Frequency filter configuration
        cfg = self.load_config()
        low_default = bool(cfg.get("filter_low", False))
        high_default = bool(cfg.get("filter_high", False))
        self.filter_low = pn.widgets.Switch(name="Filter lows (< Start)", value=low_default)
        self.filter_high = pn.widgets.Switch(name="Filter highs (> End)", value=high_default)
        self.freq_start = pn.widgets.TextInput(name="Start (Hz)", value=f"{float(cfg.get('freq_start', 0.18e12)):.1e}")
        self.freq_end = pn.widgets.TextInput(name="End (Hz)", value=f"{float(cfg.get('freq_end', 6e12)):.1e}")
        self.sharpness = pn.widgets.FloatInput(name="Sharpness", value=cfg.get("sharpness", 1.0), step=0.1, format="0.0#")
        self.scale_selector = pn.widgets.ToggleGroup(
            name="Scale", options=["Linear", "Log"], behavior="radio", value="Log"
        )
        self.filter_preview = pn.pane.HoloViews(height=180)
        self.time_filter_preview = pn.pane.HoloViews(height=180)

        # Correction parameters
        self.cb_delay = pn.widgets.Checkbox(name="Correct delay", value=True)
        self.cb_amplitude = pn.widgets.Checkbox(name="Correct amplitude (a)", value=True)
        self.cb_dilation = pn.widgets.Checkbox(name="Correct dilation (a)", value=False)
        self.limit_delay = pn.widgets.FloatInput(name="|delay|max (s)", value=1e-12, step=1e-13, format="0.0e")
        self.limit_amplitude_a = pn.widgets.FloatInput(name="|amplitude a|max", value=0.15, step=0.01)
        self.limit_dilation_a = pn.widgets.FloatInput(name="|dilation a|max", value=0.02, step=0.005)
        # Periodic sampling correction (expert)
        self.cb_periodic = pn.widgets.Checkbox(
            name="Correct periodic sampling?", value=bool(cfg.get("periodic_enable", False))
        )
        self.periodic_freq = pn.widgets.FloatInput(
            name="Frequency [THz]", value=float(cfg.get("periodic_freq_thz", 7.5)), step=0.1, format="0.0#"
        )

        # Optimization parameters
        self.maxiter = pn.widgets.IntInput(name="Iterations (Adam)", value=400, step=50)
        self.lr = pn.widgets.FloatInput(name="Learning rate", value=0.05, step=0.01)
        self.subsample = pn.widgets.IntSlider(name="Sub-sampling (x)", value=2, start=1, end=8, step=1)
        self.tol = pn.widgets.FloatInput(
            name="Early-stop tolerance", value=1e-5, step=1e-5, format="0.000010", width=200
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
        self.device_choice = "CPU"
        self.btn_cpu = pn.widgets.Button(name="CPU", button_type="primary")
        self.btn_gpu = pn.widgets.Button(name="GPU", button_type="default")
        self.btn_cpu.on_click(lambda *_: self._select_device("CPU"))
        self.btn_gpu.on_click(lambda *_: self._select_device("GPU"))

        # Plot panes
        self.plot_time = pn.pane.HoloViews(height=400)
        self.plot_std_time = pn.pane.HoloViews(height=350)
        self.spectrum_pane = pn.pane.HoloViews(height=300)
        self.std_spectrum_pane = pn.pane.HoloViews(height=300)
        self.plot_phase = pn.pane.HoloViews(height=350)
        self.plot_params_delay = pn.pane.HoloViews(height=300)
        self.plot_params_amp = pn.pane.HoloViews(height=300)

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
        self.drive_selector.param.watch(self._on_drive_change, "value")
        # Save config when periodic sampling controls change
        self.cb_periodic.param.watch(self.save_config, "value")
        self.periodic_freq.param.watch(self.save_config, "value")

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
                self.error_box,
                pn.Row(self.plot_time, self.plot_std_time),
                pn.Row(self.scale_selector),
                pn.Row(self.spectrum_pane, self.std_spectrum_pane),
                self.plot_phase,
                pn.Row(self.plot_params_delay, self.plot_params_amp),
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
            curve_f = hv.Curve((freqs_np, mask), "Frequency [Hz]", "Transmission").opts(
                title="Frequency filter preview", width=440, height=180, color=ALASKA_PRIMARY
            )
            self.filter_preview.object = curve_f

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
            curve_t = hv.Curve((t_axis, tmask), "Time [s]", "Transmission").opts(
                title="Time filter preview", width=440, height=180, color=ALASKA_BLUE
            )
            self.time_filter_preview.object = curve_t
        except Exception as e:
            self.show_error(e)

    @staticmethod
    def _db_scale(values, floor=1e-12):
        """Convert a spectrum to dB with a numerically stable floor."""
        return 20 * np.log10(np.maximum(values, floor))

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

    def _filter_many(self, freqs, *spectra):
        """Apply the active filter to several spectra in one shot."""
        return tuple(self._filter_spectrum(freqs, spec) for spec in spectra)

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
        """Run a preliminary analysis for the selected file."""
        try:
            self.error_box.visible = False
            self.progress.value = 0
            self.export_msg.visible = False
            self.export_msg.object = ""
            if not self.current_file:
                self.status.object = "Select an .h5 file first."
                return

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
            dt_fft = t_s[1] - t_s[0]
            if bool(self.cb_periodic.value):
                freq_thz = max(float(self.periodic_freq.value), 1e-9)
                dt_fft = 1.0 / (freq_thz * 1e12)
            freqs = np.fft.rfftfreq(min_len, d=dt_fft)

            self.progress.value = 15
            self.status.object = (
                f"Pre-processing... dt_raw={dt_raw:g}, scale_to_s={scale_to_s}, "
                f"fmax={freqs.max()/1e12:.2f} THz"
            )

            # Optional time-domain filtering
            pulses_time_filtered = apply_time_filter(
                t_s,
                pulses_array,
                bool(self.tfilter_low.value),
                bool(self.tfilter_high.value),
                float(self.t_start.value),
                float(self.t_end.value),
                float(self.t_sharpness.value),
            )

            # Reference trace: closest pulse to the mean (normalized dot product)
            mean_pulse = pulses_time_filtered.mean(axis=0)
            proj = pulses_array @ mean_pulse
            norms = np.einsum('ij,ij->i', pulses_array, pulses_array)
            ref_idx = int(np.argmin(np.abs(proj / (norms + 1e-30) - 1)))
            ref_pulse = pulses_time_filtered[ref_idx]

            self.progress.value = 25
            self.status.object = "Spectra & phases (preview)..."

            # FFT de toutes les traces
            fft_all = np.fft.rfft(pulses_time_filtered, axis=1)

            # --- Mean spectrum = |FFT(mean)| ---
            mean_spec = _rfft_mag_single_sided(pulses_array.mean(axis=0))
            ref_spec = _rfft_mag_single_sided(ref_pulse)

            # Magnitude spectral standard deviation
            std_spec = np.std(np.abs(fft_all), axis=0)

            # Apply frequency filters
            mean_spec_f, ref_spec_f, std_spec_f = self._filter_many(
                freqs,
                mean_spec,
                ref_spec,
                std_spec,
            )

            # Phases
            phase_all = np.unwrap(np.angle(fft_all))
            mean_phase = np.mean(phase_all, axis=0)
            ref_phase = np.unwrap(np.angle(np.fft.rfft(ref_pulse)))

            # --- Persist computed state ---
            self.time = t_s
            self.time_orig = t_orig
            self.freqs = freqs
            self.pulses = pulses_time_filtered
            self.ref_index = ref_idx
            self.ref_pulse = ref_pulse
            self.corrected = None
            self.export_payload = {}

            # --- Graphiques preview ---
            time_plots = [
                hv.Curve((t_orig, mean_pulse), "Time [orig units]", "Amp", label="Mean").opts(
                    width=900, height=350, color=ALASKA_PRIMARY
                ),
                hv.Curve((t_orig, ref_pulse), "Time [orig units]", "Amp", label="Ref").opts(
                    width=900, height=350, color=ALASKA_BLUE
                ),
            ]
            self.plot_time.object = hv.Overlay(time_plots).opts(title="Time pulses - Mean / Ref")

            # Raw temporal std
            self.plot_std_time.object = hv.Curve(
                (t_orig, pulses_array.std(axis=0)),
                "Time [orig units]", "Std"
            ).opts(title="Temporal standard deviation (raw)", width=900, height=300, color=ALASKA_SECONDARY)

            self._spec_lin = hv.Overlay([
                hv.Curve((freqs, mean_spec_f), "Frequency [Hz]", "E", label="Mean").opts(
                    width=900, height=300, color=ALASKA_PRIMARY
                ),
                hv.Curve((freqs, ref_spec_f), "Frequency [Hz]", "E", label="Ref").opts(
                    width=900, height=300, color=ALASKA_BLUE
                ),
            ]).opts(title="Spectra (linear)")

            self._spec_log = hv.Overlay([
                hv.Curve((freqs, self._db_scale(mean_spec_f)), "Frequency [Hz]", "E [dB]", label="Mean").opts(
                    width=900, height=300, color=ALASKA_PRIMARY
                ),
                hv.Curve((freqs, self._db_scale(ref_spec_f)), "Frequency [Hz]", "E [dB]", label="Ref").opts(
                    width=900, height=300, color=ALASKA_BLUE
                ),
            ]).opts(title="Spectra (log)")

            self._std_lin = hv.Curve(
                (freqs, std_spec_f), "Frequency [Hz]", "Std"
            ).opts(title="Spectral std dev (linear)", width=900, height=300, color=ALASKA_SECONDARY)

            self._std_log = hv.Curve(
                (freqs, self._db_scale(std_spec_f)), "Frequency [Hz]", "Std [dB]"
            ).opts(title="Spectral std dev (log)", width=900, height=300, color=ALASKA_SECONDARY)

            # Phases
            phase_plots = [
                hv.Curve((freqs, mean_phase), "Frequency [Hz]", "Phase", label="Mean").opts(
                    width=900, height=300, color=ALASKA_PRIMARY
                ),
                hv.Curve((freqs, ref_phase), "Frequency [Hz]", "Phase", label="Ref").opts(
                    width=900, height=300, color=ALASKA_BLUE
                ),
            ]
            self.plot_phase.object = hv.Overlay(phase_plots).opts(title="Phases")

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
                self.spectrum_pane.object = self._spec_lin
                self.std_spectrum_pane.object = self._std_lin
            else:
                self.spectrum_pane.object = self._spec_log
                self.std_spectrum_pane.object = self._std_log
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
        """Run the THz correction optimization with JAX (CPU or GPU)."""
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
            if bool(self.cb_periodic.value):
                freq_thz = max(float(self.periodic_freq.value), 1e-9)
                effective_dt = 1.0 / (freq_thz * 1e12)
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

            # === Application finale des corrections ===
            optimal_parameters = squash_to_bounds(parameter_matrix, lower_bounds, upper_bounds)
            corrected_pulses = apply_corrections_batch(
                all_pulses,
                time_vector,
                angular_frequencies,
                optimal_parameters,
            )

            compute_end = _t.perf_counter()
            compute_duration = compute_end - start_timestamp

            # === Conversion to NumPy for display and saving ===
            plot_start = _t.perf_counter()
            self.corrected = np.asarray(corrected_pulses)
            self.optimal_params = np.asarray(optimal_parameters)

            # === Update plots ===
            self.update_plots_after_correction()

            total_end = _t.perf_counter()
            plot_duration = total_end - plot_start
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
        """Update plots after the corrections have been applied."""
        try:
            if self.corrected is None:
                return
            pulses_array = self.pulses
            ref = self.ref_pulse
            t_orig = self.time_orig
            freqs = self.freqs
            mean_raw = pulses_array.mean(axis=0)
            mean_corr = self.corrected.mean(axis=0)

            time_plots = [
                hv.Curve((t_orig, mean_raw), "Time [orig units]", "Amp", label="Mean").opts(
                    width=900, height=350, color=ALASKA_PRIMARY
                ),
                hv.Curve((t_orig, ref), "Time [orig units]", "Amp", label="Ref").opts(
                    width=900, height=350, color=ALASKA_NAVY
                ),
                hv.Curve((t_orig, mean_corr), "Time [orig units]", "Amp", label="Corrected").opts(
                    width=900, height=350, color=ALASKA_BLUE
                )
            ]
            self.plot_time.object = hv.Overlay(time_plots).opts(title="Time pulses - Mean / Ref / Corrected")

            # Temporal standard deviation
            raw_std_time = pulses_array.std(axis=0)
            corrected_std_time = self.corrected.std(axis=0)

            std_plots = [
                hv.Curve((t_orig, raw_std_time), "Time [orig units]", "Std", label="Raw").opts(
                    width=900, height=300, color=ALASKA_SECONDARY
                ),
                hv.Curve((t_orig, corrected_std_time), "Time [orig units]", "Std", label="Corrected").opts(
                    width=900, height=300, color=ALASKA_BLUE
                )
            ]
            self.plot_std_time.object = hv.Overlay(std_plots).opts(title="Temporal std dev")

            # Spectres
            # Re-apply the time filter for spectral metrics
            pulses_tf = apply_time_filter(
                self.time,
                pulses_array,
                bool(self.tfilter_low.value),
                bool(self.tfilter_high.value),
                float(self.t_start.value),
                float(self.t_end.value),
                float(self.t_sharpness.value),
            )
            corrected_tf = apply_time_filter(
                self.time,
                self.corrected,
                bool(self.tfilter_low.value),
                bool(self.tfilter_high.value),
                float(self.t_start.value),
                float(self.t_end.value),
                float(self.t_sharpness.value),
            )

            fft_all = np.fft.rfft(pulses_tf, axis=1)
            fft_all_corr = np.fft.rfft(corrected_tf, axis=1)

            mean_spec_raw = _rfft_mag_single_sided(mean_raw)
            mean_spec_corr = _rfft_mag_single_sided(mean_corr)

            ref_spec = _rfft_mag_single_sided(ref)

            mean_spec_raw_f, ref_spec_f, mean_spec_corr_f = self._filter_many(
                freqs,
                mean_spec_raw,
                ref_spec,
                mean_spec_corr,
            )

            if self.scale_selector.value == "Linear":
                spec_plots = [
                    hv.Curve((freqs, mean_spec_raw_f), "Frequency [Hz]", "E", label="Mean").opts(
                        width=900, height=300, color=ALASKA_PRIMARY
                    ),
                    hv.Curve((freqs, ref_spec_f), "Frequency [Hz]", "E", label="Ref").opts(
                        width=900, height=300, color=ALASKA_NAVY
                    ),
                    hv.Curve((freqs, mean_spec_corr_f), "Frequency [Hz]", "E", label="Corrected").opts(
                        width=900, height=300, color=ALASKA_BLUE
                    )
                ]
                self._spec_lin = hv.Overlay(spec_plots).opts(title="Spectra (linear)")
            else:
                spec_plots = [
                    hv.Curve((freqs, self._db_scale(mean_spec_raw_f)), "Frequency [Hz]", "E [dB]", label="Mean").opts(
                        width=900, height=300, color=ALASKA_PRIMARY
                    ),
                    hv.Curve((freqs, self._db_scale(ref_spec_f)), "Frequency [Hz]", "E [dB]", label="Ref").opts(
                        width=900, height=300, color=ALASKA_NAVY
                    ),
                    hv.Curve((freqs, self._db_scale(mean_spec_corr_f)), "Frequency [Hz]", "E [dB]", label="Corrected").opts(
                        width=900, height=300, color=ALASKA_BLUE
                    )
                ]
                self._spec_log = hv.Overlay(spec_plots).opts(title="Spectra (log)")

            # Spectral standard deviation
            std_spec_raw = np.std(np.abs(fft_all), axis=0)
            std_spec_corr = np.std(np.abs(fft_all_corr), axis=0)

            std_spec_raw_f, std_spec_corr_f = self._filter_many(
                freqs,
                std_spec_raw,
                std_spec_corr,
            )

            if self.scale_selector.value == "Linear":
                std_plots = [
                    hv.Curve((freqs, std_spec_raw_f), "Frequency [Hz]", "Std", label="Raw").opts(
                        width=900, height=300, color=ALASKA_SECONDARY
                    ),
                    hv.Curve((freqs, std_spec_corr_f), "Frequency [Hz]", "Std", label="Corrected").opts(
                        width=900, height=300, color=ALASKA_BLUE
                    )
                ]
                self._std_lin = hv.Overlay(std_plots).opts(title="Spectral std dev (linear)")
            else:
                std_plots = [
                    hv.Curve((freqs, self._db_scale(std_spec_raw_f)), "Frequency [Hz]", "Std [dB]", label="Raw").opts(
                        width=900, height=300, color=ALASKA_SECONDARY
                    ),
                    hv.Curve((freqs, self._db_scale(std_spec_corr_f)), "Frequency [Hz]", "Std [dB]", label="Corrected").opts(
                        width=900, height=300, color=ALASKA_BLUE
                    )
                ]
                self._std_log = hv.Overlay(std_plots).opts(title="Spectral std dev (log)")

            # Phases
            phase_mean_raw = np.unwrap(np.angle(np.fft.rfft(mean_raw)))
            phase_ref = np.unwrap(np.angle(np.fft.rfft(ref)))
            phase_mean_corr = np.unwrap(np.angle(np.fft.rfft(mean_corr)))

            phase_plots = [
                hv.Curve((freqs, phase_mean_raw), "Frequency [Hz]", "Phase", label="Mean").opts(
                    width=900, height=300, color=ALASKA_PRIMARY
                ),
                hv.Curve((freqs, phase_ref), "Frequency [Hz]", "Phase", label="Ref").opts(
                    width=900, height=300, color=ALASKA_NAVY
                ),
                hv.Curve((freqs, phase_mean_corr), "Frequency [Hz]", "Phase", label="Corrected").opts(
                    width=900, height=300, color=ALASKA_BLUE
                )
            ]
            self.plot_phase.object = hv.Overlay(phase_plots).opts(title="Phases")
            self.switch_scale(None)

            # --- Correction parameters (Delay and amplitude coef a) ---
            if self.optimal_params is not None and len(self.optimal_params) == pulses_array.shape[0]:
                idx = np.arange(self.optimal_params.shape[0])
                # Delay
                delays = self.optimal_params[:, 0]
                delay_curve = hv.Curve((idx, delays), "Trace index", "Delay [s]").opts(
                    width=440, height=300, color=ALASKA_BLUE, title="Delay"
                )
                delay_points = hv.Scatter((idx, delays), "Trace index", "Delay [s]").opts(
                    color=ALASKA_BLUE, size=4
                )
                delay_ref = hv.Scatter((
                    [int(self.ref_index)], [float(delays[int(self.ref_index)])]
                ), "Trace index", "Delay [s]", label="Reference").opts(color="red", marker="triangle", size=12)
                self.plot_params_delay.object = hv.Overlay([delay_curve, delay_points, delay_ref])

                # Amplitude coefficient a
                coef_a = self.optimal_params[:, 1]
                a_curve = hv.Curve((idx, coef_a), "Trace index", "Coef a").opts(
                    width=440, height=300, color=ALASKA_BLUE, title="Coef a - amplitude"
                )
                a_points = hv.Scatter((idx, coef_a), "Trace index", "Coef a").opts(
                    color=ALASKA_BLUE, size=4
                )
                a_ref = hv.Scatter((
                    [int(self.ref_index)], [float(coef_a[int(self.ref_index)])]
                ), "Trace index", "Coef a", label="Reference").opts(color="red", marker="triangle", size=12)
                self.plot_params_amp.object = hv.Overlay([a_curve, a_points, a_ref])

            # --- Prepared data for export ---
            self.export_payload = {
                "corrected_mean": (t_orig, mean_corr),
                "corrected_std_time": (t_orig, corrected_std_time),
                "corrected_std_freq": (freqs, std_spec_corr_f),
            }

        except Exception as e:
            self.show_error(e, prefix="Plot update")

    def export_results(self, event):
        """Export corrected metrics to TXT files."""
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


    

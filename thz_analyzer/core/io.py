import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .utils import validate_2d_array

logger = logging.getLogger(__name__)

__all__ = [
    "PulseDataset",
    "load_h5_file",
    "save_results",
    "export_predictor",
    "load_predictor",
]


@dataclass
class PulseDataset:
    pulses: np.ndarray
    timeaxis: np.ndarray

    @classmethod
    def from_hdf5(cls, file_path: Path) -> Optional["PulseDataset"]:
        data = load_h5_file(file_path)
        if data is None:
            return None
        pulses, timeaxis = data
        stacked = np.vstack(pulses)
        return cls(stacked, timeaxis)

    def choose_reference_index(self) -> int:
        """Return the trace closest to the mean."""
        mean = np.mean(self.pulses, axis=0)
        pseudo_norm = self.pulses @ mean
        self_norm = np.einsum("ij,ij->i", self.pulses, self.pulses)
        proximity = pseudo_norm / np.maximum(self_norm, 1e-30)
        return int(np.argmin(np.abs(proximity - 1.0)))


def load_h5_file(file_path: Path | str) -> Optional[tuple[list[np.ndarray], np.ndarray]]:
    """Load time traces from an HDF5 file."""
    try:
        with h5py.File(file_path, "r") as f:
            if "time-traces" in f:
                trace_group = f["time-traces"]
            else:
                trace_group = f

            if "timeaxis" in trace_group:
                timeaxis = np.array(trace_group["timeaxis"])
            elif "timeaxis" in f:
                timeaxis = np.array(f["timeaxis"])
            else:
                logger.error("File %s does not contain 'timeaxis'", file_path)
                return None

            keys = sorted((k for k in trace_group.keys() if k.isdigit()), key=int)
            if not keys:
                logger.error("No numeric trace datasets found in %s", file_path)
                return None

            pulses = [np.array(trace_group[k]) for k in keys]

        return pulses, timeaxis

    except Exception:
        logger.exception("Failed to load HDF5 %s", file_path)
        return None


def _to_json_serializable(value: Any) -> Any:
    """Convert NumPy values to JSON-friendly objects."""
    if isinstance(value, dict):
        return {str(key): _to_json_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _save_series_txt(
    *,
    output_dir: Path,
    name: str,
    axis: np.ndarray,
    values: np.ndarray,
    file_stem: Optional[str] = None,
    include_header: bool = True,
) -> Path:
    """Save a 1D series as a two-column text file."""
    axis_arr = np.asarray(axis, dtype=np.float64).ravel()
    values_arr = np.asarray(values, dtype=np.float64).ravel()
    if axis_arr.shape != values_arr.shape:
        raise ValueError(f"Series '{name}' axis and values must have the same shape.")
    data = np.column_stack((axis_arr, values_arr))
    output_name = file_stem or name
    file_path = output_dir / f"{output_name}.txt"
    if include_header:
        header = f"{name}_axis\t{name}_value"
        np.savetxt(file_path, data, delimiter="\t", header=header)
    else:
        np.savetxt(file_path, data, delimiter="\t")
    return file_path


def save_results(
    output_dir: Path | str,
    time_axis: np.ndarray,
    corrected_traces: np.ndarray,
    optimal_params: np.ndarray,
    frequency_axis: Optional[np.ndarray] = None,
    corrected_mean: Optional[np.ndarray] = None,
    corrected_std_time: Optional[np.ndarray] = None,
    corrected_std_freq: Optional[np.ndarray] = None,
    ncm: Optional[np.ndarray] = None,
    ncm_sample: Optional[np.ndarray] = None,
    ncm_ref_sim: Optional[np.ndarray] = None,
    precision_matrix: Optional[np.ndarray] = None,
    matrix_diagnostics: Optional[dict] = None,
    ncm_info: Optional[dict] = None,
    file_prefix: str = "",
) -> list[Path]:
    """Save the correction outputs and optional covariance data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces_arr = validate_2d_array(corrected_traces, "corrected_traces")
    n_traces, n_samples = traces_arr.shape

    time_arr = np.asarray(time_axis, dtype=np.float64).ravel()
    if time_arr.ndim != 1 or time_arr.size != n_samples:
        raise ValueError(
            "time_axis must be 1-D with length matching corrected_traces.shape[1]."
        )

    params_arr = np.asarray(optimal_params, dtype=np.float64)
    if params_arr.ndim != 2 or params_arr.shape[0] != n_traces:
        raise ValueError(
            "optimal_params must be 2-D with the same number of rows as corrected_traces."
        )

    mean_arr = (
        np.asarray(corrected_mean, dtype=np.float64).ravel()
        if corrected_mean is not None
        else traces_arr.mean(axis=0)
    )
    std_time_arr = (
        np.asarray(corrected_std_time, dtype=np.float64).ravel()
        if corrected_std_time is not None
        else traces_arr.std(axis=0)
    )

    if frequency_axis is None:
        if n_samples >= 2:
            dt = float(time_arr[1] - time_arr[0])
            if not np.isfinite(dt) or dt == 0.0:
                raise ValueError("Invalid time axis spacing: cannot compute frequency axis.")
            freq_arr = np.fft.rfftfreq(n_samples, d=dt)
        else:
            freq_arr = np.array([0.0], dtype=np.float64)
    else:
        freq_arr = np.asarray(frequency_axis, dtype=np.float64).ravel()

    if corrected_std_freq is None:
        fft_vals = np.fft.rfft(traces_arr, axis=1)
        std_freq_arr = np.abs(np.std(fft_vals, axis=0))
    else:
        std_freq_arr = np.asarray(corrected_std_freq, dtype=np.float64).ravel()

    if mean_arr.shape != time_arr.shape:
        raise ValueError("corrected_mean must have the same shape as time_axis.")
    if std_time_arr.shape != time_arr.shape:
        raise ValueError("corrected_std_time must have the same shape as time_axis.")
    if std_freq_arr.shape != freq_arr.shape:
        raise ValueError("corrected_std_freq must have the same shape as frequency_axis.")

    written_files: list[Path] = []
    prefix = f"{file_prefix.strip()}_" if file_prefix and file_prefix.strip() else ""

    written_files.append(
        _save_series_txt(
            output_dir=output_dir,
            name="corrected_mean",
            axis=time_arr,
            values=mean_arr,
            file_stem=f"{prefix}corrected_mean",
            include_header=False,
        )
    )
    written_files.append(
        _save_series_txt(
            output_dir=output_dir,
            name="corrected_std_time",
            axis=time_arr,
            values=std_time_arr,
            file_stem=f"{prefix}corrected_std_time",
        )
    )
    written_files.append(
        _save_series_txt(
            output_dir=output_dir,
            name="corrected_std_freq",
            axis=freq_arr,
            values=std_freq_arr,
            file_stem=f"{prefix}corrected_std_freq",
        )
    )

    params_path = output_dir / f"{prefix}optimal_params.txt"
    col_names = ["delay", "amplitude_a", "dilation_a"]
    if params_arr.shape[1] > len(col_names):
        col_names.extend([f"param_{idx}" for idx in range(len(col_names), params_arr.shape[1])])
    header_cols = "\t".join(col_names[: params_arr.shape[1]])
    np.savetxt(params_path, params_arr, delimiter="\t", fmt="%.6e", header=header_cols)
    written_files.append(params_path)

    traces_h5_path = output_dir / f"{prefix}corrected_traces.h5"
    with h5py.File(traces_h5_path, "w") as hdf:
        hdf.create_dataset("timeaxis", data=time_arr)
        hdf.create_dataset("corrected_traces", data=traces_arr, compression="gzip", compression_opts=4)
    written_files.append(traces_h5_path)

    ncm_metadata = dict(ncm_info or {})
    if ncm is not None:
        ncm_arr = np.asarray(ncm, dtype=np.float64)
        if ncm_arr.ndim != 2:
            raise ValueError("ncm must be a 2-D array.")

        hdf5_path = output_dir / f"{prefix}noise_covariance_matrix.h5"
        with h5py.File(hdf5_path, "w") as hdf:
            dataset = hdf.create_dataset(
                "noise_covariance_matrix",
                data=ncm_arr,
                compression="gzip",
                compression_opts=4,
            )
            dataset.attrs["shape"] = ncm_arr.shape
            dataset.attrs["method"] = ncm_metadata.get("method", "unknown")
            if "shrinkage" in ncm_metadata:
                dataset.attrs["shrinkage"] = ncm_metadata["shrinkage"]
            if ncm_sample is not None:
                hdf.create_dataset(
                    "ncm_sample",
                    data=np.asarray(ncm_sample, dtype=np.float64),
                    compression="gzip", compression_opts=4,
                )
            if ncm_ref_sim is not None:
                hdf.create_dataset(
                    "ncm_ref_simulated",
                    data=np.asarray(ncm_ref_sim, dtype=np.float64),
                    compression="gzip", compression_opts=4,
                )

            if precision_matrix is not None:
                precision_arr = np.asarray(precision_matrix, dtype=np.float64)
                if precision_arr.ndim != 2:
                    raise ValueError("precision_matrix must be a 2-D array.")
                prec_dataset = hdf.create_dataset(
                    "precision_matrix",
                    data=precision_arr,
                    compression="gzip",
                    compression_opts=4,
                )
                prec_dataset.attrs["shape"] = precision_arr.shape
        written_files.append(hdf5_path)

        ncm_txt_path = output_dir / f"{prefix}ncm_full.txt"
        np.savetxt(
            ncm_txt_path,
            ncm_arr,
            fmt="%.6e",
            header=(
                f"Noise Covariance Matrix ({ncm_arr.shape[0]}x{ncm_arr.shape[1]})\n"
                f"Method: {ncm_metadata.get('method', 'unknown')}"
            ),
        )
        written_files.append(ncm_txt_path)

        ncm_diag_path = output_dir / f"{prefix}ncm_diagonal.txt"
        np.savetxt(
            ncm_diag_path,
            np.diag(ncm_arr),
            fmt="%.6e",
            header="NCM Diagonal Elements",
        )
        written_files.append(ncm_diag_path)

        if precision_matrix is not None:
            precision_arr = np.asarray(precision_matrix, dtype=np.float64)
            precision_txt_path = output_dir / f"{prefix}precision_full.txt"
            np.savetxt(
                precision_txt_path,
                precision_arr,
                fmt="%.6e",
                header=f"Precision Matrix ({precision_arr.shape[0]}x{precision_arr.shape[1]})",
            )
            written_files.append(precision_txt_path)

            precision_diag_path = output_dir / f"{prefix}precision_diagonal.txt"
            np.savetxt(
                precision_diag_path,
                np.diag(precision_arr),
                fmt="%.6e",
                header="Precision Matrix Diagonal Elements",
            )
            written_files.append(precision_diag_path)

    if matrix_diagnostics is not None:
        diag_path = output_dir / f"{prefix}matrix_diagnostics.json"
        serializable_diag = _to_json_serializable(matrix_diagnostics)
        payload = {
            "diagnostics": serializable_diag,
            "ncm_info": _to_json_serializable(ncm_metadata),
            "timestamp": datetime.now().isoformat(),
        }
        diag_path.write_text(json.dumps(payload, indent=2))
        written_files.append(diag_path)

    return written_files


def export_predictor(
    output_path: str,
    ncm: np.ndarray,
    precision_matrix: np.ndarray,
    metadata: dict,
) -> None:
    """Export a predictor file."""
    path = Path(output_path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    ncm_arr = np.asarray(ncm, dtype=np.float64)
    precision_arr = np.asarray(precision_matrix, dtype=np.float64)
    if ncm_arr.ndim != 2:
        raise ValueError("ncm must be a 2-D array.")
    if precision_arr.ndim != 2:
        raise ValueError("precision_matrix must be a 2-D array.")
    if ncm_arr.shape != precision_arr.shape:
        raise ValueError("ncm and precision_matrix must have the same shape.")

    metadata_payload = dict(metadata or {})
    metadata_payload["export_timestamp"] = datetime.now().isoformat()
    metadata_payload["export_version"] = "1.0"

    metadata_json = json.dumps(_to_json_serializable(metadata_payload), indent=2)

    np.savez_compressed(
        path,
        ncm=ncm_arr,
        precision=precision_arr,
        metadata=metadata_json,
    )

    logger.info("Predictor exported to %s", path)
    logger.info("NCM shape: %s", ncm_arr.shape)
    logger.info("Method: %s", metadata_payload.get("method", "unknown"))
    logger.info("Backend: %s", metadata_payload.get("backend", "unknown"))


def load_predictor(input_path: str) -> dict:
    """Load a predictor file."""
    path = Path(input_path)
    with np.load(path, allow_pickle=False) as data:
        if "ncm" not in data or "precision" not in data or "metadata" not in data:
            raise ValueError(
                "Invalid predictor format: expected keys 'ncm', 'precision', 'metadata'."
            )
        ncm_arr = np.asarray(data["ncm"], dtype=np.float64)
        precision_arr = np.asarray(data["precision"], dtype=np.float64)
        metadata_raw = data["metadata"]
        metadata_json = str(metadata_raw.item() if hasattr(metadata_raw, "item") else metadata_raw)

    metadata = json.loads(metadata_json)

    return {
        "ncm": ncm_arr,
        "precision": precision_arr,
        "metadata": metadata,
    }

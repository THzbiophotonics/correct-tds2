"""Utilities to load and save THz datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PulseDataset", "load_h5_file", "save_results"]


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


def load_h5_file(file_path: Path | str) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
    """Load an HDF5 file containing a ``timeaxis`` dataset and numbered traces."""
    try:
        with h5py.File(file_path, "r") as f:
            if "timeaxis" not in f:
                logger.error("File %s does not contain 'timeaxis'", file_path)
                return None
            timeaxis = np.array(f["timeaxis"])
            keys = sorted((k for k in f.keys() if k.isdigit()), key=int)
            if not keys:
                logger.error("No numeric trace datasets found in %s", file_path)
                return None
            pulses = [np.array(f[k]) for k in keys]
        return pulses, timeaxis
    except Exception:
        logger.exception("Failed to load HDF5 %s", file_path)
        return None


def save_results(series_map, output_dir: Path | str):
    """
    Persist computed series to TXT files that can be read by other tools.

    Args:
        series_map: Mapping of ``name -> (axis, values)`` pairs.
        output_dir: Directory where the TXT files will be created.

    Returns:
        list[Path]: Paths to the generated files.
    """
    if not series_map:
        raise ValueError("No series provided for export.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for name, series in series_map.items():
        if series is None:
            continue
        axis, values = series
        axis_arr = np.asarray(axis)
        values_arr = np.asarray(values)
        if axis_arr.shape != values_arr.shape:
            raise ValueError(f"Series '{name}' axis and values must have the same shape.")
        data = np.column_stack((axis_arr, values_arr))
        header = f"{name}_axis\t{name}_value"
        file_path = output_dir / f"{name}.txt"
        np.savetxt(file_path, data, delimiter="\t", header=header, comments="")
        written_files.append(file_path)
    return written_files

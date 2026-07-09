# Correct-TDS2 (Panel + JAX)

A JAX-accelerated correction tool for THz time-domain spectroscopy traces, built with Panel.

## Overview

Terahertz time-domain spectroscopy (THz-TDS) measurements are affected
by subtle time-base distortions — timing jitter (delay), amplitude
drift, and dilation/stretching of the acquired trace — that bias
signal estimation and the material parameters extracted from it.
Correct-TDS2 estimates and corrects these distortions: it aligns a set
of traces against a reference, fits delay / amplitude / dilation
correction parameters by numerical optimization, and estimates the
noise covariance matrix (NCM) needed to quantify uncertainty on the
corrected signal.

This implementation follows the signal model and correction approach
described in Denakpo et al. (2025) — see [Citation](#citation) — and
reimplements it as an interactive **Panel** web app with **JAX**-based
optimization, so the correction can run on CPU or be GPU-accelerated
for large batches of traces.

## 📚 Documentation

| Guide | What it covers |
|---|---|
| [Installation](#installation) & [Launch](#launch) (below) | Quick setup, CPU or GPU, first run |
| [GPU on Windows via WSL](INSTALL_GPU_WSL.md) | JAX has no native Windows GPU build — set up WSL + CUDA to unlock GPU acceleration on a Windows machine |
| [Batch mode](BATCH_MODE.md) | Run the full correction pipeline on many `.h5` files at once |
| [FAQ / Troubleshooting](FAQ.md) | Poetry not found, GPU not detected, WSL install issues |

## Installation

- Requirements: Python 3.11+, Poetry installed (`pipx install poetry` or `pip install poetry`).
- Install project dependencies:
  - `poetry install`
- JAX is required (choose based on your machine):
  - CPU only: `poetry run pip install -U "jax[cpu]"`
  - NVIDIA GPU (pick the line matching your installed CUDA):
    - CUDA 13.x: `poetry run pip install -U "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
    - CUDA 12.x: `poetry run pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
    - CUDA 11.x: `poetry run pip install -U "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

> **On Windows**, JAX has no native GPU build — the commands above only give you CPU. To use your NVIDIA GPU, follow the [GPU on Windows via WSL](INSTALL_GPU_WSL.md) guide instead.

## Installation without Poetry

If Poetry is not available, you can install directly with `pip` inside a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install numpy "holoviews>=1.21,<2" "bokeh>=3.9" "panel>=1.8" "h5py>=3.15,<4" "scikit-learn>=1.4,<2" "optax>=0.2.8,<0.3"

# JAX — choose based on your machine:
pip install -U "jax[cpu]"                                                                           # CPU only
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # GPU CUDA 12
pip install -U "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # GPU CUDA 13

# Then launch the app
panel serve thz_analyzer/main_app.py --show --autoreload
```

## Launch

- Start the app:
  - `poetry run panel serve thz_analyzer/main_app.py --show --autoreload`
- Out-of-memory on GPU because JAX preallocates memory upfront? Add `--args --no-preallocate` (must come last):
  - `poetry run panel serve thz_analyzer/main_app.py --show --autoreload --args --no-preallocate`

## First Test

- Start the app (command above) — your browser opens the Panel UI.
- In “Choose a .h5 file”, select a THz traces HDF5 file.
- Optionally adjust Frequency/Time filters and the “Scale” (Linear/Log).
- Click “Analyze (preview)” to see mean, spectra, and phases.
- Select the device (CPU or GPU) and click “Optimize (JAX)”.
- When finished, “Corrected” plots and correction parameters (delay, coef a) are shown.
- Export: “Export results (.txt)” writes text files in a folder named after the .h5 file.

Notes
- The first GPU run can be slower due to JIT compilation; subsequent runs are faster.
- The status shows separate timings for compute vs plots to help diagnose performance.

Need help? See the [documentation index](#-documentation) above.

## Citation

If you use Correct-TDS2 in your research, please cite the paper it
implements:

> E. Denakpo, T. Hannotte, N. Osseiran, F. Orieux and R. Peretti,
> "Signal estimation and uncertainties extraction in TeraHertz Time
> Domain Spectroscopy," *IEEE Transactions on Instrumentation and
> Measurement*, 2025. Preprint: [arXiv:2410.08587](https://arxiv.org/abs/2410.08587)



## License

[MIT](LICENSE)

Florian Letertre THzBiophotonics — IEMN CNRS

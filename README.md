# Correct-TDS2 (Panel + JAX)

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

[FAQ](FAQ.md)


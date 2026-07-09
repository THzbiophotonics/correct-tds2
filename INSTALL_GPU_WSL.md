# 🐧 GPU on Windows via WSL — Installation Guide

## Why doesn't the GPU work directly on Windows?

JAX no longer ships a CUDA build that works natively on Windows:
the `jax[cudaXX]` packages (GPU acceleration) only install and run
on Linux.

As a result:

- **On native Windows** → Correct-TDS2 still works, but optimization
  runs on **CPU** only. It works, just slower.
- **On Linux (or WSL, which runs a real Linux kernel inside Windows)**
  → you can install `jax[cuda13]`/`jax[cuda12]` and get **GPU**
  acceleration.

**WSL (Windows Subsystem for Linux)** is therefore the easiest way to
get GPU support while staying on Windows: it installs an Ubuntu system
that runs alongside Windows and can access your NVIDIA card.

> Prerequisite: an **NVIDIA** GPU with up-to-date Windows drivers (the
> regular Windows NVIDIA driver is enough — you don't need to install
> a separate CUDA driver inside WSL, it's provided through the Windows
> driver via WSL).

------------------------------------------------------------------------

## 🪜 Step 1 — Install WSL and Ubuntu

Open **PowerShell as administrator** and run, in order:

```powershell
wsl --update
wsl --install -d Ubuntu-24.04
```

- `wsl --update` updates the WSL engine itself (required for GPU
  support).
- `wsl --install -d Ubuntu-24.04` installs a dedicated Ubuntu 24.04
  distribution.

On first launch, Ubuntu will ask you to create a **username** and
**password** for Linux (separate from your Windows account) — this is
expected.

> If Windows asks you to **restart**, do so before continuing.

------------------------------------------------------------------------

## 🪜 Step 2 — Install the base tools inside Ubuntu

Once you're in the Ubuntu terminal (it opens automatically after
installation, otherwise run `wsl` from PowerShell):

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv pipx
pipx install poetry
```

- `sudo apt update` refreshes the list of available packages.
- `python3 python3-pip python3-venv`: Python and its base tooling.
- `pipx`: installs Python command-line tools in isolated environments
  (recommended over `pip install --user`).
- `pipx install poetry`: installs Poetry, the dependency manager used
  by this project.

------------------------------------------------------------------------

## 🪜 Step 3 — Reload the terminal

After `pipx install poetry`, the folder where Poetry was installed
isn't necessarily on the `PATH` of your current session yet.

**Close and reopen your terminal** (or run `source ~/.bashrc`) so the
`poetry` command is recognized. You can verify it with:

```bash
poetry --version
```

(See also the [Poetry command not recognized](FAQ.md) section if the
issue persists.)

------------------------------------------------------------------------

## 🪜 Step 4 — Enter WSL and install the project

If you're not already inside Ubuntu, run:

```bash
wsl
```

Then move to the project folder (your Windows files are accessible
from WSL under `/mnt/c/...`), and run:

```bash
poetry install
```

This installs all the project's Python dependencies (Panel,
HoloViews, h5py, etc.) — **except JAX**, which is installed separately
depending on your hardware.

------------------------------------------------------------------------

## 🪜 Step 5 — Install JAX with GPU (CUDA) support

```bash
poetry run pip install -U "jax[cuda13]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

This installs the JAX build compiled for **CUDA 13**, with GPU access.
If your card/driver matches CUDA 12 instead, use `jax[cuda12]` instead
(see the [README](README.md) for version details).

------------------------------------------------------------------------

## 🪜 Step 6 — Launch the application

```bash
poetry run panel serve thz_analyzer/main_app.py --show --args --no-preallocate
```

- `--args --no-preallocate`: `--args` forwards everything after it to the
  app itself, where `--no-preallocate` prevents JAX from immediately
  pre-allocating almost all available GPU memory (useful when you want to
  leave GPU memory free for something else, or to avoid out-of-memory
  errors at startup). `--args` must be the last option on the line.
- `--show`: automatically opens the application in your browser.

Not hitting memory issues? Just drop `--args --no-preallocate`:

```bash
poetry run panel serve thz_analyzer/main_app.py --show
```

Once the app is running, in the interface select **GPU** as the device
before clicking **"Optimize (JAX)"** to benefit from acceleration.

------------------------------------------------------------------------

## ✅ Quick summary

```bash
# In PowerShell (admin)
wsl --update
wsl --install -d Ubuntu-24.04
# restart if prompted

# In Ubuntu (WSL)
sudo apt update
sudo apt install -y python3 python3-pip python3-venv pipx
pipx install poetry
# close/reopen the terminal

wsl
cd /mnt/c/path/to/correct-tds2
poetry install
poetry run pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run panel serve thz_analyzer/main_app.py --show --args --no-preallocate
```

------------------------------------------------------------------------

Running into an error along the way (WSL install hangs, GPU not
detected, port not reachable)? Check the
[GPU / WSL issues](FAQ.md#gpu--wsl-issues) section of the FAQ.

------------------------------------------------------------------------

[← Back to documentation index](README.md#-documentation)

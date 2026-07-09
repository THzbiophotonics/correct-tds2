# ❓ FAQ — Troubleshooting

This page covers common errors and how to fix them. For installation
and feature guides, see the [documentation index](README.md#-documentation).

------------------------------------------------------------------------

## Poetry command not recognized

If you encounter errors such as:

-   `'poetry' is not recognized as an internal or external command`
-   `command not found: poetry`

This usually means that Poetry's installation directory is not included
in your system's PATH.

### 🪟 Windows: Add Poetry to PATH

Use the following command in **PowerShell** or **Command Prompt**:

``` powershell
setx PATH "%PATH%;C:\\Users\\<your-username>\\AppData\\Roaming\\Python\\Scripts"
```

After running the command, **close and reopen your terminal** so the
changes take effect.

### 🐧 Linux / WSL: Add Poetry to PATH

To add Poetry to PATH temporarily (only for the current terminal
session):

``` bash
export PATH="/home/<your-username>/.local/bin:$PATH"
```

To apply this permanently, add the same line to one of your shell
configuration files:

-   `~/.bashrc`
-   `~/.zshrc`
-   `~/.profile`

Then reload your shell or restart your terminal.

### 🧪 How to verify that Poetry is correctly installed

Run:

``` bash
poetry --version
```

If it prints a version number, Poetry is available system-wide.

------------------------------------------------------------------------

## GPU / WSL issues

These issues relate to the [GPU on Windows via WSL](INSTALL_GPU_WSL.md)
setup.

### `wsl --install` fails or hangs

Check that virtualization is enabled in the BIOS, and that the
"Virtual Machine Platform" and "Windows Subsystem for Linux" Windows
features are enabled
(`Control Panel → Programs → Turn Windows features on or off`).

### JAX doesn't detect the GPU inside WSL

- Make sure the **Windows** NVIDIA driver (not a Linux one) is
  up to date: <https://www.nvidia.com/Download/index.aspx>.
- Check that the card is visible inside WSL: `nvidia-smi` should show
  your GPU.
- Make sure you actually ran `wsl --update` before installing Ubuntu:
  older WSL versions didn't expose the GPU correctly.

### `Permission denied: 'python'` (or `python: command not found`) inside WSL

Some Ubuntu/WSL installs only ship a `python3` binary, no `python` alias,
which breaks commands relying on a plain `python`. Fix it with:

``` bash
sudo apt install python-is-python3
```

### The app's port isn't reachable from the Windows browser

Normally WSL2 forwards `localhost` to Windows automatically, and
`--show` should open the browser on its own. If it doesn't, copy the
URL shown in the terminal (e.g. `http://localhost:5006/...`) and paste
it manually into your Windows browser.

------------------------------------------------------------------------

[← Back to documentation index](README.md#-documentation)

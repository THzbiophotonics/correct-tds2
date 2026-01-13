# ❓ FAQ --- Poetry PATH Issues

## **Poetry command not recognized**

If you encounter errors such as:

-   `'poetry' is not recognized as an internal or external command`
-   `command not found: poetry`

This usually means that Poetry's installation directory is not included
in your system's PATH.

------------------------------------------------------------------------

## 🪟 **Windows: Add Poetry to PATH**

Use the following command in **PowerShell** or **Command Prompt**:

``` powershell
setx PATH "%PATH%;C:\\Users\\<your-username>\\AppData\\Roaming\\Python\\Scripts"
```

After running the command, **close and reopen your terminal** so the
changes take effect.

------------------------------------------------------------------------

## 🐧 **Linux: Add Poetry to PATH**

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

------------------------------------------------------------------------

## 🧪 **How to verify that Poetry is correctly installed**

Run:

``` bash
poetry --version
```

If it prints a version number, Poetry is available system-wide.
# 📦 Batch Mode — How to Process Multiple Files at Once

Batch mode lets you run the full correction pipeline (preview → optimization → export) on a series of `.h5` files without manual intervention. Each sample file is processed sequentially with its paired reference, and results are saved automatically.

---

## Step 1 — Open the Batch tab

In the app, click the **"Batch mode"** tab (last tab in the top navigation bar). You will see:

- A file browser to select your `.h5` files
- A status line showing how many files are selected
- An **"Auto-link sample/ref"** button
- An editable mapping text area
- A **"Run batch (sequential)"** button

---

## Step 2 — Select your files

Use the file browser to navigate to the folder containing your `.h5` files. **Select all the files you want to include** (both sample and reference files). You can select files from a single folder; cross-folder batches require editing the mapping manually (see Step 4).

> If you are on Windows and your files are on a different drive (e.g. `D:`), use the drive selector at the top of the app to switch drives before opening the file browser.

---

## Step 3 — Auto-link sample/reference pairs

Click **"Auto-link sample/ref"**. The app will automatically detect which files are samples and which are references based on their filenames, then pair them.

**How the auto-detection works:**

- A file is treated as a **reference** if its name contains `reference`, `ref`, or similar variations (e.g. `ref`, `ref1`, `reference_air`).
- A file is treated as a **sample** if its name contains `sample`, `samp`, or `sam` (e.g. `sample_GaAs`, `samp01`).
- Files matching neither pattern are treated as **samples**.

**How the pairing works:**

- The app strips the sample/reference tokens from each filename and compares the remaining parts.
- A sample is paired with the reference whose stripped name is most similar (similarity threshold: 82 %).
- Example: `sample_GaAs_run1.h5` → stripped key `GaAs_run1` will match `ref_GaAs_run1.h5` → stripped key `GaAs_run1`.

The mapping is then written into the text area for you to review.

---

## Step 4 — Review and edit the mapping

The mapping text area shows one job per line:

```
# One job per line: sample_path ==> reference_path(optional)
C:/data/sample_GaAs.h5 ==> C:/data/ref_GaAs.h5
C:/data/sample_InP.h5 ==> C:/data/ref_InP.h5
C:/data/sample_bare.h5 ==>
# Reference-only lines (not executed, editable)
==> C:/data/ref_extra.h5
```

**Editing rules:**

| Syntax | Meaning |
|--------|---------|
| `sample.h5 ==> ref.h5` | Process sample with the given reference |
| `sample.h5 ==>` | Process sample with no reference (no combined NCM) |
| `# comment` | Ignored line |
| `==> ref.h5` | Reference-only entry — kept for reference, not executed |

You can:
- **Remove a line** to skip that sample entirely.
- **Change the reference** on any line by editing the path after `==>`.
- **Add a line manually** with a full absolute path if the file was not in the original selection.

---

## Step 5 — Configure settings before running

Before clicking **"Run batch"**, make sure the settings in the other tabs are configured the way you want — they apply to all jobs:

- **Filters tab**: frequency and time domain filtering parameters.
- **Optimization tab**: correction type (delay / amplitude / dilation), optimizer, number of steps, bounds.
- **Export tab**: output directory and file prefix. Each job saves its results to the configured output directory. If you use a file prefix, it is applied to every exported file.

> **Tip:** Run a single file first in single-file mode to validate your settings, then switch to batch mode.

---

## Step 6 — Run the batch

Click **"Run batch (sequential)"**. The app processes each job one at a time:

1. Load the sample (and reference if provided).
2. Run the preview / periodic correction if enabled.
3. Run the optimization.
4. Export results to the output directory.

The status bar updates after each file: `Running 2/8: sample=sample_InP.h5, reference=ref_InP.h5`.

When finished, a summary is shown:

```
Batch finished: 7/8 success.
Failures:
- sample_broken.h5: ValueError (time axis has zero spacing)
```

Successful jobs produce the same output files as single-file mode (corrected traces `.h5`, parameters `.txt`, covariance matrices, etc.), each prefixed with the configured file prefix.

---

[← Back to documentation index](README.md#-documentation)

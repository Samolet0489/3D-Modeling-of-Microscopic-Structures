
# Commands to Run Bachelor Thesis Code

Complete step-by-step guide for Phase 1 (2.5D pipeline).

## Setup (one-time only)

Open **PowerShell** and run from the project root:

```powershell
cd C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code

# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# Install dependencies
pip install numpy scipy matplotlib imageio
```

After setup, you should see `(.venv)` prefix in your PowerShell prompt.

---

## Every Session

Activate the virtual environment:
```powershell
cd .\Thesis_code\
.\.venv\Scripts\Activate.ps1
```

---

## Run Demo  (sanity check)

Shows a 4-panel figure: volume slice, mask, clean image, noisy image.

Note: the current default noise behavior is supervisor-aligned Poisson noise with randomized intensity scaling. Gaussian noise is not used by default.

```powershell
python -m src.demo_pipeline
```

**Expected output:** matplotlib window pops up → close it to return to terminal.

---

## Generate Dataset

Create 500 synthetic image/mask pairs for training. Images and masks are saved as 8-bit PNG files (pixel-aligned).

```powershell
python -m src.dataset_generator .\data\synthetic_2d --n 500
```

**Output folder structure:**
```
data/synthetic_2d/
├── images/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── masks/
    ├── 000000.png
    ├── 000001.png
    └── ...
```

Each `images/XXXXXX.png` pairs with `masks/XXXXXX.png` — ready for Phase 2 (training a segmentation model).

---

## Check Dataset

Random visualization to verify generated images and masks look reasonable.

```powershell
python -m src.check_dataset
```

Shows a grid of 6 random samples with mask contours (red) overlaid on images (grayscale).

**Options:**
- `--data-dir`: path to dataset folder (default: `./data/synthetic_2d`)
- `--n`: number of samples to display (default: 6)

Example: show 12 samples from a different folder
```powershell
python -m src.check_dataset --data-dir .\data\my_dataset --n 12
```

**Graceful failure:** If folder or PNG files are missing, the script prints diagnostic messages and exits cleanly.

---

## Common Workflow

From the project root (`Thesis_code`), in a single session:

```powershell
# Activate venv (if not already active)
.venv\Scripts\Activate.ps1

# Check that the pipeline works
python -m src.demo_pipeline

# Generate 100 training samples (quick test)
python -m src.dataset_generator .\data\synthetic_2d --n 100

# Visualize what was generated
python -m src.check_dataset

# Generate larger dataset for actual training
python -m src.dataset_generator .\data\synthetic_2d --n 5000
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `cannot be loaded because running scripts is disabled` | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError: No module named 'imageio'` | Run: `pip install imageio` |
| `No module named 'src'` | Ensure you are IN `Thesis_code` folder. Check: `pwd` shows `.../Thesis_code` |
| `Data directory not found` when running check_dataset | Run `python -m src.dataset_generator .\data\synthetic_2d --n 10` first |

---

## Code Structure

```
Thesis_code/
└── src/
    ├── __init__.py          # Package initialization
    ├── shapes.py            # Volume & mask generation (mitochondria tube model)
    ├── psf.py               # 3D Gaussian PSF kernel
    ├── render.py            # PSF convolution & 2D projection/slicing
    ├── noise.py             # Supervisor-aligned microscopy noise (Poisson-only by default, Gaussian optional but off)
    ├── dataset_generator.py # Batch image/mask generation + CLI
    ├── demo_pipeline.py     # Single-sample demo (4-panel figure)
    └── check_dataset.py     # Dataset visualization tool (NEW)
```

All scripts are runnable via `python -m src.<module_name>` from the `Thesis_code` folder.
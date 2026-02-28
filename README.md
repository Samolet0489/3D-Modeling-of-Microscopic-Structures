# 3D-Modeling-of-Microscopic-Structures

**Bachelor thesis project:**
*"3D Modeling of Microscopic Structures for Synthetic Data Generation Applications in AI Model Training for Microscopy Data"*

The repository contains a small Python package under `Thesis_code/src` that
implements a phase‑1 pipeline for generating 2.5‑D synthetic microscopy data
(from internally‑computed 3D volumes).

---


## 📦 Project layout
```
Thesis_code/
└── src/               # Python package
    ├── __init__.py
    ├── shapes.py            # 3D volume & mask generation
    ├── psf.py               # 3D Gaussian PSF kernel
    ├── render.py            # Projection / slicing routines
    ├── noise.py             # Poisson + Gaussian noise models
    ├── dataset_generator.py # CLI for creating synthetic datasets
    ├── demo_pipeline.py     # Single‑sample sanity demo (4‑panel figure)
    └── check_dataset.py     # Tool to visualize image/mask pairs
```

The `data/` directory (created when running the generator) holds the
resulting PNG files.

---

## ⚙️ Setup & environment

Run once to initialise:
```powershell
cd C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
python -m venv .venv                          # create venv
.venv\Scripts\Activate.ps1                  # activate it
pip install numpy scipy matplotlib imageio   # install requirements
```

Every time you start work:
```powershell
cd C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run `Activate.ps1`, you may need to
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

---

## ▶️ Common commands

| Task | Command |
|------|---------|
| Run demo | `python -m src.demo_pipeline` |
| Generate dataset | `python -m src.dataset_generator <out_dir> --n <N> [--seed <S>]` |
| Visualise dataset | `python -m src.check_dataset --data-dir <out_dir> --n <M>` |
| Full quick test | see below |

### Example workflows
```powershell
# demo + small dataset + check
python -m src.demo_pipeline
python -m src.dataset_generator .\data\synthetic_2d --n 50 --seed 123
python -m src.check_dataset --data-dir .\data\synthetic_2d --n 12

# create large training set
python -m src.dataset_generator .\data\synthetic_2d --n 2000 --seed 42
```

The dataset generator randomises parameters per-sample (tubes, PSF,
noise levels). Providing a `--seed` makes results reproducible.

---

## ✅ Output structure

```
data/synthetic_2d/
├── images/       # noisy 2D images (uint8 PNG)
│   ├── 000000.png
│   └── ...
└── masks/        # corresponding 2D masks (uint8 PNG)
    ├── 000000.png
    └── ...
```

File names are zero-padded and aligned pixel-for-pixel.

---

## 📝 Notes for future phases
* Phase 2 will add `dataset2d.py`, `unet2d.py`, and `train_2d.py` for
  2D segmentation experiments (real-only, synthetic-only, hybrid).
* Phase 3 will extend generation to full 3D volumes and implement a 3D U-Net.

The code is intentionally simple and well-documented to make translating the
implementation into thesis text straightforward.

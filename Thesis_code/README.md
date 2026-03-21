# Thesis: 3D Modeling of Microscopic Structures (Synthetic Microscopy)

This repository implements a **3D-first synthetic microscopy pipeline** and a **2D U-Net** training baseline.

**Pipeline (Phase 1):**
1. Generate 3D structure mask + intensity volume (Model A / Model B)
2. Simulate microscopy imaging (3D PSF blur + 2D projection)
3. Add noise (Poisson + Gaussian)
4. Save paired 2D images + masks as PNG

**Segmentation (Phase 2):**
- Train a 2D U-Net (PyTorch) on the generated PNG pairs
- Evaluate Dice / IoU and save qualitative prediction snapshots

---

## Folder structure

```
Thesis_code/
  data/
    <dataset_name>/
      images/        # PNG grayscale
      masks/         # PNG binary (0/255)
  runs/
    <run_name>/      # checkpoints + prediction snapshots
  src/
    __init__.py
    shapes.py
    psf.py
    render.py
    noise.py
    dataset_generator.py
    check_dataset.py
    viz_3d.py
    dataset2d.py
    unet2d.py
    metrics.py
    train_2d.py
```

---

## Setup (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
```

Install dependencies:

```powershell
python -m pip install numpy scipy matplotlib imageio
```

Install PyTorch (CUDA build, recommended for RTX 30xx):

```powershell
python -m pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Phase 1: Generate and inspect synthetic data

### Generate a dataset
Example: Model B with max projection

```powershell
python -m src.dataset_generator .\data\synthetic_B --n 500 --model B --projection max --seed 1
```

### Visual sanity check (mask overlay)
```powershell
python -m src.check_dataset --data-dir .\data\synthetic_B --n 12
```

### Visualize 3D mask (projections + 3D scatter)
```powershell
python -m src.viz_3d --model B --seed 42
```

---

## Phase 2: Train 2D U-Net

### Overfit sanity test (debug correctness)
```powershell
python -m src.train_2d --data .\data\synthetic_B --epochs 30 --batch-size 4 --limit-train 16 --device cuda --num-workers 0 --out-dir .\runs\overfit_cuda
```

### Full training run
```powershell
python -m src.train_2d --data .\data\synthetic_B --epochs 20 --batch-size 8 --device cuda --num-workers 2 --out-dir .\runs\synthetic_B_run1
```

Outputs written to the run folder:
- `best.pt`, `last.pt` (checkpoints)
- `preds_epoch_XXX.png` (training snapshots)
- `preds_test.png` (test evaluation snapshot)

---

## Notes

- **Model A** is a simple baseline morphology (easier segmentation).
- **Model B** is intended to become more mitochondria-like (branching / network-like).
- If you change the generator realism, generate a new dataset folder (e.g., `synthetic_B_v2`) to keep experiments comparable.

---

## Roadmap

- Phase 2.5/3: Improve Model B realism (persistent random walk, branching networks, thickness variation)
- Re-train and compare robustness (e.g., noise sweeps)
- Optional: Hybrid training with real microscopy data (synthetic pretrain → real fine-tune)

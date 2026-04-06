# Commands to Run Thesis Code (Updated Fine-Tuned Version)

Run all commands from:

```powershell
C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
```

Activate environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 1. Quick sanity check of the current fine-tuned pipeline

This shows:
- raw mask
- visible mask
- clean image
- noisy image

```powershell
python -m src.demo_pipeline --model B --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 12 --background-strength 0.08 --band-strength 0.025 --gradient-strength 0.03 --seed 42
```

---

## 2. Simpler professor-like profile (Option A)

Use this when you want sparser, cleaner structures closer to the professor sample style.

### Single demo

```powershell
python -m src.demo_pipeline --model B --projection focus_max --slab-thickness 5 --mask-low-thr 0.14 --mask-high-thr 0.28 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 2 --apply-gaussian-noise --gaussian-std 11 --background-strength 0.06 --band-strength 0.015 --gradient-strength 0.02 --seed 42
```

### Generate simple-profile dataset

```powershell
python -m src.dataset_generator data/datasets/20260406_modelB_simple_profile --model B --n 300 --projection focus_max --slab-thickness 5 --mask-low-thr 0.14 --mask-high-thr 0.28 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 2 --apply-gaussian-noise --gaussian-std 11 --background-strength 0.06 --band-strength 0.015 --gradient-strength 0.02 --n-seeds 3 --step-length 1.4 --n-steps 95 --branch-prob 0.05 --branch-steps 28 --radius-min 1 --radius-max 3 --persistence 0.88 --z-scale 0.36 --seed-spread 20 --seed 42
```

---

## 3. Best balanced fine-tuned profile (current main candidate)

Use this as the current main Model B candidate.

### Single demo

```powershell
python -m src.demo_pipeline --model B --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 12 --background-strength 0.08 --band-strength 0.025 --gradient-strength 0.03 --seed 42
```

### Generate balanced dataset

```powershell
python -m src.dataset_generator data/datasets/20260406_modelB_balanced_final --model B --n 1000 --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 12 --background-strength 0.08 --band-strength 0.025 --gradient-strength 0.03 --n-seeds 4 --step-length 1.5 --n-steps 120 --branch-prob 0.08 --branch-steps 40 --radius-min 1 --radius-max 3 --persistence 0.85 --z-scale 0.35 --seed-spread 20 --randomize-per-sample --seed 42
```

---

## 4. Stronger-noise variant

Use this only if you want to test a harsher background/noise setting.

```powershell
python -m src.dataset_generator data/datasets/20260406_modelB_stronger_noise --model B --n 300 --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 14 --background-strength 0.10 --band-strength 0.035 --gradient-strength 0.04 --n-seeds 3 --step-length 1.4 --n-steps 95 --branch-prob 0.05 --branch-steps 28 --radius-min 1 --radius-max 3 --persistence 0.88 --z-scale 0.36 --seed-spread 20 --randomize-per-sample --seed 42
```

---

## 5. Inspect generated dataset

```powershell
python -m src.check_dataset --data-dir data/datasets/20260406_modelB_balanced_final --n 6
```

Example for the simple profile:

```powershell
python -m src.check_dataset --data-dir data/datasets/20260406_modelB_simple_profile --n 6
```

---

## 6. Smoke-train on a small dataset

```powershell
python -m src.train_2d --data data/datasets/20260406_modelB_simple_profile --epochs 3 --batch-size 8 --num-workers 0 --limit-train 16 --out-dir results/20260406_train_smoke --device cuda
```

---

## 7. Train on the balanced final dataset

```powershell
python -m src.train_2d --data data/datasets/20260406_modelB_balanced_final --epochs 25 --batch-size 8 --num-workers 0 --out-dir results/20260406_train_modelB_balanced_final --device cuda
```

---

## 8. Model A baseline command

Model A is still available and should be kept as the thesis baseline.

### Demo

```powershell
python -m src.demo_pipeline --model A --projection center_slice --seed 42
```

### Dataset

```powershell
python -m src.dataset_generator data/datasets/20260406_modelA_baseline --model A --n 500 --projection center_slice --seed 42
```

---

## 9. Recommended working order

```powershell
# 1) simple professor-like subset
python -m src.dataset_generator data/datasets/20260406_modelB_simple_profile --model B --n 300 --projection focus_max --slab-thickness 5 --mask-low-thr 0.14 --mask-high-thr 0.28 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 2 --apply-gaussian-noise --gaussian-std 11 --background-strength 0.06 --band-strength 0.015 --gradient-strength 0.02 --n-seeds 3 --step-length 1.4 --n-steps 95 --branch-prob 0.05 --branch-steps 28 --radius-min 1 --radius-max 3 --persistence 0.88 --z-scale 0.36 --seed-spread 20 --seed 42

# 2) inspect
python -m src.check_dataset --data-dir data/datasets/20260406_modelB_simple_profile --n 6

# 3) main balanced dataset
python -m src.dataset_generator data/datasets/20260406_modelB_balanced_final --model B --n 1000 --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 12 --background-strength 0.08 --band-strength 0.025 --gradient-strength 0.03 --n-seeds 4 --step-length 1.5 --n-steps 120 --branch-prob 0.08 --branch-steps 40 --radius-min 1 --radius-max 3 --persistence 0.85 --z-scale 0.35 --seed-spread 20 --randomize-per-sample --seed 42

# 4) inspect
python -m src.check_dataset --data-dir data/datasets/20260406_modelB_balanced_final --n 6

# 5) smoke train
python -m src.train_2d --data data/datasets/20260406_modelB_simple_profile --epochs 3 --batch-size 8 --num-workers 0 --limit-train 16 --out-dir results/20260406_train_smoke --device cuda
```

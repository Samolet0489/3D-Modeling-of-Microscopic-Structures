# Fine-Tuned Final Commands

This file stores the current best commands so they do not need to be reconstructed again.

Run everything from:

```powershell
C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
.\.venv\Scripts\Activate.ps1
```

---

## Current best simple professor-like profile

```powershell
python -m src.dataset_generator data/datasets/20260406_modelB_simple_profile --model B --n 300 --projection focus_max --slab-thickness 5 --mask-low-thr 0.14 --mask-high-thr 0.28 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 2 --apply-gaussian-noise --gaussian-std 11 --background-strength 0.06 --band-strength 0.015 --gradient-strength 0.02 --n-seeds 3 --step-length 1.4 --n-steps 95 --branch-prob 0.05 --branch-steps 28 --radius-min 1 --radius-max 3 --persistence 0.88 --z-scale 0.36 --seed-spread 20 --seed 42
```

Use to inspect:

```powershell
python -m src.check_dataset --data-dir data/datasets/20260406_modelB_simple_profile --n 6
```

---

## Current best balanced final profile

```powershell
python -m src.dataset_generator data/datasets/20260406_modelB_balanced_final --model B --n 1000 --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --mask-min-size 10 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 12 --background-strength 0.08 --band-strength 0.025 --gradient-strength 0.03 --n-seeds 4 --step-length 1.5 --n-steps 120 --branch-prob 0.08 --branch-steps 40 --radius-min 1 --radius-max 3 --persistence 0.85 --z-scale 0.35 --seed-spread 20 --randomize-per-sample --seed 42
```

Use to inspect:

```powershell
python -m src.check_dataset --data-dir data/datasets/20260406_modelB_balanced_final --n 6
```

---

## Current best demo command

```powershell
python -m src.demo_pipeline --model B --projection focus_max --slab-thickness 5 --mask-low-thr 0.13 --mask-high-thr 0.27 --seed-erode-iters 1 --mask-gamma 1.35 --max-hole-size 3 --apply-gaussian-noise --gaussian-std 12 --background-strength 0.08 --band-strength 0.025 --gradient-strength 0.03 --seed 42
```

---

## Current smoke-train command

```powershell
python -m src.train_2d --data data/datasets/20260406_modelB_simple_profile --epochs 3 --batch-size 8 --num-workers 0 --limit-train 16 --out-dir results/20260406_train_smoke --device cuda
```

---

## Notes

- Keep Model A as the baseline reference.
- Use the simple profile when you want samples closer to the professor's sparse style.
- Use the balanced final profile as the current main candidate for thesis experiments.

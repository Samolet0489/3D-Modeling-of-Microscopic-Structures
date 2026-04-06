# Thesis Project Status

## Current stage
Phase 2 baseline is complete.
Model B mask and noise fine-tuning is now in the final polishing stage.

We now have:
- a working synthetic-data generation pipeline
- a working 2D segmentation training pipeline
- a cleaned project structure
- a retained Model A baseline generator
- an improved Model B generator with border-aware generation and dataset QC
- a refined mask-generation path based on visible rendered signal constrained by raw support
- a refined microscopy-noise path with stronger background realism
- current fine-tuned command sets for both a simple profile and a balanced final profile

---

## What is already done

### Phase 1 / synthetic pipeline
- 3D shape generation
- Model A baseline generator (still present and usable)
- Model B polyline / branched generator
- PSF blur
- 3D -> 2D rendering
- focus-limited projection option
- refined visible-mask extraction
- microscopy noise with structured background realism
- PNG dataset generation
- dataset sanity-check visualization

### Phase 2 / training baseline
- PyTorch dataset loader for images + masks
- deterministic train/val/test split
- paired augmentations
- 2D U-Net
- Dice + IoU metrics
- training script with:
  - positive-class weighting for class imbalance
  - overfit sanity mode
  - checkpoints
  - prediction snapshots
  - test evaluation

---

## Current code interpretation

### Model A
Model A is still in the project and should be kept.
It is not dead code.
It remains useful for:
- baseline comparison in the thesis
- showing progression from simple to improved morphology
- generating simpler controlled samples if needed

### Model B
Model B remains the main active generator.
It is currently the best candidate for the thesis experiments.
The recent work focused on:
- better visual mask agreement
- better background/noise realism
- simpler and balanced profile selection through command-line settings

---

## Current fine-tuned profiles

### 1. Simple professor-like profile
Use when you want sparser, simpler shapes closer to the style shown by the professor.

Selected config:
- model = B
- projection = focus_max
- slab_thickness = 5
- mask_low_thr = 0.14
- mask_high_thr = 0.28
- mask_min_size = 10
- seed_erode_iters = 1
- mask_gamma = 1.35
- max_hole_size = 2
- apply_gaussian_noise = true
- gaussian_std = 11
- background_strength = 0.06
- band_strength = 0.015
- gradient_strength = 0.02
- n_seeds = 3
- step_length = 1.4
- n_steps = 95
- branch_prob = 0.05
- branch_steps = 28
- radius_min = 1
- radius_max = 3
- persistence = 0.88
- z_scale = 0.36
- seed_spread = 20

### 2. Balanced final profile
Use as the current main thesis candidate.

Selected config:
- model = B
- projection = focus_max
- slab_thickness = 5
- mask_low_thr = 0.13
- mask_high_thr = 0.27
- mask_min_size = 10
- seed_erode_iters = 1
- mask_gamma = 1.35
- max_hole_size = 3
- apply_gaussian_noise = true
- gaussian_std = 12
- background_strength = 0.08
- band_strength = 0.025
- gradient_strength = 0.03
- n_seeds = 4
- step_length = 1.5
- n_steps = 120
- branch_prob = 0.08
- branch_steps = 40
- radius_min = 1
- radius_max = 3
- persistence = 0.85
- z_scale = 0.35
- seed_spread = 20
- randomize_per_sample = true

---

## Current interpretation of quality

### Mask status
The mask problem is no longer a blocking issue.
The current refined masks are now:
- structurally believable
- visibly aligned with the rendered signal
- stable on both sparse and denser samples

Only small imperfections remain in thicker regions.
Those are now considered polishing issues, not pipeline failures.

### Noise status
Noise realism was previously the weakest part.
After the latest refinement pass, the background now looks much more microscopy-like.
The next task is only final parameter calibration, not a redesign.

---

## Immediate next step
1. Generate and inspect the simple professor-like dataset
2. Generate and inspect the balanced final dataset
3. Run smoke training on the simple dataset
4. Lock the final parameter set
5. Start thesis writing immediately after the final parameter lock

---

## What is next after final parameter lock
- write the Methods section for synthetic data generation
- document Model A vs Model B as baseline vs improved approach
- document the final mask and noise refinement decisions
- run the main training experiments
- prepare thesis figures from the final datasets

---

## Notes
- Keep Model A in the project as a baseline and thesis comparison point.
- Do not remove it unless there is a very strong reason.
- Do not do further large rewrites of the mask pipeline.
- Only small parameter-level polishing should happen from here.

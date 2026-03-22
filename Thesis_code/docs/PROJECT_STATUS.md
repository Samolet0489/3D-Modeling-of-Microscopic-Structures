# Thesis Project Status

## Current stage
Phase 2 baseline is complete, and Model B realism improvement is in progress.

We now have:
- a working synthetic-data generation pipeline
- a working 2D segmentation training pipeline
- a cleaned project structure
- a first improved Model B candidate with border-aware generation and dataset QC

---

## What is already done

### Phase 1 / synthetic pipeline
- 3D shape generation
- Model A baseline generator
- Model B polyline / branched generator
- PSF blur
- 3D -> 2D rendering
- microscopy noise
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

## Baseline results before realism improvement

### Synthetic_A
- best val Dice ≈ 0.878
- test Dice ≈ 0.915
- test IoU ≈ 0.849

### Synthetic_B (older baseline)
- best val Dice ≈ 0.735
- test Dice ≈ 0.725
- test IoU ≈ 0.575

### Interpretation
- the pipeline works end-to-end
- the model can learn from synthetic data
- Model A currently outperforms the older Model B
- the main bottleneck is synthetic realism, especially Model B morphology

---

## 2026-03-22 - Model B realism improvement progress

A revised Model B with smoother tubules, border-aware walker behavior, and dataset QC was introduced.

### Generator-side improvements
- smoother segment rasterization
- smoother radius drift along branches
- flatter colonies around the focal plane
- border repulsion before walkers reach image walls
- inward turning when walkers approach boundaries
- walker termination after repeated edge collisions
- automatic sample QC based on foreground fraction and border dominance

### Failed dense setting (`qc1`)
An initial dense revision of Model B failed the QC gate because the center-slice masks were too full.

Observed issue:
- foreground fraction often around 0.49-0.69
- rejection reason: `too_full`

Interpretation:
- border artifacts were reduced
- but morphology density was too high for the selected center-slice projection setup

### Selected improved Model B candidate (`qc1b`)
A lighter parameter set produced acceptable and visually improved samples.

Selected config:
- n_seeds = 3
- step_length = 1.4
- n_steps = 90
- branch_prob = 0.04
- branch_steps = 30
- radius_min = 1
- radius_max = 3
- persistence = 0.88
- z_scale = 0.36
- seed_spread = 20

Observed QC behavior:
- foreground fraction approximately 0.08-0.22
- all 24/24 smoke-test samples accepted on the first attempt
- border fraction effectively 0.00 in the smoke test

### Smoke-test training result on `qc1b`
Quick overfit-style training on the `qc1b` smoke dataset produced:
- Best validation Dice ≈ 0.871
- Test Dice ≈ 0.871
- Test IoU ≈ 0.772

Interpretation:
This is currently the first good controlled “improved Model B” candidate.
The next step is to generate a larger dataset with these fixed parameters and run a proper synthetic-only training experiment for comparison against:
- the older Model B baseline
- Model A baseline

---

## What is next

### Immediate next step
1. Generate a larger dataset using the selected improved Model B config
2. Visually inspect the dataset
3. Run a proper synthetic-only training experiment
4. Compare against older Model B and Model A

### After that
5. Decide whether another realism iteration is needed
6. Write the realism-improvement subsection for the thesis
7. Continue later with:
   - real-only experiments
   - hybrid synthetic + real experiments
   - robustness / domain shift tests

---

## Selected current Model B config

Use this as the current improved Model B candidate:

- n_seeds = 3
- step_length = 1.4
- n_steps = 90
- branch_prob = 0.04
- branch_steps = 30
- radius_min = 1
- radius_max = 3
- persistence = 0.88
- z_scale = 0.36
- seed_spread = 20

Note:
This config should be passed via command line during dataset generation and recorded in each dataset folder’s `run_config.json`. It should not be hardcoded into the generator.

---

## Useful commands

All commands should be run from:

`C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code`

Activate environment first:

```powershell
.\.venv\Scripts\Activate.ps1
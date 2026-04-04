# nextTime.md

# Thesis – Next Time Resume Notes

## Where the project stands right now

The cleaned project structure is working.

Active project root:
`C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code`

The important pipeline pieces are now working:
- synthetic 3D generator
- 3D -> 2D rendering
- dataset generation
- dataset QC
- dataset visualization check
- 2D U-Net training
- checkpoint saving
- prediction snapshot saving

## What was improved recently

### 1. Project structure cleanup
The project was reorganized so that:
- code lives in `src/`
- tests live in `tests/`
- datasets live in `data/datasets/`
- outputs live in `results/`
- notes/status live in `docs/`

### 2. `dataset_generator.py` improvement
The generator now respects fixed command-line parameters instead of silently randomizing them per sample.
It also writes:
- `run_config.json`
- `sample_manifest.jsonl`

### 3. `shapes.py` improvement
Model B was improved with:
- smoother segment rasterization
- smoother radius drift
- flatter colony layout around the focal plane
- border-aware motion
- inward turning near boundaries
- walker stopping after repeated edge collisions

### 4. QC filtering was added
The dataset generator now rejects bad samples if they are:
- too empty
- too full
- too border-heavy

This helped reveal that one early revised setting was too dense.


### 5. `noise.py` improvement
The microscopy noise model was updated based on supervisor feedback.

What changed:
- Gaussian noise was removed from the default path
- the noise function now uses:
  - random offset normalization
  - random max-signal scaling
  - Poisson noise only by default
- output compatibility with the rest of the pipeline was preserved by still returning normalized `float32` images in `[0, 1]`

Why this matters:
- the old idea risked over-degrading the generated images
- the new version keeps the images noisy and more microscope-like without destroying the structural signal needed for segmentation training

Validation:
- simple smoke tests worked
- pipeline-level preview with actual synthetic mitochondria also worked
- the noisy images show visible acquisition-like degradation while preserving morphology

---

## Important experiment history

### Older baseline results
- `synthetic_A`:
  - best val Dice ≈ 0.878
  - test Dice ≈ 0.915
  - test IoU ≈ 0.849

- older `synthetic_B`:
  - best val Dice ≈ 0.735
  - test Dice ≈ 0.725
  - test IoU ≈ 0.575

### Dense revised Model B (`qc1`)
This failed QC because center-slice masks were too full.
Foreground fraction was often about `0.49–0.69`, above the `max_fg_fraction = 0.45` limit.

Interpretation:
- border issue improved
- morphology density became too high

### Selected improved Model B candidate (`qc1b`)
This lighter setting worked well:

- `n_seeds = 3`
- `step_length = 1.4`
- `n_steps = 90`
- `branch_prob = 0.04`
- `branch_steps = 30`
- `radius_min = 1`
- `radius_max = 3`
- `persistence = 0.88`
- `z_scale = 0.36`
- `seed_spread = 20`

Smoke-test behavior:
- foreground fraction roughly `0.08–0.22`
- all smoke-test samples accepted on first attempt

Smoke-test training result:
- best validation Dice ≈ `0.871`
- test Dice ≈ `0.871`
- test IoU ≈ `0.772`

### Full improved Model B run (`20260322_synthetic_B_v2`)
A larger dataset was generated with the selected config and trained successfully.

Full training result:
- best validation Dice ≈ `0.917811`
- test Dice ≈ `0.924039`
- test IoU ≈ `0.858985`

This is a very important result because the improved Model B now performs at least as well as, and currently slightly better than, the earlier Model A baseline.

---

## Most important current conclusion

The improved Model B is now the first strong controlled realism-improved candidate.

At the moment the story is:

1. old Model B underperformed
2. border-aware / QC-aware redesign improved the generator
3. a dense revised setting failed QC for being too full
4. a lighter revised setting (`qc1b`) worked well
5. the larger improved Model B run (`B_v2`) achieved:
   - test Dice ≈ `0.924`
   - test IoU ≈ `0.859`
6. the updated Poisson-only noise model now works correctly inside the pipeline and is suitable for continued dataset generation and training experiments

That means the realism-improvement work is paying off.

---

## Exact selected config to keep using for now

Do **not** hardcode this into the generator.

Keep using it in commands and keep it documented in status files and dataset `run_config.json`.

Selected config:
- `n_seeds = 3`
- `step_length = 1.4`
- `n_steps = 90`
- `branch_prob = 0.04`
- `branch_steps = 30`
- `radius_min = 1`
- `radius_max = 3`
- `persistence = 0.88`
- `z_scale = 0.36`
- `seed_spread = 20`

---

## Known issue to remember

### AMP / GradScaler compatibility
A training run with `--amp` failed because the local PyTorch version does not accept:

```python
GradScaler(device_type="cuda")
```

Temporary workaround:
- run training **without** `--amp`

Permanent fix needed later in `src/train_2d.py`.

Recommended replacement logic:

```python
use_amp = (device.type == "cuda" and args.amp)

if use_amp:
    try:
        scaler = GradScaler("cuda")
    except TypeError:
        try:
            scaler = GradScaler()
        except TypeError:
            scaler = None
            use_amp = False
            print("Warning: AMP GradScaler not supported in this PyTorch version. Falling back to full precision.")
else:
    scaler = None
```

---

## Commands that are confirmed to work

Run from:

```powershell
cd C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
.\.venv\Scripts\Activate.ps1
```

### Environment checks
```powershell
python -c "import numpy, scipy, matplotlib, imageio, PIL; print('core libs ok')"
python -c "import torch, torchvision; print('torch ok')"
```

### Smoke tests
```powershell
python -m tests.test_polyline
python -m tests.test_model_b_realism
```

### Generate improved Model B smoke dataset
```powershell
python -m src.dataset_generator data/datasets/20260322_synthetic_B_qc1b --n 24 --model B --projection center_slice --seed 42 --n-seeds 3 --step-length 1.4 --n-steps 90 --branch-prob 0.04 --branch-steps 30 --radius-min 1 --radius-max 3 --persistence 0.88 --z-scale 0.36 --seed-spread 20
```

### Check smoke dataset
```powershell
python -m src.check_dataset --data-dir data/datasets/20260322_synthetic_B_qc1b --n 6
```

### Smoke training
```powershell
python -m src.train_2d --data data/datasets/20260322_synthetic_B_qc1b --limit-train 16 --epochs 10 --batch-size 4 --num-workers 0 --out-dir results/20260322_overfit_B_qc1b
```

### Generate larger improved Model B dataset
```powershell
python -m src.dataset_generator data/datasets/20260322_synthetic_B_v2 --n 800 --model B --projection center_slice --seed 42 --n-seeds 3 --step-length 1.4 --n-steps 90 --branch-prob 0.04 --branch-steps 30 --radius-min 1 --radius-max 3 --persistence 0.88 --z-scale 0.36 --seed-spread 20
```

### Check larger dataset
```powershell
python -m src.check_dataset --data-dir data/datasets/20260322_synthetic_B_v2 --n 6
```

### Full training run that worked
```powershell
python -m src.train_2d --data data/datasets/20260322_synthetic_B_v2 --epochs 25 --batch-size 8 --num-workers 0 --out-dir results/20260322_train_B_v2
```

---

## What to do next when coming back

### Immediate next step
1. open and inspect:
   - `results/20260322_train_B_v2/preds_test.png`
   - the snapshot images saved during training

2. update `docs/PROJECT_STATUS.md` with the final `B_v2` result:
   - best val Dice ≈ `0.917811`
   - test Dice ≈ `0.924039`
   - test IoU ≈ `0.858985`

3. decide whether to do one of these next:
   - **Option A:** write the thesis subsection for Model B realism improvement
   - **Option B:** rerun / confirm Model A in the same cleaned setup for an apples-to-apples comparison
   - **Option C:** patch AMP compatibility and minor cleanup warnings

### Recommended order
Best next order:
1. document the result
2. inspect prediction images
3. patch AMP compatibility
4. optionally clean `check_dataset.py` warning
5. then continue with comparison / thesis writing

---

## Thesis note text you can reuse later

During the refinement of Model B, the synthetic generator was modified to produce more controlled and realistic mitochondrial-like structures. Two major improvements were introduced. First, the polyline-based walker in the 3D generator was updated with smoother segment rasterization and border-avoidance behavior, reducing the earlier tendency of structures to stick to image boundaries. Second, the dataset generation pipeline was extended with a quality-control gate that rejects unsuitable samples based on foreground coverage and border dominance. These changes transformed the generator from a purely procedural prototype into a more controlled experimental tool suitable for ablation studies.

An initial revised configuration of Model B produced structures that were no longer dominated by border artifacts, but the resulting center-slice masks were too dense for the selected 2D projection setting. The QC mechanism correctly rejected these samples because the foreground fraction frequently exceeded the defined threshold. This intermediate result was still valuable, since it confirmed that the new border logic was working while also showing that morphology density must be controlled carefully when generating 2D supervision from 3D synthetic data.

A lighter Model B configuration with fewer seeds, fewer steps, reduced branching, and a smaller maximum radius produced more acceptable structures. In smoke-test generation, the resulting masks typically occupied around 8% to 22% of the image area and all samples passed the QC gate on the first attempt. Visual inspection showed that these structures were more centered, less border-hugging, and more plausibly tubule-like than earlier variants. This configuration became the first controlled improved Model B candidate.

A larger synthetic-only training experiment on the improved Model B dataset achieved a best validation Dice of approximately 0.918 and a test Dice of approximately 0.924, with a test IoU of about 0.859. These results suggest that improving synthetic morphology realism can significantly improve downstream segmentation performance, and in this iteration the revised Model B reached performance comparable to or slightly better than the earlier baseline Model A.

---

## Reminder to future me

Do not get lost in rewriting infrastructure again.
The main pipeline works.

The next useful work is:
- document results
- compare improved Model B against previous baselines
- write thesis material from the progress already made
- only then continue with optional cleanup or further realism tuning

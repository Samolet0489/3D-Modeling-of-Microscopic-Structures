# Model B Implementation - Random-Walk Polyline Tubules

## Overview

Added **Model B** (polyline/random-walk generator) to complement the existing **Model A** (baseline cylindrical tubes). This enables comparative experiments to validate synthetic data generation approaches for microscopy AI training.

## What's New

### 1. Model B Generator in `src/shapes.py`

**Function:** `generate_mito_volume_polyline()`

Creates curved, interconnected 3D tubules using random-walk centerlines with morphological dilation.

**Parameters:**
- `shape`: (Z, Y, X) volume dimensions (default: 32×128×128)
- `n_branches`: Number of random walks (default: 3)
- `step_length`: Step size in voxels (default: 1.0)
- `n_steps`: Steps per branch (default: 20)
- `tube_radius`: Thickness of resulting tubules (default: 2)
- `seed`: Random seed for reproducibility

**Algorithm:**
1. Generate N random-walk branches starting at random positions in 3D
2. Each step: move in random direction (uniform on unit sphere)
3. Clip to volume boundaries to prevent loss
4. Thicken centerline via morphological dilation with spherical kernel
5. Combine all branches into final volume and mask

**Example:**
```python
from src.shapes import generate_mito_volume_polyline

volume, mask = generate_mito_volume_polyline(
    shape=(32, 128, 128),
    n_branches=5,
    step_length=1.5,
    n_steps=25,
    tube_radius=2,
    seed=42
)
```

### 2. Updated `src/dataset_generator.py`

Added `--model` and `--projection` CLI arguments for flexible dataset generation.

**New CLI Flags:**
```bash
python -m src.dataset_generator <out_dir> \
    --n 100 \
    --model {A,B} \
    --projection {center_slice,max,sum} \
    --seed 42
```

**Model-Specific Parameter Randomization:**
- **Model A**: Randomizes `n_tubes` (3-15 per sample)
- **Model B**: Randomizes `n_branches` (2-8), `step_length` (0.5-2.0), `n_steps` (15-40)

**Projection Methods:**
- `center_slice`: Central Z-slice (original approach)
- `max`: Maximum intensity projection (reveals fuller tubule structure)
- `sum`: Sum projection with binary thresholding

**Example Workflows:**

Generate 500-sample Model A dataset with max projection:
```bash
python -m src.dataset_generator data/synthetic_A \
    --n 500 --model A --projection max --seed 42
```

Generate 500-sample Model B dataset with center-slice projection:
```bash
python -m src.dataset_generator data/synthetic_B \
    --n 500 --model B --projection center_slice --seed 42
```

### 3. Updated `src/demo_pipeline.py`

Added `--model` and `--projection` arguments for direct comparison.

**New CLI Flags:**
```bash
python -m src.demo_pipeline \
    --model {A,B} \
    --projection {center_slice,max,sum} \
    --seed 42
```

**Example Comparisons:**

View Model A with max projection:
```bash
python -m src.demo_pipeline --model A --projection max --seed 42
```

View Model B with center-slice:
```bash
python -m src.demo_pipeline --model B --projection center_slice --seed 42
```

### 4. Updated `src/__init__.py`

Exported `generate_mito_volume_polyline` for programmatic use:
```python
from src import generate_mito_volume_polyline
```

## File Structure Changes

```
src/
├── shapes.py              ← Added generate_mito_volume_polyline()
├── dataset_generator.py   ← Added --model and --projection args
├── demo_pipeline.py       ← Added --model and --projection args
└── __init__.py            ← Exports new generator
```

## Backward Compatibility

- ✅ Model A (baseline tubes) unchanged and default
- ✅ Existing datasets still valid
- ✅ All functions have sensible defaults
- ✅ Can specify projection explicitly in code

## Typical Comparative Workflow

```bash
# Generate Model A baseline (500 samples, max projection)
python -m src.dataset_generator data/synthetic_A \
    --n 500 --model A --projection max --seed 2024

# Generate Model B alternative (500 samples, max projection)
python -m src.dataset_generator data/synthetic_B \
    --n 500 --model B --projection max --seed 2024

# Compare visually
python -m src.demo_pipeline --model A --seed 100
python -m src.demo_pipeline --model B --seed 100

# Both should show 4-panel figures with aligned image/mask
```

## Dependencies

No new external dependencies. Model B uses:
- `numpy` (random walks, coordinate arithmetic)
- `scipy.ndimage.binary_dilation` (morphological thickening)

Both were already project dependencies.

## Next Steps

Use these datasets in Phase 2 (training):
- Train identical CNN/UNet architectures on Model A vs. Model B data
- Compare generalization to real microscopy
- Validate hypothesis about synthetic data effectiveness
- Document which synthetic approach better predicts real data behavior

## References

- Model A: Baseline vertical cylindrical tubes with random placement
- Model B: Random-walk 3D curves with realistic branching (mimics mitochondrial morphology)
- All generation is seeded for reproducible experiments
- Projection methods allow studying 2D representation effects

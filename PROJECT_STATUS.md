# Thesis Project Status

## Current stage
Phase 2 baseline is complete.
We have a working synthetic-data pipeline and a working 2D segmentation training pipeline.

## What is already done

### Phase 1 / synthetic pipeline
- 3D shape generation
- Model A baseline generator
- Model B polyline / branched generator
- PSF blur
- 3D -> 2D rendering
- Microscopy noise
- PNG dataset generation
- dataset sanity-check visualization

### Phase 2 / training baseline
- PyTorch dataset loader for images + masks
- deterministic train/val/test split
- paired augmentations
- 2D U-Net
- Dice + IoU metrics
- training script with:
  - pos_weight for class imbalance
  - overfit sanity mode
  - checkpoints
  - prediction snapshots
  - test evaluation

## Latest results
- synthetic_B:
  - best val Dice ≈ 0.735
  - test Dice ≈ 0.725
  - test IoU ≈ 0.575

- synthetic_A:
  - best val Dice ≈ 0.878
  - test Dice ≈ 0.915
  - test IoU ≈ 0.849

## Interpretation
- Pipeline works end-to-end.
- Model can learn from synthetic data.
- Model A currently outperforms Model B.
- Main bottleneck is now synthetic realism, especially Model B morphology.

## What is next
1. Improve Model B realism
   - longer tubules
   - better branching
   - more connected network-like structures
   - more natural thickness variation

2. Generate a new Model B dataset

3. Retrain with the same Phase 2 pipeline

4. Compare:
   - old Model B
   - improved Model B
   - Model A baseline

5. Write thesis subsection for:
   - baseline established
   - realism improvement motivation
   - updated comparison

## Still not done
- real-only experiments
- hybrid synthetic+real experiments
- robustness tests with added noise / domain shift

## Useful commands

### Visual test of generator
python -m src.demo_pipeline --model B --projection max

### 3D visualization
python -m src.viz_3d --model B

### Generate dataset
python -m src.dataset_generator data/synthetic_B_v2 --n 1000 --model B --projection center_slice --seed 42

### Check image/mask alignment
python -m src.check_dataset --data-dir data/synthetic_B_v2 --n 6

### Overfit sanity test
python -m src.train_2d --data data/synthetic_B_v2 --limit-train 16 --epochs 50 --batch-size 4 --out-dir runs/overfit_B

### Full training
python -m src.train_2d --data data/synthetic_B_v2 --epochs 20 --batch-size 8 --amp --out-dir runs/synthetic_B_v2
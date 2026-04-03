

```powershell
cd C:\0_Programming\BachelorThesis\3D-Modeling-of-Microscopic-Structures\Thesis_code
.\.venv\Scripts\Activate.ps1

python -c "import numpy, scipy, matplotlib, imageio, PIL; print('core libs ok')"
python -c "import torch, torchvision; print('torch ok')"

python -m tests.test_polyline
python -m tests.test_model_b_realism

python -m src.demo_pipeline --model B --projection max --seed 42
python -m src.viz_3d --model B --seed 42

python -m src.dataset_generator data/datasets/20260322_synthetic_B_smoke --n 24 --model B --projection center_slice --seed 42
python -m src.check_dataset --data-dir data/datasets/20260322_synthetic_B_smoke --n 6

python -m src.train_2d --data data/datasets/20260322_synthetic_B_smoke --limit-train 16 --epochs 10 --batch-size 4 --num-workers 0 --out-dir results/20260322_overfit_B_smoke



```
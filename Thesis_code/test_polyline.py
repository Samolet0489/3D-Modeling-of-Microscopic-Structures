from src.shapes import generate_mito_volume_polyline

vol, mask = generate_mito_volume_polyline(n_seeds=2, n_steps=5, branch_steps=10, seed=42)
print('done', vol.shape, mask.sum())

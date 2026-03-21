from src.shapes import generate_mito_volume_polyline

print("Testing Model B with persistence and z_scale...")
vol, mask = generate_mito_volume_polyline(
    n_seeds=2, 
    n_steps=10, 
    persistence=0.85, 
    z_scale=0.35, 
    seed_spread=10, 
    seed=42
)
print(f"✓ Success! Shape: {vol.shape}, Mask sum: {mask.sum()}, Volume sum: {vol.sum():.2f}")

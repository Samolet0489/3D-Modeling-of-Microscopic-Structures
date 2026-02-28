"""Demo script for Phase 1 of the synthetic microscopy pipeline.

This module can be executed either directly (when running inside the ``src``
directory) or via ``python -m src.demo_pipeline`` from the project root.  It
creates a single synthetic sample and displays four panels:

  * a 2‑D slice from the 3‑D volume
  * the corresponding mask slice
  * the clean (blurred) image
  * the noisy (Poisson + Gaussian) image

The purpose is to verify that all building blocks (volume generation, PSF
convolution, projection and noise) work together.
"""

import matplotlib.pyplot as plt

from .shapes import generate_mito_volume, generate_mito_volume_polyline
from .psf import gaussian_psf_kernel
from .render import render_image
from .noise import add_noise

def main(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    projection="max",
    model="A",
    n_branches=3,
    step_length=1,
    n_steps=20,
    seed=42,
    noise_seed=123,
):
    """Run the demo pipeline with selected model and projection.
    
    Parameters
    ----------
    model : {"A", "B"}
        "A" for baseline cylindrical tubes, "B" for random-walk polylines.
    projection : {"center_slice", "max", "sum"}
        Projection method for 3D→2D conversion.
    n_branches, step_length, n_steps : int/float
        Parameters for Model B (polyline generator).
    """
    if model == "A":
        vol, mask = generate_mito_volume(
            shape=shape,
            n_tubes=n_tubes,
            tube_radius=tube_radius,
            seed=seed
        )
    elif model == "B":
        vol, mask = generate_mito_volume_polyline(
            shape=shape,
            n_branches=n_branches,
            step_length=step_length,
            n_steps=n_steps,
            tube_radius=tube_radius,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'A' or 'B'.")
    psf = gaussian_psf_kernel(size=psf_size, sigma_xy=psf_sigma_xy, sigma_z=psf_sigma_z)
    clean = render_image(vol, psf, projection=projection)
    noisy = add_noise(clean, photon_level=photon_level, read_noise_sigma=read_noise_sigma, seed=noise_seed)

    # Extract volume and mask slices matching the projection method
    if projection == "center_slice":
        z_mid = vol.shape[0] // 2
        vol_slice = vol[z_mid]
        mask_slice = mask[z_mid]
    elif projection == "max":
        vol_slice = vol.max(axis=0)
        mask_slice = mask.max(axis=0)
    elif projection == "sum":
        vol_slice = vol.sum(axis=0)
        mask_slice = (mask.sum(axis=0) > 0).astype(float)
    else:
        raise ValueError(f"Unknown projection: {projection}")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title(f"Volume ({projection})")
    axes[1].imshow(mask_slice, cmap="gray")
    axes[1].set_title(f"Mask ({projection})")
    axes[2].imshow(clean, cmap="gray")
    axes[2].set_title("Clean image")
    axes[3].imshow(noisy, cmap="gray")
    axes[3].set_title("Noisy image")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run demo pipeline and show example outputs.")
    parser.add_argument(
        "--model", choices=["A", "B"], default="A",
        help="Model A (baseline tubes) or Model B (random-walk polylines)"
    )
    parser.add_argument(
        "--projection", choices=["center_slice", "max", "sum"], default="max",
        help="Projection method for 3D→2D conversion"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()
    
    main(projection=args.projection, model=args.model, seed=args.seed)
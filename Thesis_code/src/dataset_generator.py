"""Dataset generation utilities for Phase 1 (2.5D) of the project.

Functions in this module are written to be reusable in downstream training
scripts.  A simple command‑line interface is provided for producing a folder
of image/mask pairs suitable for 2D segmentation.
"""

import os
import argparse

import numpy as np
from imageio import imwrite

from .shapes import generate_mito_volume, generate_mito_volume_polyline
from .psf import gaussian_psf_kernel
from .render import render_image
from .noise import add_noise


def generate_single_example(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    projection="center_slice",
    model="A",
    n_branches=3,
    step_length=1,
    n_steps=20,
    seed=None,
):
    """Create one synthetic example using either Model A (tubes) or Model B (polyline).

    Parameters mirror those from ``generate_and_save_dataset`` and are simply
    exposed here for convenience when the function is used programmatically.
    
    Parameters
    ----------
    model : {"A", "B"}
        "A" for baseline cylindrical tubes, "B" for random-walk polylines.
    n_branches : int
        (Model B only) Number of random-walk branches.
    step_length : float
        (Model B only) Length of each step in the random walk.
    n_steps : int
        (Model B only) Number of steps per branch.
    """
    if model == "A":
        volume, mask = generate_mito_volume(
            shape=shape,
            n_tubes=n_tubes,
            tube_radius=tube_radius,
            seed=seed,
        )
    elif model == "B":
        volume, mask = generate_mito_volume_polyline(
            shape=shape,
            n_branches=n_branches,
            step_length=step_length,
            n_steps=n_steps,
            tube_radius=tube_radius,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'A' or 'B'.")

    psf = gaussian_psf_kernel(size=psf_size, sigma_xy=psf_sigma_xy, sigma_z=psf_sigma_z)
    clean = render_image(volume, psf, projection=projection)

    # normalize clean image to [0,1]
    clean = clean - clean.min()
    if clean.max() > 0:
        clean = clean / clean.max()

    noisy = add_noise(clean, photon_level=photon_level, read_noise_sigma=read_noise_sigma, seed=seed)
    return volume, mask, psf, clean, noisy


def generate_and_save_dataset(
    out_dir,
    n_samples=100,
    shape=(32, 128, 128),
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    projection="center_slice",
    model="A",
    seed=None,
):
    """Create ``n_samples`` examples and save them to ``out_dir``.

    Images are stored as ``PNG`` files under ``out_dir/images`` and masks under
    ``out_dir/masks``.  File names are ``000000.png`` style zero‑padded.
    
    Parameters
    ----------
    projection : {"center_slice", "max", "sum"}
        Projection method for 3D→2D conversion.
    model : {"A", "B"}
        "A" for baseline cylindrical tubes, "B" for random-walk polylines.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"generating {n_samples} samples using model={model}, projection={projection}, tube_radius range=[1,4]")

    for idx in range(n_samples):
        # randomize parameters per sample
        if model == "A":

            n_tubes = int(rng.integers(3, 16))
            vol_params = {"n_tubes": n_tubes}
        else:  # model == "B"
            n_branches = int(rng.integers(2, 8))
            step_length = float(rng.uniform(0.5, 2.0))
            n_steps = int(rng.integers(15, 40))
            vol_params = {
                "n_branches": n_branches,
                "step_length": step_length,
                "n_steps": n_steps,
            }

        # choose integer radius in voxels (1-4 inclusive)
        tube_radius = int(rng.integers(1, 5))
        
        # print brief info for debugging/verification
        print(f"sample {idx:06d}: model={model}, projection={projection}, tube_radius={tube_radius}")
        psf_sigma_xy = float(rng.uniform(0.5, 3.0))
        psf_sigma_z = float(rng.uniform(0.5, 5.0))
        photon_level = float(rng.uniform(2000.0, 5000.0))
        read_noise_sigma = float(rng.uniform(0.01, 0.08))

        vol, mask, psf, clean, noisy = generate_single_example(
            shape=shape,
            tube_radius=tube_radius,
            psf_size=psf_size,
            psf_sigma_xy=psf_sigma_xy,
            psf_sigma_z=psf_sigma_z,
            photon_level=photon_level,
            read_noise_sigma=read_noise_sigma,
            projection=projection,
            model=model,
            seed=rng.integers(0, 2**31 - 1),
            **vol_params,
        )

        # Extract 2D mask matching the image projection
        if projection == "center_slice":
            z_mid = mask.shape[0] // 2
            mask2d = mask[z_mid]
        elif projection == "max":
            mask2d = mask.max(axis=0)
        elif projection == "sum":
            mask2d = mask.sum(axis=0) > 0
        else:
            raise ValueError(f"Unknown projection: {projection}")

        imwrite(os.path.join(out_dir, "images", f"{idx:06d}.png"), (noisy * 255).astype(np.uint8))
        imwrite(os.path.join(out_dir, "masks", f"{idx:06d}.png"), (mask2d.astype(np.uint8) * 255))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic microscopy dataset.")
    parser.add_argument("out_dir", help="Output directory for images/masks")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to create")
    parser.add_argument(
        "--model", choices=["A", "B"], default="A",
        help="Model A (baseline tubes) or Model B (random-walk polylines)"
    )
    parser.add_argument(
        "--projection", choices=["center_slice", "max", "sum"], default="center_slice",
        help="Projection method for 3D→2D conversion"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_and_save_dataset(
        args.out_dir,
        n_samples=args.n,
        model=args.model,
        projection=args.projection,
        seed=args.seed
    )

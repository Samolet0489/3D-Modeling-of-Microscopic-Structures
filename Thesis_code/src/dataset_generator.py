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
    n_seeds=4,
    step_length=1,
    n_steps=80,
    branch_prob=0.08,
    branch_steps=40,
    radius_min=1,
    radius_max=4,
    persistence=0.85,
    z_scale=0.35,
    seed_spread=20,
    seed=None,
):
    """Create one synthetic example using either Model A (tubes) or Model B (polyline).

    Parameters mirror those from ``generate_and_save_dataset`` and are simply
    exposed here for convenience when the function is used programmatically.
    
    Parameters
    ----------
    model : {"A", "B"}
        "A" for baseline cylindrical tubes, "B" for random-walk polylines.
    n_seeds : int
        (Model B only) Number of initial seed points for random walks.
    step_length : float
        (Model B only) Length of each step in the random walk.
    n_steps : int
        (Model B only) Number of steps for each main walker.
    branch_prob : float
        (Model B only) Probability of spawning a new branch at each step.
    branch_steps : int
        (Model B only) Length (in steps) given to any spawned branch.
    radius_min : float
        (Model B only) Minimum radius for tubules.
    radius_max : float
        (Model B only) Maximum radius for tubules.
    persistence : float
        (Model B only) Direction persistence (0=random, 1=straight). Default 0.85.
    z_scale : float
        (Model B only) Planar bias: z-motion scaling. Default 0.35 (mostly XY plane).
    seed_spread : float
        (Model B only) Colony center seed spread (std dev in voxels). Default 20.
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
            n_seeds=n_seeds,
            step_length=step_length,
            n_steps=n_steps,
            branch_prob=branch_prob,
            branch_steps=branch_steps,
            radius_min=radius_min,
            radius_max=radius_max,
            persistence=persistence,
            z_scale=z_scale,
            seed_spread=seed_spread,
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
    n_seeds=4,
    step_length=1.5,
    n_steps=120,
    branch_prob=0.08,
    branch_steps=40,
    radius_min=1,
    radius_max=4,
    persistence=0.85,
    z_scale=0.35,
    seed_spread=20,
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

    print(
        "Run config:",
        f"model={model}, projection={projection}, n_samples={n_samples}, shape={shape},",
        f"psf_sigma_xy={psf_sigma_xy}, psf_sigma_z={psf_sigma_z},",
        f"n_seeds={n_seeds}, step_length={step_length}, n_steps={n_steps},",
        f"branch_prob={branch_prob}, branch_steps={branch_steps},",
        f"radius_min={radius_min}, radius_max={radius_max},",
        f"persistence={persistence}, z_scale={z_scale}, seed_spread={seed_spread}"
    )
    print(f"generating {n_samples} samples using model={model}")

    for idx in range(n_samples):
        # randomize parameters per sample
        if model == "A":

            n_tubes = int(rng.integers(3, 16))
            vol_params = {"n_tubes": n_tubes}
        else:  # model == "B"
            n_seeds = int(rng.integers(2, 8))
            step_length = float(rng.uniform(0.5, 2.0))
            n_steps = int(rng.integers(60, 100))
            branch_prob = float(rng.uniform(0.05, 0.15))
            branch_steps = int(rng.integers(20, 60))
            persistence_var = float(rng.uniform(0.75, 0.95))
            z_scale_var = float(rng.uniform(0.2, 0.5))
            seed_spread_var = float(rng.uniform(10, 35))
            vol_params = {
                "n_seeds": n_seeds,
                "step_length": step_length,
                "n_steps": n_steps,
                "branch_prob": branch_prob,
                "branch_steps": branch_steps,
                "radius_min": 1,
                "radius_max": 4,
                "persistence": persistence_var,
                "z_scale": z_scale_var,
                "seed_spread": seed_spread_var,
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
    # model-B realism parameters
    parser.add_argument("--n-seeds", type=int, default=4, help="(Model B) number of initial seed points")
    parser.add_argument("--step-length", type=float, default=1.5, help="(Model B) step length")
    parser.add_argument("--n-steps", type=int, default=120, help="(Model B) steps per main walker")
    parser.add_argument("--branch-prob", type=float, default=0.08, help="(Model B) probability of branching")
    parser.add_argument("--branch-steps", type=int, default=40, help="(Model B) length of spawned branches")
    parser.add_argument("--radius-min", type=int, default=1, help="(Model B) minimum radius")
    parser.add_argument("--radius-max", type=int, default=4, help="(Model B) maximum radius")
    parser.add_argument("--persistence", type=float, default=0.85, help="(Model B) direction persistence")
    parser.add_argument("--z-scale", type=float, default=0.35, help="(Model B) z-plane bias factor")
    parser.add_argument("--seed-spread", type=float, default=20, help="(Model B) colony center seed spread")
    args = parser.parse_args()
    # summary of run configuration for logging
    print("Run config:", vars(args))

    generate_and_save_dataset(
        args.out_dir,
        n_samples=args.n,
        model=args.model,
        projection=args.projection,
        seed=args.seed,
        n_seeds=args.n_seeds,
        step_length=args.step_length,
        n_steps=args.n_steps,
        branch_prob=args.branch_prob,
        branch_steps=args.branch_steps,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        persistence=args.persistence,
        z_scale=args.z_scale,
        seed_spread=args.seed_spread,
    )

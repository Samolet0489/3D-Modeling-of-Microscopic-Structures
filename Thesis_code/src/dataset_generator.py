"""Dataset generation utilities for Phase 1 (2.5D) of the project.

Functions in this module are written to be reusable in downstream training
scripts.  A simple command‑line interface is provided for producing a folder
of image/mask pairs suitable for 2D segmentation.
"""

import os
import argparse

import numpy as np
from imageio import imwrite

from .shapes import generate_mito_volume
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
    seed=None,
):
    """Create one synthetic example using the mitochondria tube model.

    Parameters mirror those from ``generate_and_save_dataset`` and are simply
    exposed here for convenience when the function is used programmatically.
    """
    volume, mask = generate_mito_volume(
        shape=shape,
        n_tubes=n_tubes,
        tube_radius=tube_radius,
        seed=seed,
    )

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
    n_tubes=5,
    tube_radius=2,
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
):
    """Create ``n_samples`` examples and save them to ``out_dir``.

    Images are stored as ``PNG`` files under ``out_dir/images`` and masks under
    ``out_dir/masks``.  File names are ``000000.png`` style zero‑padded.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    for idx in range(n_samples):
        vol, mask, psf, clean, noisy = generate_single_example(
            shape=shape,
            n_tubes=n_tubes,
            tube_radius=tube_radius,
            psf_size=psf_size,
            psf_sigma_xy=psf_sigma_xy,
            psf_sigma_z=psf_sigma_z,
            photon_level=photon_level,
            read_noise_sigma=read_noise_sigma,
            projection="center_slice",
            seed=None,
        )

        # determine 2D mask matching the (center-slice) projection used
        z_mid = mask.shape[0] // 2
        mask2d = mask[z_mid]

        imwrite(os.path.join(out_dir, "images", f"{idx:06d}.png"), (noisy * 255).astype(np.uint8))
        imwrite(os.path.join(out_dir, "masks", f"{idx:06d}.png"), (mask2d.astype(np.uint8) * 255))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic microscopy dataset.")
    parser.add_argument("out_dir", help="Output directory for images/masks")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to create")
    args = parser.parse_args()

    generate_and_save_dataset(args.out_dir, n_samples=args.n)

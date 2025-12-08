import numpy as np
import matplotlib.pyplot as plt

from shapes import create_ellipsoid_volume
from psf import make_gaussian_psf, apply_psf
from render import project_volume, get_middle_slice
from noise import add_microscopy_noise


def generate_single_example():
    # Create 3D ellipsoid volume + mask
    volume_shape = (32, 128, 128)  # (Z, Y, X)
    volume, mask = create_ellipsoid_volume(
        volume_shape=volume_shape,
        radii=(6, 22, 18),
        intensity=1.0,
    )

    # PSF and blurring
    psf = make_gaussian_psf(
        size=(17, 17, 9),
        sigma=(2.0, 1.5, 1.5),
    )
    blurred_volume = apply_psf(volume, psf)

    # Projection to 2D (clean image)
    clean_image = project_volume(blurred_volume, mode="max")

    # Normalize clean image to [0,1] for nicer plots
    clean_image = clean_image - clean_image.min()
    if clean_image.max() > 0:
        clean_image = clean_image / clean_image.max()

    # Add noise
    noisy_image = add_microscopy_noise(
        clean_image,
        peak_photons=5000,
        read_sigma=0.03,
    )

    return volume, mask, psf, clean_image, noisy_image


def show_quick_figure(volume, mask, psf, clean_image, noisy_image):
    """
    Plot four panels:
      (1) volume middle slice (mask)
      (2) PSF middle slice
      (3) clean projected image
      (4) noisy image
    """
    # middle Z slice of mask
    mask_slice = get_middle_slice(mask.astype(np.float32))
    # middle Z slice of PSF
    psf_slice = get_middle_slice(psf)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    ax = axes[0, 0]
    im0 = ax.imshow(mask_slice, cmap="gray")
    ax.set_title("Volume middle slice (mask)")
    ax.axis("off")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im1 = ax.imshow(psf_slice / psf_slice.max(), cmap="magma")
    ax.set_title("PSF middle slice")
    ax.axis("off")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im2 = ax.imshow(clean_image, cmap="gray")
    ax.set_title("Clean projected image")
    ax.axis("off")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im3 = ax.imshow(noisy_image, cmap="gray")
    ax.set_title("Noisy image")
    ax.axis("off")
    fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)  # for reproducibility while debugging

    volume, mask, psf, clean_image, noisy_image = generate_single_example()
    show_quick_figure(volume, mask, psf, clean_image, noisy_image)

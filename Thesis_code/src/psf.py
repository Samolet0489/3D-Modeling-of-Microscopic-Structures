import numpy as np
from scipy.signal import fftconvolve


def make_gaussian_psf(
    size=(17, 17, 9),
    sigma=(2.0, 2.0, 1.0),
):
    """
    Create a 3D Gaussian PSF kernel.

    size: (Z, Y, X) kernel size (odd numbers work best)
    sigma: (sz, sy, sx) standard deviations in voxels
    """
    z_size, y_size, x_size = size
    sz, sy, sx = sigma

    z = np.linspace(-(z_size // 2), z_size // 2, z_size)
    y = np.linspace(-(y_size // 2), y_size // 2, y_size)
    x = np.linspace(-(x_size // 2), x_size // 2, x_size)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    psf = np.exp(-(
        (Z ** 2) / (2 * sz ** 2) +
        (Y ** 2) / (2 * sy ** 2) +
        (X ** 2) / (2 * sx ** 2)
    ))

    psf /= psf.sum()  # normalize so total energy = 1
    return psf.astype(np.float32)


def apply_psf(volume, psf):
    """
    Convolve 3D volume with 3D PSF using FFT.
    Returns blurred volume with same shape.
    """
    blurred = fftconvolve(volume, psf, mode="same")
    return blurred.astype(np.float32)

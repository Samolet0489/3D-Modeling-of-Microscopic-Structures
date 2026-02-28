import numpy as np
from .psf import apply_psf


def project_volume(volume, mode="max"):
    """
    Project a 3D volume (Z, Y, X) to 2D (Y, X).

    mode:
      - "max" : max-intensity projection
      - "sum" : sum over Z
    """
    if mode == "max":
        return volume.max(axis=0)
    elif mode == "sum":
        return volume.sum(axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_middle_slice(volume):
    """
    Return middle Z slice (for visualization).
    """
    z_mid = volume.shape[0] // 2
    return volume[z_mid, :, :]


def render_image(volume, psf, projection="center_slice"):
    """Blur a 3D volume with a PSF and return a 2D image.

    Parameters
    ----------
    volume : ndarray
        3D input volume.
    psf : ndarray
        3D point-spread function kernel.
    projection : str
        How to reduce the blurred volume to 2D.  Supported values:
        - "center_slice" : return the middle Z slice of the blurred volume.
        - any mode accepted by :func:`project_volume` ("max", "sum").

    Returns
    -------
    image : ndarray
        2D float image.
    """
    blurred = apply_psf(volume, psf)

    if projection == "center_slice":
        z_mid = blurred.shape[0] // 2
        return blurred[z_mid]
    else:
        # delegate to projection routines (max, sum)
        return project_volume(blurred, mode=projection)

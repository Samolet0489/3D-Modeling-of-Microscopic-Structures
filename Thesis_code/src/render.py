import numpy as np
from .psf import apply_psf


def project_volume(volume, mode="max", slab_thickness=5):
    """
    Project a 3D volume (Z, Y, X) to 2D (Y, X).

    mode:
      - "max"         : max-intensity projection over full Z
      - "sum"         : sum over Z
      - "focus_max"   : max projection only over a central Z slab
    """
    if mode == "max":
        return volume.max(axis=0)

    if mode == "sum":
        return volume.sum(axis=0)

    if mode == "focus_max":
        slab_thickness = max(1, int(slab_thickness))
        z_mid = volume.shape[0] // 2
        half = slab_thickness // 2
        z0 = max(0, z_mid - half)
        z1 = min(volume.shape[0], z_mid + half + 1)
        return volume[z0:z1].max(axis=0)

    raise ValueError(f"Unknown mode: {mode}")


def get_middle_slice(volume):
    """
    Return middle Z slice (for visualization).
    """
    z_mid = volume.shape[0] // 2
    return volume[z_mid, :, :]


def render_image(volume, psf, projection="center_slice", slab_thickness=5):
    """Blur a 3D volume with a PSF and return a 2D image."""
    blurred = apply_psf(volume, psf)

    if projection == "center_slice":
        z_mid = blurred.shape[0] // 2
        return blurred[z_mid]

    return project_volume(blurred, mode=projection, slab_thickness=slab_thickness)

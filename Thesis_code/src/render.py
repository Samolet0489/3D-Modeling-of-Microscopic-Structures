import numpy as np


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

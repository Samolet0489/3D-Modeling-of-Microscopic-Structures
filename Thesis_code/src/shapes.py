import numpy as np


def create_ellipsoid_volume(
    volume_shape=(32, 128, 128),
    center=None,
    radii=(6, 20, 20),
    intensity=1.0,
):
    """
    Create a 3D volume with a single solid ellipsoid.

    volume_shape: (Z, Y, X)
    center: (z0, y0, x0) in voxel coordinates. If None -> center of volume.
    radii: (rz, ry, rx) semi-axes of ellipsoid in voxels.
    intensity: value inside the ellipsoid.
    """
    z_size, y_size, x_size = volume_shape

    if center is None:
        center = (z_size / 2, y_size / 2, x_size / 2)

    z0, y0, x0 = center
    rz, ry, rx = radii

    # coordinate grids
    z = np.arange(z_size)[:, None, None]
    y = np.arange(y_size)[None, :, None]
    x = np.arange(x_size)[None, None, :]

    # Ellipsoid equation <= 1 is inside
    ellipsoid_eq = (((z - z0) / rz) ** 2 +
                    ((y - y0) / ry) ** 2 +
                    ((x - x0) / rx) ** 2)

    mask = ellipsoid_eq <= 1.0

    volume = np.zeros(volume_shape, dtype=np.float32)
    volume[mask] = intensity

    # mask is boolean GT, volume is float intensity
    return volume, mask

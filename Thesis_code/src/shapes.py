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


def generate_mito_volume(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    intensity=1.0,
    seed=None,
):
    """Create a synthetic volume resembling a mitochondrion made of cylindrical tubes.

    The volume is a stack of circular cross sections with randomly placed
    vertical tubes.  Each tube spans the entire Z axis and has the given radius.

    Parameters
    ----------
    shape : tuple
        (Z, Y, X) dimensions of the volume.
    n_tubes : int
        Number of individual tubes to generate.  Tubes may overlap.
    tube_radius : float
        Radius of each tube in voxels (in the Y/X plane).
    intensity : float
        Value assigned inside the tubes.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    volume : ndarray
        Float32 volume with intensity inside the tubes.
    mask : ndarray
        Boolean mask indicating tube locations.
    """
    z_size, y_size, x_size = shape

    # reproducibility
    rng = np.random.RandomState(seed)

    volume = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    # prepare coordinate grids for a single slice
    y_grid = np.arange(y_size)[:, None]
    x_grid = np.arange(x_size)[None, :]

    for _ in range(n_tubes):
        # choose a random center in the Y/X plane ensuring the tube fits
        cy = rng.randint(tube_radius, y_size - tube_radius)
        cx = rng.randint(tube_radius, x_size - tube_radius)

        # compute circular cross-section mask
        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        slice_mask = dist <= tube_radius

        # apply to all Z slices
        mask[:, slice_mask] = True
        volume[:, slice_mask] = intensity

    return volume, mask

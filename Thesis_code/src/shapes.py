import numpy as np
from scipy.ndimage import binary_dilation


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

    The radius argument is cast to an integer voxel radius internally so that
    downstream operations using it for indexing won't fail if the caller passes
    a float.  This keeps the public API the same.

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
    # defensive cast: ensure integer radius
    tube_radius = int(np.ceil(tube_radius))

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


def generate_mito_volume_polyline(
    shape=(32, 128, 128),
    n_branches=3,
    step_length=1,
    n_steps=20,
    tube_radius=2,
    intensity=1.0,
    seed=None,
):
    """Create a synthetic volume resembling a mitochondrion using random-walk polyline centerlines.

    The tube_radius parameter is cast to an integer at entry so the morphological
    dilation and boundary clipping behave predictably.

    This model generates curved, interconnected tubules by simulating random walks
    in 3D space. Each branch starts at random locations and follows a random-walk
    path, then the centerline is thickened via morphological dilation.

    Parameters
    ----------
    shape : tuple
        (Z, Y, X) dimensions of the volume.
    n_branches : int
        Number of independent random-walk branches to generate.  Branches may overlap.
    step_length : float
        Length of each step in the random walk (in voxels).
    n_steps : int
        Number of steps for each random walk.
    tube_radius : float
        Radius of each tubule after dilation (in voxels).
    intensity : float
        Value assigned inside the tubules.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    volume : ndarray
        Float32 volume with intensity inside the tubules.
    mask : ndarray
        Boolean mask indicating tubule locations.
    """
    # defensive cast: ensure integer radius for boundary checks/dilation
    tube_radius = int(np.ceil(tube_radius))

    z_size, y_size, x_size = shape

    # Use new-style random generator for consistency with dataset_generator
    rng = np.random.default_rng(seed)

    # Initialize volume and mask
    volume = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    for _ in range(n_branches):
        # Random starting position within bounds
        start_z = rng.integers(tube_radius, z_size - tube_radius)
        start_y = rng.integers(tube_radius, y_size - tube_radius)
        start_x = rng.integers(tube_radius, x_size - tube_radius)

        # Trace random walk
        path = [(start_z, start_y, start_x)]
        z, y, x = float(start_z), float(start_y), float(start_x)

        for _ in range(n_steps):
            # Random direction in 3D (uniform on unit sphere)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)

            # Move along direction
            z += direction[0] * step_length
            y += direction[1] * step_length
            x += direction[2] * step_length

            # Clip to boundaries
            z = np.clip(z, tube_radius, z_size - tube_radius - 1)
            y = np.clip(y, tube_radius, y_size - tube_radius - 1)
            x = np.clip(x, tube_radius, x_size - tube_radius - 1)

            path.append((z, y, x))

        # Create a thin line mask along the path
        line_mask = np.zeros(shape, dtype=bool)

        for i in range(len(path) - 1):
            z0, y0, x0 = path[i]
            z1, y1, x1 = path[i + 1]

            # Bresenham-like line drawing in 3D (simple version: round to nearest voxel)
            num_intermediate = max(int(np.ceil(step_length * 2)), 2)
            for t in np.linspace(0, 1, num_intermediate):
                zi = int(np.round(z0 + t * (z1 - z0)))
                yi = int(np.round(y0 + t * (y1 - y0)))
                xi = int(np.round(x0 + t * (x1 - x0)))

                # Bounds check
                if 0 <= zi < z_size and 0 <= yi < y_size and 0 <= xi < x_size:
                    line_mask[zi, yi, xi] = True

        # Thicken the line via morphological dilation
        # Convert to int for dilation, then back to bool
        radius = int(np.ceil(tube_radius))
        dilation_struct = np.zeros((2 * radius + 1, 2 * radius + 1, 2 * radius + 1),
                                   dtype=bool)
        z_d, y_d, x_d = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        dilation_struct[z_d**2 + y_d**2 + x_d**2 <= radius**2] = True

        thickened = binary_dilation(line_mask, structure=dilation_struct, iterations=1)

        # Merge into volume and mask
        mask |= thickened
        volume[thickened] = intensity

    return volume, mask

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
    intensity=1.0,
    seed=None,
):
    """Create a synthetic volume resembling a mitochondrion using persistent random-walk polylines with planar bias.

    This model generates filament-like, interconnected tubular networks via persistent random walks
    in 3D space with occasional branching. Direction persistence and planar bias (reduced z motion)
    create elongated, realistic mitochondrial morphology. Seed points cluster around a colony center.
    Thickness is applied via morphological dilation per radius for smooth, continuous tubes.

    Parameters
    ----------
    shape : tuple
        (Z, Y, X) dimensions of the volume.
    n_seeds : int, default 4
        Number of initial seed points for random walks.
    step_length : float, default 1
        Length of each step in the random walk (in voxels).
    n_steps : int, default 80
        Number of steps for each main walker.
    branch_prob : float, default 0.08
        Probability of spawning a new branch at each step of a main walker.
    branch_steps : int, default 40
        Number of steps allocated to any spawned branch walker.
    radius_min : float, default 1
        Minimum radius for tubules (in voxels).
    radius_max : float, default 4
        Maximum radius for tubules (in voxels).
    persistence : float, default 0.85
        Persistence of direction: dir = normalize(persistence*dir + (1-persistence)*noise).
        Higher values = straighter, more filament-like paths.
    z_scale : float, default 0.35
        Scaling factor for z-component before normalization, creating planar bias.
        Lower values = motion mostly in XY plane (default 0.35 reduces z motion ~65%).
    seed_spread : float, default 20
        Spread (std dev in voxels) of Gaussian cloud around colony center for initial seeds.
    intensity : float, default 1.0
        Value assigned inside the tubules.
    seed : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    volume : ndarray
        Float32 volume with intensity inside the tubules.
    mask : ndarray
        Boolean mask indicating tubule locations.
    """
    # Defensive casts
    radius_min = int(np.ceil(radius_min))
    radius_max = int(np.ceil(radius_max))
    persistence = np.clip(persistence, 0.0, 1.0)

    z_size, y_size, x_size = shape
    rng = np.random.default_rng(seed)

    # Sample one colony center
    colony_z = rng.uniform(radius_max, z_size - radius_max)
    colony_y = rng.uniform(radius_max, y_size - radius_max)
    colony_x = rng.uniform(radius_max, x_size - radius_max)

    # Initialize walkers from Gaussian cloud around colony center
    active_walkers = []
    for _ in range(n_seeds):
        z = colony_z + rng.normal(loc=0, scale=seed_spread)
        y = colony_y + rng.normal(loc=0, scale=seed_spread)
        x = colony_x + rng.normal(loc=0, scale=seed_spread)
        z = np.clip(z, radius_max, z_size - radius_max - 1)
        y = np.clip(y, radius_max, y_size - radius_max - 1)
        x = np.clip(x, radius_max, x_size - radius_max - 1)

        # Initialize with random direction
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        active_walkers.append({
            'pos': [z, y, x],
            'direction': direction,
            'steps_left': n_steps,
            'can_spawn': True,
            'radii': []  # Track radius assignment for each position
        })

    # Collect all points with their assigned radii
    all_points = []  # list of (z, y, x, r)
    point_radii = {}  # map (zi, yi, xi) -> r for persistence

    while active_walkers:
        new_walkers = []
        active_walkers = [w for w in active_walkers if w['steps_left'] > 0]
        if not active_walkers:
            break

        for walker in active_walkers:
            # Persistent random walk:
            # dir_new = normalize(persistence*dir_old + (1-persistence)*noise)
            noise = rng.normal(size=3)
            direction = persistence * walker['direction'] + (1 - persistence) * noise

            # Apply planar bias: scale z component
            direction[0] *= z_scale  # z-component reduced
            direction /= np.linalg.norm(direction)
            walker['direction'] = direction

            # Move along direction
            walker['pos'][0] += direction[0] * step_length
            walker['pos'][1] += direction[1] * step_length
            walker['pos'][2] += direction[2] * step_length

            # Clip to boundaries
            walker['pos'][0] = np.clip(walker['pos'][0], radius_max, z_size - radius_max - 1)
            walker['pos'][1] = np.clip(walker['pos'][1], radius_max, y_size - radius_max - 1)
            walker['pos'][2] = np.clip(walker['pos'][2], radius_max, x_size - radius_max - 1)

            # Assign random radius
            r = rng.integers(radius_min, radius_max + 1)
            all_points.append((walker['pos'][0], walker['pos'][1], walker['pos'][2], r))

            walker['steps_left'] -= 1

            # Spawn new branch with probability, inheriting direction with perturbation
            if walker['can_spawn'] and rng.random() < branch_prob:
                low = max(1, int(branch_steps * 0.75))
                high = int(branch_steps * 1.25) + 1

                # Perturb current direction slightly
                pert_noise = rng.normal(size=3, scale=0.15)
                new_dir = walker['direction'] + pert_noise
                new_dir /= np.linalg.norm(new_dir)

                new_walker = {
                    'pos': walker['pos'].copy(),
                    'direction': new_dir,
                    'steps_left': rng.integers(low, high),
                    'can_spawn': False,
                    'radii': []
                }
                new_walkers.append(new_walker)

        active_walkers.extend(new_walkers)

    # Build centerline mask: convert all points to integer voxels
    centerline = np.zeros(shape, dtype=bool)
    point_to_radius = {}  # Map voxel coord to radius for later dilation

    for z, y, x, r in all_points:
        zi, yi, xi = int(np.round(z)), int(np.round(y)), int(np.round(x))
        if 0 <= zi < z_size and 0 <= yi < y_size and 0 <= xi < x_size:
            centerline[zi, yi, xi] = True
            # Store maximum radius for this voxel (in case multiple points map to same voxel)
            key = (zi, yi, xi)
            point_to_radius[key] = max(point_to_radius.get(key, 0), r)

    # Apply tube thickening via dilation per radius
    # For each radius, create a binary_dilation structure (ball) and dilate
    mask = np.zeros(shape, dtype=bool)

    for r in range(radius_min, radius_max + 1):
        # Find all centerline points assigned to this radius
        pts_r = np.zeros(shape, dtype=bool)
        for (zi, yi, xi), assigned_r in point_to_radius.items():
            if assigned_r == r:
                pts_r[zi, yi, xi] = True

        if not pts_r.any():
            continue

        # Create ball structure with radius r
        ball_struct = np.zeros((2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=bool)
        zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        ball_struct[zz**2 + yy**2 + xx**2 <= r**2] = True

        # Dilate and merge
        dilated = binary_dilation(pts_r, structure=ball_struct, iterations=1)
        mask |= dilated

    # Create volume
    volume = np.zeros(shape, dtype=np.float32)
    volume[mask] = intensity

    return volume, mask

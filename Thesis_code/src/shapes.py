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

    z = np.arange(z_size)[:, None, None]
    y = np.arange(y_size)[None, :, None]
    x = np.arange(x_size)[None, None, :]

    ellipsoid_eq = (
        ((z - z0) / rz) ** 2
        + ((y - y0) / ry) ** 2
        + ((x - x0) / rx) ** 2
    )

    mask = ellipsoid_eq <= 1.0

    volume = np.zeros(volume_shape, dtype=np.float32)
    volume[mask] = intensity

    return volume, mask


def generate_mito_volume(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    intensity=1.0,
    seed=None,
):
    """Create a synthetic volume resembling a mitochondrion made of cylindrical tubes."""
    tube_radius = int(np.ceil(tube_radius))

    z_size, y_size, x_size = shape
    rng = np.random.RandomState(seed)

    volume = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)

    y_grid = np.arange(y_size)[:, None]
    x_grid = np.arange(x_size)[None, :]

    for _ in range(n_tubes):
        cy = rng.randint(tube_radius, y_size - tube_radius)
        cx = rng.randint(tube_radius, x_size - tube_radius)

        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        slice_mask = dist <= tube_radius

        mask[:, slice_mask] = True
        volume[:, slice_mask] = intensity

    return volume, mask


def _safe_unit_vector(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return vec / norm


def _safe_box(shape, margin):
    z_size, y_size, x_size = shape
    low = np.array([margin, margin, margin], dtype=np.float32)
    high = np.array(
        [
            z_size - margin - 1,
            y_size - margin - 1,
            x_size - margin - 1,
        ],
        dtype=np.float32,
    )
    return low, high


def _clip_position(pos, shape, margin):
    low, high = _safe_box(shape, margin)
    pos = np.asarray(pos, dtype=np.float32).copy()
    return np.clip(pos, low, high)


def _rasterize_segment(start, end, oversample=3.0):
    """Return interpolated points along a line segment."""
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)

    segment = end - start
    dist = float(np.linalg.norm(segment))
    n_points = max(2, int(np.ceil(dist * oversample)) + 1)

    ts = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    return [start + t * segment for t in ts]


def _make_ball_structure(radius):
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz**2 + yy**2 + xx**2) <= r**2


def _border_repulsion(pos, shape, margin, band):
    """
    Return a soft repulsion vector that pushes walkers away from borders
    before they actually hit the boundary.
    """
    low, high = _safe_box(shape, margin)
    pos = np.asarray(pos, dtype=np.float32)

    rep = np.zeros(3, dtype=np.float32)

    for axis in range(3):
        low_band = low[axis] + band
        high_band = high[axis] - band

        if pos[axis] < low_band:
            rep[axis] += (low_band - pos[axis]) / max(band, 1e-6)
        elif pos[axis] > high_band:
            rep[axis] -= (pos[axis] - high_band) / max(band, 1e-6)

    return rep


def _turn_inward_from_bounds(prev_pos, proposed_dir, shape, margin, colony_center):
    """
    If a proposed move would leave the safe interior box, reflect the
    offending direction components and bias the walker back toward the colony.
    """
    low, high = _safe_box(shape, margin)

    new_dir = np.asarray(proposed_dir, dtype=np.float32).copy()
    proposed_pos = prev_pos + new_dir

    hit_boundary = False
    for axis in range(3):
        if proposed_pos[axis] < low[axis]:
            new_dir[axis] = abs(new_dir[axis])
            hit_boundary = True
        elif proposed_pos[axis] > high[axis]:
            new_dir[axis] = -abs(new_dir[axis])
            hit_boundary = True

    if hit_boundary:
        inward = np.asarray(colony_center, dtype=np.float32) - np.asarray(prev_pos, dtype=np.float32)
        inward = _safe_unit_vector(inward)
        new_dir = _safe_unit_vector(0.70 * new_dir + 0.30 * inward)

    return new_dir, hit_boundary


def generate_mito_volume_polyline(
    shape=(32, 128, 128),
    n_seeds=4,
    step_length=1,
    n_steps=80,
    branch_prob=0.08,
    branch_steps=40,
    radius_min=1,
    radius_max=3,
    persistence=0.85,
    z_scale=0.35,
    seed_spread=20,
    intensity=1.0,
    seed=None,
):
    """
    Create a synthetic volume using smoother persistent random-walk tubules.

    This version is still the familiar baseline, but slightly thinner by default.
    """
    radius_min = int(np.ceil(radius_min))
    radius_max = int(np.ceil(radius_max))
    if radius_min > radius_max:
        raise ValueError("radius_min must be <= radius_max")

    persistence = float(np.clip(persistence, 0.0, 1.0))

    z_size, y_size, x_size = shape
    rng = np.random.default_rng(seed)

    safe_margin = radius_max + 4

    colony_z = float(
        np.clip(
            rng.normal(loc=z_size / 2.0, scale=max(2.0, z_size * 0.07)),
            safe_margin,
            z_size - safe_margin - 1,
        )
    )

    xy_center_margin = safe_margin + max(8.0, seed_spread * 0.8)

    colony_y = float(
        rng.uniform(
            xy_center_margin,
            y_size - xy_center_margin - 1,
        )
    )
    colony_x = float(
        rng.uniform(
            xy_center_margin,
            x_size - xy_center_margin - 1,
        )
    )

    colony_center = np.array([colony_z, colony_y, colony_x], dtype=np.float32)

    seed_spread_z = max(2.0, seed_spread * max(0.20, z_scale * 0.8))
    turn_band = max(8.0, step_length * 5.0)

    active_walkers = []
    for _ in range(n_seeds):
        start_pos = np.array(
            [
                colony_z + rng.normal(0.0, seed_spread_z),
                colony_y + rng.normal(0.0, seed_spread),
                colony_x + rng.normal(0.0, seed_spread),
            ],
            dtype=np.float32,
        )
        start_pos = _clip_position(start_pos, shape, safe_margin)

        direction = _safe_unit_vector(rng.normal(size=3))
        start_radius = float(rng.uniform(radius_min, radius_max))

        active_walkers.append(
            {
                "pos": start_pos,
                "direction": direction,
                "steps_left": int(n_steps),
                "can_spawn": True,
                "radius": start_radius,
                "edge_hits": 0,
            }
        )

    point_to_radius = {}

    while active_walkers:
        new_walkers = []
        survivors = []

        for walker in active_walkers:
            if walker["steps_left"] <= 0:
                continue

            prev_pos = walker["pos"].copy()

            noise = rng.normal(size=3)
            repulsion = _border_repulsion(prev_pos, shape, safe_margin, turn_band)
            repulsion[0] *= 0.6

            new_direction = (
                persistence * walker["direction"]
                + (1.0 - persistence) * noise
                + 0.55 * repulsion
            )

            new_direction[0] *= z_scale
            new_direction = _safe_unit_vector(new_direction)

            step_vec = new_direction * float(step_length)
            corrected_dir, hit_boundary = _turn_inward_from_bounds(
                prev_pos=prev_pos,
                proposed_dir=step_vec,
                shape=shape,
                margin=safe_margin,
                colony_center=colony_center,
            )

            if hit_boundary:
                walker["edge_hits"] += 1
            else:
                walker["edge_hits"] = max(0, walker["edge_hits"] - 1)

            if walker["edge_hits"] >= 3:
                continue

            new_direction = _safe_unit_vector(corrected_dir)
            new_pos = prev_pos + new_direction * float(step_length)
            new_pos = _clip_position(new_pos, shape, safe_margin)

            walker["radius"] = float(
                np.clip(
                    walker["radius"] + rng.normal(loc=0.0, scale=0.10),
                    radius_min,
                    radius_max,
                )
            )
            radius_i = int(np.clip(np.round(walker["radius"]), radius_min, radius_max))

            for point in _rasterize_segment(prev_pos, new_pos, oversample=3.0):
                zi, yi, xi = np.round(point).astype(int)
                if 0 <= zi < z_size and 0 <= yi < y_size and 0 <= xi < x_size:
                    key = (zi, yi, xi)
                    point_to_radius[key] = max(point_to_radius.get(key, 0), radius_i)

            walker["pos"] = new_pos
            walker["direction"] = new_direction
            walker["steps_left"] -= 1

            if walker["steps_left"] > 0:
                survivors.append(walker)

            if walker["can_spawn"] and walker["edge_hits"] == 0 and rng.random() < branch_prob:
                low = max(1, int(branch_steps * 0.75))
                high = max(low + 1, int(branch_steps * 1.25) + 1)

                branch_direction = _safe_unit_vector(
                    walker["direction"] + rng.normal(loc=0.0, scale=0.20, size=3)
                )
                branch_radius = float(
                    np.clip(
                        walker["radius"] + rng.normal(loc=0.0, scale=0.15),
                        radius_min,
                        radius_max,
                    )
                )

                new_walkers.append(
                    {
                        "pos": walker["pos"].copy(),
                        "direction": branch_direction,
                        "steps_left": int(rng.integers(low, high)),
                        "can_spawn": False,
                        "radius": branch_radius,
                        "edge_hits": 0,
                    }
                )

        active_walkers = survivors + new_walkers

    radius_to_points = {r: [] for r in range(radius_min, radius_max + 1)}
    for coord, r in point_to_radius.items():
        radius_to_points[r].append(coord)

    mask = np.zeros(shape, dtype=bool)

    for r in range(radius_min, radius_max + 1):
        coords = radius_to_points[r]
        if not coords:
            continue

        pts_r = np.zeros(shape, dtype=bool)
        zz, yy, xx = zip(*coords)
        pts_r[np.array(zz), np.array(yy), np.array(xx)] = True

        ball_struct = _make_ball_structure(r)
        dilated = binary_dilation(pts_r, structure=ball_struct, iterations=1)
        mask |= dilated

    volume = np.zeros(shape, dtype=np.float32)
    volume[mask] = intensity

    return volume, mask

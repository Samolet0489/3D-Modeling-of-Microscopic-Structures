import numpy as np
import open3d as o3d

def cylinder_between(p0, p1, radius=0.1, resolution=16):
    """
    Create an Open3D cylinder mesh of given radius
    connecting p0 to p1.
    """
    # 1) vector from p0→p1
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-6:
        return None

    # 2) make cylinder along +Z of height=length
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=1
    )

    # 3) compute rotation from [0,0,1] to v_unit
    v_unit = v / length
    z_axis = np.array([0, 0, 1.0])
    dot = np.dot(z_axis, v_unit)
    axis = np.cross(z_axis, v_unit)
    axis_len = np.linalg.norm(axis)

    if axis_len < 1e-6:
        # parallel or anti-parallel
        if dot < 0:
            # 180° about X
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([np.pi, 0, 0])
            )
        else:
            R = np.eye(3)
    else:
        axis = axis / axis_len
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cyl.rotate(R, center=(0, 0, 0))

    # 4) move it so its center lands at (p0+p1)/2
    midpoint = (p0 + p1) / 2.0
    cyl.translate(midpoint)
    cyl.compute_vertex_normals()
    return cyl

def random_filament(start, segments=30, length_mu=0.5, length_sigma=0.1, persist=0.8):
    """
    Generate a single filament as a random walk with persistence.
    Returns list of (p0, p1) segment endpoints.
    """
    pts = [np.array(start, dtype=float)]
    # initial direction random
    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)

    for _ in range(segments):
        # jiggle the direction a bit
        noise = np.random.normal(scale=1.0, size=3)
        direction = persist * direction + (1-persist) * (noise/np.linalg.norm(noise))
        direction /= np.linalg.norm(direction)

        L = max(1e-3, np.random.normal(loc=length_mu, scale=length_sigma))
        pts.append(pts[-1] + direction * L)

    # return segment pairs
    return list(zip(pts[:-1], pts[1:]))

def build_mito_network(
    num_filaments=8,
    segments_per=25,
    radius=0.08,
    **filament_kwargs
):
    """
    Build and return one Open3D mesh containing all filaments.
    """
    all_mesh = None
    # scatter their starting points in a small sphere
    for i in range(num_filaments):
        start_pt = np.random.normal(scale=2.0, size=3)
        segs = random_filament(start_pt, segments=segments_per, **filament_kwargs)
        for p0, p1 in segs:
            cyl = cylinder_between(p0, p1, radius=radius)
            if cyl is None:
                continue
            if all_mesh is None:
                all_mesh = cyl
            else:
                all_mesh += cyl
    return all_mesh

if __name__ == "__main__":
    # tweak these parameters as you like:
    mesh = build_mito_network(
        num_filaments=12,
        segments_per=30,
        length_mu=0.6,
        length_sigma=0.1,
        persist=0.85,
        radius=0.05,
    )
    o3d.visualization.draw_geometries([mesh], window_name="Mito Colony")

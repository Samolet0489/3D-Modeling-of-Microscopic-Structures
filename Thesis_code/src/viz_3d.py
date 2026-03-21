import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .shapes import generate_mito_volume, generate_mito_volume_polyline


def _sample_points(points, max_points=20000, seed=0):
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def main(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    seed=42,
    max_points=20000,
    model="A",
    n_seeds=4,
    step_length=1.5,
    n_steps=120,
    branch_prob=0.08,
    branch_steps=40,
    radius_min=1,
    radius_max=4,
    persistence=0.85,
    z_scale=0.35,
    seed_spread=20,
):
    # choose generator according to requested model
    if model == "A":
        vol, mask = generate_mito_volume(shape=shape, n_tubes=n_tubes, tube_radius=tube_radius, seed=seed)
    elif model == "B":
        vol, mask = generate_mito_volume_polyline(
            shape=shape,
            n_seeds=n_seeds,
            step_length=step_length,
            n_steps=n_steps,
            branch_prob=branch_prob,
            branch_steps=branch_steps,
            radius_min=radius_min,
            radius_max=radius_max,
            persistence=persistence,
            z_scale=z_scale,
            seed_spread=seed_spread,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Must be 'A' or 'B'.")
    mask = mask.astype(bool)

    # 1) Orthogonal projections
    xy = mask.max(axis=0)      # (Y, X)
    xz = mask.max(axis=1)      # (Z, X)
    yz = mask.max(axis=2)      # (Z, Y)

    # 2) Montage of z-slices
    z_slices = np.linspace(0, shape[0] - 1, num=min(8, shape[0]), dtype=int)

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(xy, cmap="gray")
    ax1.set_title("Mask projection XY (max over Z)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(xz, cmap="gray")
    ax2.set_title("Mask projection XZ (max over Y)")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(yz, cmap="gray")
    ax3.set_title("Mask projection YZ (max over X)")
    ax3.axis("off")

    # montage
    ax4 = fig.add_subplot(2, 3, 4)
    montage = np.concatenate([mask[z].astype(np.uint8) for z in z_slices], axis=1)
    ax4.imshow(montage, cmap="gray")
    ax4.set_title(f"Z-slice montage (z={list(z_slices)})")
    ax4.axis("off")

    # 3D scatter of mask voxels
    pts = np.argwhere(mask)  # (N, 3) with columns [z, y, x]
    pts = _sample_points(pts, max_points=max_points, seed=seed)

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    ax5.scatter(pts[:, 2], pts[:, 1], pts[:, 0], s=1)  # x, y, z
    ax5.set_title("3D voxel scatter (downsampled)")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")

    # simple info panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(
        0.0, 0.9,
        f"Volume shape: {vol.shape}\n"
        f"Mask voxels: {int(mask.sum())}\n"
        f"n_tubes: {n_tubes}\n"
        f"tube_radius: {tube_radius}\n"
        f"seed: {seed}",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the 3D synthetic mask/volume.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["A", "B"], default="A",
                        help="Generator model: A=tubes, B=polyline")
    parser.add_argument("--n-tubes", type=int, default=5)
    parser.add_argument("--tube-radius", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--n-seeds", type=int, default=4,
                        help="(Model B) number of initial seed points")
    parser.add_argument("--step-length", type=float, default=1.5,
                        help="(Model B) step length")
    parser.add_argument("--n-steps", type=int, default=120,
                        help="(Model B) steps per main walker")
    parser.add_argument("--branch-prob", type=float, default=0.08,
                        help="(Model B) probability of branching")
    parser.add_argument("--branch-steps", type=int, default=40,
                        help="(Model B) length of spawned branches")
    parser.add_argument("--radius-min", type=int, default=1,
                        help="(Model B) minimum radius")
    parser.add_argument("--radius-max", type=int, default=4,
                        help="(Model B) maximum radius")
    parser.add_argument("--persistence", type=float, default=0.85,
                        help="(Model B) direction persistence")
    parser.add_argument("--z-scale", type=float, default=0.35,
                        help="(Model B) z-plane bias")
    parser.add_argument("--seed-spread", type=float, default=20,
                        help="(Model B) colony seed spread")
    args = parser.parse_args()

    main(
        seed=args.seed,
        n_tubes=args.n_tubes,
        tube_radius=args.tube_radius,
        max_points=args.max_points,
        model=args.model,
        n_seeds=args.n_seeds,
        step_length=args.step_length,
        n_steps=args.n_steps,
        branch_prob=args.branch_prob,
        branch_steps=args.branch_steps,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        persistence=args.persistence,
        z_scale=args.z_scale,
        seed_spread=args.seed_spread,
    )
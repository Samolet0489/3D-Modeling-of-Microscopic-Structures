"""Demo script for Phase 1 of the synthetic microscopy pipeline.

This module can be executed either directly (when running inside the ``src``
directory) or via ``python -m src.demo_pipeline`` from the project root.
It creates a single synthetic sample and displays five panels:

  * a 2-D slice / projected view from the 3-D volume
  * the raw projected geometry mask
  * the refined visible mask
  * the clean (blurred) image
  * the noisy image
"""

import matplotlib.pyplot as plt

from .dataset_generator import generate_single_example, _mask_to_2d


def main(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    projection="focus_max",
    slab_thickness=5,
    mask_low_thr=0.13,
    mask_high_thr=0.27,
    mask_min_size=10,
    seed_erode_iters=1,
    mask_gamma=1.35,
    max_hole_size=3,
    model="A",
    n_seeds=4,
    step_length=1.5,
    n_steps=120,
    branch_prob=0.08,
    branch_steps=40,
    radius_min=1,
    radius_max=3,
    persistence=0.85,
    z_scale=0.35,
    seed_spread=20,
    seed=42,
    noise_seed=123,
):
    vol, raw_mask, psf, clean, noisy, visible_mask = generate_single_example(
        shape=shape,
        n_tubes=n_tubes,
        tube_radius=tube_radius,
        psf_size=psf_size,
        psf_sigma_xy=psf_sigma_xy,
        psf_sigma_z=psf_sigma_z,
        photon_level=photon_level,
        read_noise_sigma=read_noise_sigma,
        projection=projection,
        slab_thickness=slab_thickness,
        mask_low_thr=mask_low_thr,
        mask_high_thr=mask_high_thr,
        mask_min_size=mask_min_size,
        seed_erode_iters=seed_erode_iters,
        mask_gamma=mask_gamma,
        max_hole_size=max_hole_size,
        model=model,
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

    if projection == "center_slice":
        z_mid = vol.shape[0] // 2
        vol_slice = vol[z_mid]
    else:
        vol_slice = _mask_to_2d(vol > 0, projection, slab_thickness=slab_thickness)

    raw_mask_slice = _mask_to_2d(raw_mask, projection, slab_thickness=slab_thickness)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title(f"Volume ({projection})")

    axes[1].imshow(raw_mask_slice, cmap="gray")
    axes[1].set_title("Raw mask")

    axes[2].imshow(visible_mask, cmap="gray")
    axes[2].set_title("Visible mask")

    axes[3].imshow(clean, cmap="gray")
    axes[3].set_title("Clean image")

    axes[4].imshow(noisy, cmap="gray")
    axes[4].set_title("Noisy image")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run demo pipeline and show example outputs.")
    parser.add_argument(
        "--model", choices=["A", "B"], default="A",
        help="Model A (baseline tubes) or Model B (random-walk polylines)"
    )
    parser.add_argument(
        "--projection", choices=["center_slice", "max", "sum", "focus_max"], default="focus_max",
        help="Projection method for 3D→2D conversion"
    )
    parser.add_argument("--slab-thickness", type=int, default=5, help="Z slab thickness for focus_max projection")
    parser.add_argument("--mask-low-thr", type=float, default=0.13, help="Low threshold for visible mask creation")
    parser.add_argument("--mask-high-thr", type=float, default=0.27, help="High threshold for visible mask creation")
    parser.add_argument("--mask-min-size", type=int, default=10, help="Minimum connected component size in mask")
    parser.add_argument("--seed-erode-iters", type=int, default=1, help="Light erosion applied only to seed support")
    parser.add_argument("--mask-gamma", type=float, default=1.35, help="Gamma used to suppress fuzzy halos in clean image")
    parser.add_argument("--max-hole-size", type=int, default=3, help="Fill only holes up to this size inside masks")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    main(
        projection=args.projection,
        slab_thickness=args.slab_thickness,
        mask_low_thr=args.mask_low_thr,
        mask_high_thr=args.mask_high_thr,
        mask_min_size=args.mask_min_size,
        seed_erode_iters=args.seed_erode_iters,
        mask_gamma=args.mask_gamma,
        max_hole_size=args.max_hole_size,
        model=args.model,
        seed=args.seed,
    )

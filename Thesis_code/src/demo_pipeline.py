"""Demo script for Phase 1 of the synthetic microscopy pipeline."""

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
    apply_gaussian_noise=True,
    gaussian_std=12.0,
    background_strength=0.08,
    band_strength=0.025,
    gradient_strength=0.03,
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
        apply_gaussian_noise=apply_gaussian_noise,
        gaussian_std=gaussian_std,
        background_strength=background_strength,
        band_strength=band_strength,
        gradient_strength=gradient_strength,
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

    raw_mask_slice = _mask_to_2d(raw_mask, projection, slab_thickness=slab_thickness)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(raw_mask_slice, cmap="gray")
    axes[0].set_title("Raw mask")
    axes[1].imshow(visible_mask, cmap="gray")
    axes[1].set_title("Visible mask")
    axes[2].imshow(clean, cmap="gray")
    axes[2].set_title("Clean image")
    axes[3].imshow(noisy, cmap="gray")
    axes[3].set_title("Noisy image")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run demo pipeline and show example outputs.")
    parser.add_argument("--model", choices=["A", "B"], default="A")
    parser.add_argument("--projection", choices=["center_slice", "max", "sum", "focus_max"], default="focus_max")
    parser.add_argument("--slab-thickness", type=int, default=5)
    parser.add_argument("--mask-low-thr", type=float, default=0.13)
    parser.add_argument("--mask-high-thr", type=float, default=0.27)
    parser.add_argument("--mask-min-size", type=int, default=10)
    parser.add_argument("--seed-erode-iters", type=int, default=1)
    parser.add_argument("--mask-gamma", type=float, default=1.35)
    parser.add_argument("--max-hole-size", type=int, default=3)
    parser.add_argument("--apply-gaussian-noise", action="store_true")
    parser.add_argument("--gaussian-std", type=float, default=12.0)
    parser.add_argument("--background-strength", type=float, default=0.08)
    parser.add_argument("--band-strength", type=float, default=0.025)
    parser.add_argument("--gradient-strength", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
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
        apply_gaussian_noise=args.apply_gaussian_noise,
        gaussian_std=args.gaussian_std,
        background_strength=args.background_strength,
        band_strength=args.band_strength,
        gradient_strength=args.gradient_strength,
        model=args.model,
        seed=args.seed,
    )

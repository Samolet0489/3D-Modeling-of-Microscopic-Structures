"""Dataset generation utilities for Phase 1 (2.5D) of the project.

Functions in this module are written to be reusable in downstream training
scripts. A simple command-line interface is provided for producing a folder
of image/mask pairs suitable for 2D segmentation.
"""

import os
import json
import argparse

import numpy as np
from imageio import imwrite
from scipy.ndimage import binary_propagation, label

from .shapes import generate_mito_volume, generate_mito_volume_polyline
from .psf import gaussian_psf_kernel
from .render import render_image
from .noise import add_noise


def _mask_to_2d(mask, projection, slab_thickness=7):
    if projection == "center_slice":
        z_mid = mask.shape[0] // 2
        return mask[z_mid]

    if projection == "max":
        return mask.max(axis=0)

    if projection == "sum":
        return mask.sum(axis=0) > 0

    if projection == "focus_max":
        slab_thickness = max(1, int(slab_thickness))
        z_mid = mask.shape[0] // 2
        half = slab_thickness // 2
        z0 = max(0, z_mid - half)
        z1 = min(mask.shape[0], z_mid + half + 1)
        return mask[z0:z1].max(axis=0)

    raise ValueError(f"Unknown projection: {projection}")


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _build_fixed_volume_params(
    model,
    n_tubes,
    tube_radius,
    n_seeds,
    step_length,
    n_steps,
    branch_prob,
    branch_steps,
    radius_min,
    radius_max,
    persistence,
    z_scale,
    seed_spread,
):
    if model == "A":
        return {
            "n_tubes": int(n_tubes),
            "tube_radius": int(np.ceil(tube_radius)),
        }

    return {
        "n_seeds": int(n_seeds),
        "step_length": float(step_length),
        "n_steps": int(n_steps),
        "branch_prob": float(branch_prob),
        "branch_steps": int(branch_steps),
        "radius_min": int(np.ceil(radius_min)),
        "radius_max": int(np.ceil(radius_max)),
        "persistence": float(persistence),
        "z_scale": float(z_scale),
        "seed_spread": float(seed_spread),
    }


def _sample_randomized_volume_params(model, rng):
    if model == "A":
        return {
            "n_tubes": int(rng.integers(3, 16)),
            "tube_radius": int(rng.integers(1, 5)),
        }

    return {
        "n_seeds": int(rng.integers(2, 8)),
        "step_length": float(rng.uniform(0.8, 2.0)),
        "n_steps": int(rng.integers(80, 141)),
        "branch_prob": float(rng.uniform(0.04, 0.12)),
        "branch_steps": int(rng.integers(25, 61)),
        "radius_min": 1,
        "radius_max": 4,
        "persistence": float(rng.uniform(0.82, 0.95)),
        "z_scale": float(rng.uniform(0.20, 0.45)),
        "seed_spread": float(rng.uniform(10, 28)),
    }


def _build_fixed_imaging_params(psf_sigma_xy, psf_sigma_z, photon_level, read_noise_sigma):
    return {
        "psf_sigma_xy": float(psf_sigma_xy),
        "psf_sigma_z": float(psf_sigma_z),
        "photon_level": float(photon_level),
        "read_noise_sigma": float(read_noise_sigma),
    }


def _sample_randomized_imaging_params(rng):
    return {
        "psf_sigma_xy": float(rng.uniform(0.7, 2.0)),
        "psf_sigma_z": float(rng.uniform(1.0, 3.5)),
        "photon_level": float(rng.uniform(2000.0, 5000.0)),
        "read_noise_sigma": float(rng.uniform(0.01, 0.08)),
    }


def _make_border_mask(shape_2d, border_width=2):
    h, w = shape_2d
    bw = max(1, int(border_width))

    border = np.zeros((h, w), dtype=bool)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    return border


def _remove_small_components(mask2d, min_size=12):
    labels, n = label(mask2d.astype(np.uint8))
    if n == 0:
        return mask2d.astype(bool)

    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[labels]


def _visible_mask_from_clean(clean, raw_mask2d, low_thr=0.12, high_thr=0.24, min_size=12):
    """
    Build a visible 2D mask from the clean projected image, but constrain it
    to the raw projected geometry.

    This is intentionally moderate:
    - no aggressive erosion
    - no aggressive morphology
    - strong/weak thresholding only inside raw support
    """
    img = clean.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    raw = raw_mask2d.astype(bool)

    strong = (img >= float(high_thr)) & raw
    weak = (img >= float(low_thr)) & raw

    if strong.sum() == 0:
        mask = weak
    else:
        mask = binary_propagation(strong, mask=weak)

    mask = _remove_small_components(mask, min_size=min_size)
    return mask.astype(bool)


def _evaluate_mask_qc(
    mask2d,
    min_fg_fraction=0.01,
    max_fg_fraction=0.45,
    max_border_fraction=0.18,
    border_width=2,
):
    """
    Evaluate whether a 2D mask is acceptable for saving.

    Checks:
    - not too empty
    - not too full
    - not too dominated by border-touching foreground
    """
    mask_bool = mask2d.astype(bool)

    fg_pixels = int(mask_bool.sum())
    fg_fraction = float(mask_bool.mean())

    if fg_pixels == 0:
        border_pixels = 0
        border_fraction = 1.0
    else:
        border_mask = _make_border_mask(mask_bool.shape, border_width=border_width)
        border_pixels = int(np.logical_and(mask_bool, border_mask).sum())
        border_fraction = float(border_pixels / max(fg_pixels, 1))

    reasons = []

    if fg_fraction < float(min_fg_fraction):
        reasons.append("too_empty")

    if max_fg_fraction is not None and fg_fraction > float(max_fg_fraction):
        reasons.append("too_full")

    if border_fraction > float(max_border_fraction):
        reasons.append("too_border_heavy")

    metrics = {
        "fg_pixels": fg_pixels,
        "fg_fraction": fg_fraction,
        "border_pixels": border_pixels,
        "border_fraction": border_fraction,
        "accepted": len(reasons) == 0,
        "reasons": reasons,
    }
    return metrics


def generate_single_example(
    shape=(32, 128, 128),
    n_tubes=5,
    tube_radius=2,
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    projection="focus_max",
    slab_thickness=7,
    mask_low_thr=0.12,
    mask_high_thr=0.24,
    mask_min_size=12,
    model="A",
    n_seeds=4,
    step_length=1.0,
    n_steps=80,
    branch_prob=0.08,
    branch_steps=40,
    radius_min=1,
    radius_max=4,
    persistence=0.85,
    z_scale=0.35,
    seed_spread=20,
    seed=None,
):
    """Create one synthetic example using either Model A or Model B."""
    if model == "A":
        volume, mask = generate_mito_volume(
            shape=shape,
            n_tubes=n_tubes,
            tube_radius=tube_radius,
            seed=seed,
        )
    elif model == "B":
        volume, mask = generate_mito_volume_polyline(
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

    psf = gaussian_psf_kernel(size=psf_size, sigma_xy=psf_sigma_xy, sigma_z=psf_sigma_z)
    clean = render_image(volume, psf, projection=projection, slab_thickness=slab_thickness)

    clean = clean - clean.min()
    if clean.max() > 0:
        clean = clean / clean.max()

    raw_mask2d = _mask_to_2d(mask, projection, slab_thickness=slab_thickness)
    mask2d = _visible_mask_from_clean(
        clean,
        raw_mask2d,
        low_thr=mask_low_thr,
        high_thr=mask_high_thr,
        min_size=mask_min_size,
    )

    noisy = add_noise(
        clean,
        photon_level=photon_level,
        read_noise_sigma=read_noise_sigma,
        seed=seed,
    )
    return volume, mask, psf, clean, noisy, mask2d


def generate_and_save_dataset(
    out_dir,
    n_samples=100,
    shape=(32, 128, 128),
    psf_size=(9, 9, 9),
    psf_sigma_xy=1.0,
    psf_sigma_z=2.0,
    photon_level=2000.0,
    read_noise_sigma=0.03,
    projection="focus_max",
    slab_thickness=7,
    mask_low_thr=0.12,
    mask_high_thr=0.24,
    mask_min_size=12,
    model="A",
    n_tubes=5,
    tube_radius=2,
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
    randomize_per_sample=False,
    min_fg_fraction=0.01,
    max_fg_fraction=0.45,
    max_border_fraction=0.18,
    border_width=2,
    max_attempts_per_sample=30,
    seed=None,
):
    """
    Create n_samples examples and save them to out_dir.

    Images are stored as PNG files under out_dir/images and masks under out_dir/masks.
    File names are zero-padded like 000000.png.
    """
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    if max_fg_fraction is not None and max_fg_fraction <= 0:
        max_fg_fraction = None

    rng = np.random.default_rng(seed)

    run_config = {
        "out_dir": out_dir,
        "n_samples": int(n_samples),
        "shape": list(shape),
        "psf_size": list(psf_size),
        "projection": projection,
        "slab_thickness": int(slab_thickness),
        "mask_low_thr": float(mask_low_thr),
        "mask_high_thr": float(mask_high_thr),
        "mask_min_size": int(mask_min_size),
        "model": model,
        "randomize_per_sample": bool(randomize_per_sample),
        "seed": seed,
        "fixed_volume_params": _build_fixed_volume_params(
            model=model,
            n_tubes=n_tubes,
            tube_radius=tube_radius,
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
        ),
        "fixed_imaging_params": _build_fixed_imaging_params(
            psf_sigma_xy=psf_sigma_xy,
            psf_sigma_z=psf_sigma_z,
            photon_level=photon_level,
            read_noise_sigma=read_noise_sigma,
        ),
        "qc": {
            "min_fg_fraction": float(min_fg_fraction),
            "max_fg_fraction": None if max_fg_fraction is None else float(max_fg_fraction),
            "max_border_fraction": float(max_border_fraction),
            "border_width": int(border_width),
            "max_attempts_per_sample": int(max_attempts_per_sample),
        },
    }

    _write_json(os.path.join(out_dir, "run_config.json"), run_config)

    manifest_path = os.path.join(out_dir, "sample_manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    print("Run config:")
    print(json.dumps(run_config, indent=2))
    print(f"Generating {n_samples} samples using model={model} ...")

    fixed_volume_params = run_config["fixed_volume_params"]
    fixed_imaging_params = run_config["fixed_imaging_params"]

    for idx in range(n_samples):
        accepted = False
        last_qc = None
        last_seed = None
        last_volume_params = None
        last_imaging_params = None
        last_noisy = None
        last_mask2d = None

        for attempt in range(1, max_attempts_per_sample + 1):
            sample_seed = int(rng.integers(0, 2**31 - 1))

            if randomize_per_sample:
                volume_params = _sample_randomized_volume_params(model, rng)
                imaging_params = _sample_randomized_imaging_params(rng)
            else:
                volume_params = dict(fixed_volume_params)
                imaging_params = dict(fixed_imaging_params)

            vol, mask, psf, clean, noisy, mask2d = generate_single_example(
                shape=shape,
                psf_size=psf_size,
                projection=projection,
                slab_thickness=slab_thickness,
                mask_low_thr=mask_low_thr,
                mask_high_thr=mask_high_thr,
                mask_min_size=mask_min_size,
                model=model,
                seed=sample_seed,
                **volume_params,
                **imaging_params,
            )

            qc = _evaluate_mask_qc(
                mask2d=mask2d,
                min_fg_fraction=min_fg_fraction,
                max_fg_fraction=max_fg_fraction,
                max_border_fraction=max_border_fraction,
                border_width=border_width,
            )

            print(
                f"sample {idx:06d} | attempt {attempt:02d} | "
                f"seed={sample_seed} | "
                f"fg={qc['fg_fraction']:.4f} | border={qc['border_fraction']:.4f} | "
                f"accepted={qc['accepted']}"
            )

            if qc["accepted"]:
                accepted = True
                last_qc = qc
                last_seed = sample_seed
                last_volume_params = volume_params
                last_imaging_params = imaging_params
                last_noisy = noisy
                last_mask2d = mask2d
                break

            last_qc = qc
            last_seed = sample_seed
            last_volume_params = volume_params
            last_imaging_params = imaging_params
            last_noisy = noisy
            last_mask2d = mask2d

        if not accepted:
            raise RuntimeError(
                f"Could not generate acceptable sample {idx:06d} after "
                f"{max_attempts_per_sample} attempts. Last QC: {last_qc}"
            )

        img_path = os.path.join(images_dir, f"{idx:06d}.png")
        mask_path = os.path.join(masks_dir, f"{idx:06d}.png")

        imwrite(img_path, (last_noisy * 255).astype(np.uint8))
        imwrite(mask_path, (last_mask2d.astype(np.uint8) * 255))

        record = {
            "idx": idx,
            "seed": last_seed,
            "model": model,
            "projection": projection,
            "slab_thickness": int(slab_thickness),
            "mask_low_thr": float(mask_low_thr),
            "mask_high_thr": float(mask_high_thr),
            "mask_min_size": int(mask_min_size),
            "image_file": os.path.relpath(img_path, out_dir),
            "mask_file": os.path.relpath(mask_path, out_dir),
            "volume_params": last_volume_params,
            "imaging_params": last_imaging_params,
            "qc": last_qc,
        }
        _append_jsonl(manifest_path, record)

    print(f"Saved dataset to: {out_dir}")
    print(f"Saved run config to: {os.path.join(out_dir, 'run_config.json')}")
    print(f"Saved sample manifest to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic microscopy dataset.")
    parser.add_argument("out_dir", help="Output directory for images/masks")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to create")
    parser.add_argument(
        "--model",
        choices=["A", "B"],
        default="A",
        help="Model A (baseline tubes) or Model B (random-walk polylines)",
    )
    parser.add_argument(
        "--projection",
        choices=["center_slice", "max", "sum", "focus_max"],
        default="focus_max",
        help="Projection method for 3D→2D conversion",
    )
    parser.add_argument("--slab-thickness", type=int, default=7, help="Z slab thickness for focus_max projection")
    parser.add_argument("--mask-low-thr", type=float, default=0.12, help="Low threshold for visible mask creation")
    parser.add_argument("--mask-high-thr", type=float, default=0.24, help="High threshold for visible mask creation")
    parser.add_argument("--mask-min-size", type=int, default=12, help="Minimum connected component size in mask")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # shared / imaging parameters
    parser.add_argument("--psf-sigma-xy", type=float, default=1.0, help="PSF sigma in XY")
    parser.add_argument("--psf-sigma-z", type=float, default=2.0, help="PSF sigma in Z")
    parser.add_argument("--photon-level", type=float, default=2000.0, help="Photon count scaling")
    parser.add_argument("--read-noise-sigma", type=float, default=0.03, help="Gaussian read noise sigma")

    # model A params
    parser.add_argument("--n-tubes", type=int, default=5, help="(Model A) number of tubes")
    parser.add_argument("--tube-radius", type=int, default=2, help="(Model A) tube radius")

    # model B params
    parser.add_argument("--n-seeds", type=int, default=4, help="(Model B) number of initial seed points")
    parser.add_argument("--step-length", type=float, default=1.5, help="(Model B) step length")
    parser.add_argument("--n-steps", type=int, default=120, help="(Model B) steps per main walker")
    parser.add_argument("--branch-prob", type=float, default=0.08, help="(Model B) probability of branching")
    parser.add_argument("--branch-steps", type=int, default=40, help="(Model B) length of spawned branches")
    parser.add_argument("--radius-min", type=int, default=1, help="(Model B) minimum radius")
    parser.add_argument("--radius-max", type=int, default=4, help="(Model B) maximum radius")
    parser.add_argument("--persistence", type=float, default=0.85, help="(Model B) direction persistence")
    parser.add_argument("--z-scale", type=float, default=0.35, help="(Model B) z-motion scaling")
    parser.add_argument("--seed-spread", type=float, default=20.0, help="(Model B) colony center seed spread")

    # behavior
    parser.add_argument(
        "--randomize-per-sample",
        action="store_true",
        help="If set, randomize structure + imaging parameters per sample and record them in sample_manifest.jsonl",
    )

    # QC gate
    parser.add_argument("--min-fg-fraction", type=float, default=0.01, help="Reject masks that are too empty")
    parser.add_argument(
        "--max-fg-fraction",
        type=float,
        default=0.45,
        help="Reject masks that are too full; use <= 0 to disable",
    )
    parser.add_argument(
        "--max-border-fraction",
        type=float,
        default=0.18,
        help="Reject masks with too much foreground on the outer image border",
    )
    parser.add_argument(
        "--border-width",
        type=int,
        default=2,
        help="Border thickness in pixels used for the border-touch check",
    )
    parser.add_argument(
        "--max-attempts-per-sample",
        type=int,
        default=30,
        help="Maximum number of regeneration attempts before failing",
    )

    args = parser.parse_args()

    generate_and_save_dataset(
        out_dir=args.out_dir,
        n_samples=args.n,
        psf_sigma_xy=args.psf_sigma_xy,
        psf_sigma_z=args.psf_sigma_z,
        photon_level=args.photon_level,
        read_noise_sigma=args.read_noise_sigma,
        projection=args.projection,
        slab_thickness=args.slab_thickness,
        mask_low_thr=args.mask_low_thr,
        mask_high_thr=args.mask_high_thr,
        mask_min_size=args.mask_min_size,
        model=args.model,
        n_tubes=args.n_tubes,
        tube_radius=args.tube_radius,
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
        randomize_per_sample=args.randomize_per_sample,
        min_fg_fraction=args.min_fg_fraction,
        max_fg_fraction=args.max_fg_fraction,
        max_border_fraction=args.max_border_fraction,
        border_width=args.border_width,
        max_attempts_per_sample=args.max_attempts_per_sample,
        seed=args.seed,
    )
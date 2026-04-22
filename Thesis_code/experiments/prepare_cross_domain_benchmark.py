from pathlib import Path
import json
import random

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_image_files(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def pair_by_name(image_dir: Path, mask_dir: Path, label: str = ""):
    images = {p.name: p for p in list_image_files(image_dir)}
    masks = {p.name: p for p in list_image_files(mask_dir)}

    only_images = sorted(set(images) - set(masks))
    only_masks = sorted(set(masks) - set(images))
    common_names = sorted(set(images) & set(masks))

    prefix = f"[{label}] " if label else ""

    if only_images or only_masks:
        print(f"\n{prefix}Mismatch between images and masks. Using only matching filename pairs.")
        if only_images:
            print(f"  Present only in images ({len(only_images)}):")
            for name in only_images[:10]:
                print(f"    {name}")
            if len(only_images) > 10:
                print("    ...")
        if only_masks:
            print(f"  Present only in masks ({len(only_masks)}):")
            for name in only_masks[:10]:
                print(f"    {name}")
            if len(only_masks) > 10:
                print("    ...")

    if not common_names:
        raise RuntimeError(f"{prefix}No matching image/mask filename pairs found.")

    pairs = [{"name": name, "image": str(images[name]), "mask": str(masks[name])} for name in common_names]
    return pairs


def proportional_split_counts(total_target: int, original_counts: dict):
    """
    Scale original split counts proportionally so they sum exactly to total_target.
    """
    original_total = sum(original_counts.values())
    raw = {
        split: (count / original_total) * total_target
        for split, count in original_counts.items()
    }

    floored = {split: int(value) for split, value in raw.items()}
    remainder = total_target - sum(floored.values())

    fractions = sorted(
        ((split, raw[split] - floored[split]) for split in raw),
        key=lambda x: x[1],
        reverse=True
    )

    for i in range(remainder):
        split = fractions[i][0]
        floored[split] += 1

    return floored


def sample_pairs(pairs, n, rng):
    if n > len(pairs):
        raise ValueError(f"Cannot sample {n} items from only {len(pairs)} available pairs.")
    sampled = rng.sample(pairs, n)
    sampled = sorted(sampled, key=lambda x: x["name"])
    return sampled


def main():
    root = Path(__file__).resolve().parents[1]

    professor_candidates = [
        root / "data" / "FromProffessor",
        root / "data" / "FromProfessor",
        root / "FromProffessor",
        root / "FromProfessor",
    ]

    professor_root = None
    for candidate in professor_candidates:
        if candidate.exists():
            professor_root = candidate
            break

    if professor_root is None:
        raise FileNotFoundError(
            "Could not find the professor dataset. Checked these locations:\n"
            + "\n".join(str(p) for p in professor_candidates)
        )

    a1_img_root = professor_root / "image"
    a1_mask_root = professor_root / "segment"

    a2_img_dir = root / "data" / "datasets" / "20260422_synthetic_B_main" / "images"
    a2_mask_dir = root / "data" / "datasets" / "20260422_synthetic_B_main" / "masks"

    rng = random.Random(42)

    print("\n=== Checking Approach 1 (Professor dataset) ===")
    print(f"Using professor dataset root: {professor_root}")

    a1_full = {}
    for split in ["train", "val", "test"]:
        img_dir = a1_img_root / split
        mask_dir = a1_mask_root / split
        pairs = pair_by_name(img_dir, mask_dir, label=f"Approach1/{split}")
        a1_full[split] = pairs
        print(f"{split:>5}: {len(pairs)} paired files")

    print("\n=== Checking Approach 2 (Your dataset) ===")
    a2_all = pair_by_name(a2_img_dir, a2_mask_dir, label="Approach2/all")
    print(f"total: {len(a2_all)} paired files")

    original_counts = {
        "train": len(a1_full["train"]),
        "val": len(a1_full["val"]),
        "test": len(a1_full["test"]),
    }

    benchmark_total = len(a2_all)
    benchmark_counts = proportional_split_counts(benchmark_total, original_counts)

    print("\n=== Creating matched benchmark ===")
    print("Original professor split sizes:")
    print(f"train: {original_counts['train']}")
    print(f"val:   {original_counts['val']}")
    print(f"test:  {original_counts['test']}")

    print("\nMatched benchmark split sizes (scaled to Approach 2 total):")
    print(f"train: {benchmark_counts['train']}")
    print(f"val:   {benchmark_counts['val']}")
    print(f"test:  {benchmark_counts['test']}")
    print(f"total: {sum(benchmark_counts.values())}")

    a1_subset = {
        "train": sample_pairs(a1_full["train"], benchmark_counts["train"], rng),
        "val": sample_pairs(a1_full["val"], benchmark_counts["val"], rng),
        "test": sample_pairs(a1_full["test"], benchmark_counts["test"], rng),
    }

    shuffled_a2 = a2_all[:]
    rng.shuffle(shuffled_a2)

    n_train = benchmark_counts["train"]
    n_val = benchmark_counts["val"]
    n_test = benchmark_counts["test"]

    a2_subset = {
        "train": sorted(shuffled_a2[:n_train], key=lambda x: x["name"]),
        "val": sorted(shuffled_a2[n_train:n_train + n_val], key=lambda x: x["name"]),
        "test": sorted(shuffled_a2[n_train + n_val:n_train + n_val + n_test], key=lambda x: x["name"]),
    }

    manifest = {
        "description": "Cross-domain benchmark manifest using a downsampled professor dataset matched to the total size of Approach 2.",
        "seed": 42,
        "benchmark_counts": benchmark_counts,
        "approach1": {
            "name": "Professor dataset / Approach 1",
            "root": str(professor_root),
            "splits": a1_subset,
        },
        "approach2": {
            "name": "Our synthetic dataset / Approach 2",
            "root": str(root / "data" / "datasets" / "20260422_synthetic_B_main"),
            "splits": a2_subset,
        },
    }

    out_path = root / "experiments" / "cross_domain_manifest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Done ===")
    print(f"Manifest written to:\n{out_path}")

    print("\nFinal benchmark counts:")
    print(f"Approach 1 train: {len(a1_subset['train'])}")
    print(f"Approach 1 val:   {len(a1_subset['val'])}")
    print(f"Approach 1 test:  {len(a1_subset['test'])}")
    print(f"Approach 2 train: {len(a2_subset['train'])}")
    print(f"Approach 2 val:   {len(a2_subset['val'])}")
    print(f"Approach 2 test:  {len(a2_subset['test'])}")


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import os
import shutil

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def safe_remove_tree(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)


def ensure_dir(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    """
    Try hard-link first.
    If that fails, fall back to copying.
    """
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        return "linked"
    except Exception:
        shutil.copy2(src, dst)
        return "copied"


def stage_split_entries(split_entries, out_images_dir: Path, out_masks_dir: Path):
    ensure_dir(out_images_dir)
    ensure_dir(out_masks_dir)

    linked = 0
    copied = 0

    for item in split_entries:
        img_src = Path(item["image"])
        mask_src = Path(item["mask"])
        name = item["name"]

        if not img_src.exists():
            raise FileNotFoundError(f"Missing image source: {img_src}")
        if not mask_src.exists():
            raise FileNotFoundError(f"Missing mask source: {mask_src}")

        img_dst = out_images_dir / name
        mask_dst = out_masks_dir / name

        img_mode = link_or_copy(img_src, img_dst)
        mask_mode = link_or_copy(mask_src, mask_dst)

        if img_mode == "linked":
            linked += 1
        else:
            copied += 1

        if mask_mode == "linked":
            linked += 1
        else:
            copied += 1

    return linked, copied


def count_files(folder: Path):
    if not folder.exists():
        return 0
    return len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "experiments" / "cross_domain_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_root = root / "experiments" / "cross_domain_data"

    print("\n=== Staging cross-domain benchmark ===")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {out_root}")

    safe_remove_tree(out_root)

    summary = {
        "manifest": str(manifest_path),
        "output_root": str(out_root),
        "seed": manifest.get("seed"),
        "benchmark_counts": manifest.get("benchmark_counts", {}),
        "approaches": {}
    }

    for approach_key in ["approach1", "approach2"]:
        approach = manifest[approach_key]
        approach_name = approach["name"]
        approach_out = out_root / approach_key

        print(f"\n--- {approach_name} ---")
        print(f"Folder: {approach_out}")

        summary["approaches"][approach_key] = {
            "name": approach_name,
            "root": str(approach_out),
            "splits": {}
        }

        total_linked = 0
        total_copied = 0

        for split in ["train", "val", "test"]:
            split_entries = approach["splits"][split]

            out_images_dir = approach_out / split / "images"
            out_masks_dir = approach_out / split / "masks"

            linked, copied = stage_split_entries(split_entries, out_images_dir, out_masks_dir)
            total_linked += linked
            total_copied += copied

            n_images = count_files(out_images_dir)
            n_masks = count_files(out_masks_dir)

            print(f"{split:>5}: images={n_images}, masks={n_masks}")

            if n_images != len(split_entries) or n_masks != len(split_entries):
                raise RuntimeError(
                    f"Split {approach_key}/{split} has wrong file count. "
                    f"Expected {len(split_entries)}, got images={n_images}, masks={n_masks}"
                )

            summary["approaches"][approach_key]["splits"][split] = {
                "count": len(split_entries),
                "images_dir": str(out_images_dir),
                "masks_dir": str(out_masks_dir),
            }

        print(f"linked files: {total_linked}")
        print(f"copied files: {total_copied}")

    summary_path = out_root / "staging_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Done ===")
    print(f"Staging summary written to:\n{summary_path}")
    print("\nYou can now inspect:")
    print(r"experiments\cross_domain_data\approach1")
    print(r"experiments\cross_domain_data\approach2")


if __name__ == "__main__":
    main()
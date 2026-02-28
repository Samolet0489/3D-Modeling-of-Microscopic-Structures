"""Quick visualization of generated dataset.

This module loads random image/mask pairs from a generated dataset folder
and plots them in a grid with mask overlay for sanity checking.

Usage:
    python -m src.check_dataset

The script looks for a default folder ``./data/synthetic_2d`` from the
current working directory. If not found or if empty, the script prints
diagnostic messages and exits gracefully.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread


def check_dataset(data_dir="./data/synthetic_2d", n_samples=6):
    """Load and visualize random image/mask pairs.

    Parameters
    ----------
    data_dir : str
        Path to dataset root folder (should contain ``images/`` and ``masks/``
        subfolders).
    n_samples : int
        Number of random pairs to display.
    """
    # Convert to Path for cross-platform compatibility
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"

    # Check if data directory exists
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path.resolve()}")
        print(f"        Expected structure: {data_path.resolve()}/images/ and /masks/")
        return False

    # Check image and mask directories
    if not images_dir.exists():
        print(f"[ERROR] Images folder not found: {images_dir.resolve()}")
        return False

    if not masks_dir.exists():
        print(f"[ERROR] Masks folder not found: {masks_dir.resolve()}")
        return False

    # Get list of images
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"[ERROR] No PNG images found in {images_dir.resolve()}")
        return False

    print(f"[OK] Found {len(image_files)} images in {images_dir.resolve()}")

    # Randomly select samples
    indices = np.random.choice(len(image_files), size=min(n_samples, len(image_files)), replace=False)
    selected_files = [image_files[i] for i in sorted(indices)]

    # Plot grid
    n_cols = min(3, len(selected_files))
    n_rows = (len(selected_files) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    axes_flat = axes.flatten()

    for ax_idx, img_file in enumerate(selected_files):
        img = imread(img_file)
        mask_file = masks_dir / img_file.name
        mask = imread(mask_file)

        # Normalize image to [0, 1] for display
        img_norm = img.astype(np.float32) / 255.0

        # Overlay: show image in grayscale, mask boundaries in red
        ax = axes_flat[ax_idx]
        ax.imshow(img_norm, cmap="gray")

        # Draw mask contour in red
        mask_binary = (mask > 128).astype(np.uint8)
        contours = np.where(mask_binary > 0)
        if contours[0].size > 0:
            ax.contour(mask_binary, colors="red", linewidths=1)

        ax.set_title(f"{img_file.name}")
        ax.axis("off")

    # Hide unused subplots
    for ax_idx in range(len(selected_files), len(axes_flat)):
        axes_flat[ax_idx].axis("off")

    plt.tight_layout()
    plt.show()
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize random image/mask pairs from a generated dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="./data/synthetic_2d",
        help="Path to dataset folder (default: ./data/synthetic_2d)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of samples to show (default: 6)",
    )
    args = parser.parse_args()

    success = check_dataset(data_dir=args.data_dir, n_samples=args.n)
    sys.exit(0 if success else 1)

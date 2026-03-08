"""
2D Segmentation Dataset with paired augmentations.

Dataset class for loading 2D images and masks with minimal augmentations.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np



class SegmentationDataset2D(Dataset):
    """
    2D Segmentation dataset with paired image-mask augmentations.
    
    Expected directory structure:
        data_dir/
            images/
                *.png (grayscale uint8)
            masks/
                *.png (uint8, foreground=255)
    
    Filenames must match between images and masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        augment: bool = True,
        seed: int = 0
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing 'images' and 'masks' subdirectories
            augment: If True, apply random augmentations
            seed: Random seed for augmentation reproducibility
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.seed = seed
        
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        
        # Get sorted list of image filenames
        image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        
        if not image_files:
            raise ValueError(f"No PNG images found in {self.images_dir}")
        
        self.filenames = image_files
        
        # Verify all masks exist
        for filename in self.filenames:
            mask_path = self.masks_dir / filename
            if not mask_path.exists():
                raise ValueError(f"Mask not found for image {filename}")
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return image-mask pair.
        
        Returns:
            image: Float32 tensor of shape (1, H, W) normalized to [0, 1]
            mask: Float32 tensor of shape (1, H, W) with values {0, 1}
        """
        filename = self.filenames[idx]
        
        # Load image and mask
        image_path = self.images_dir / filename
        mask_path = self.masks_dir / filename
        
        image = Image.open(image_path).convert('L')  # Grayscale
        mask = Image.open(mask_path).convert('L')    # Grayscale
        
        # Convert to numpy arrays
        image_np = np.array(image, dtype=np.float32)  # [0, 255]
        mask_np = np.array(mask, dtype=np.float32)    # [0, 255]
        
        # Normalize image to [0, 1]
        image_np = image_np / 255.0
        
        # Convert mask to binary {0, 1}
        mask_np = (mask_np > 127).astype(np.float32)
        
        # Apply augmentations if enabled
        if self.augment:
            image_np, mask_np = self._apply_augmentations(image_np, mask_np)
        
        # After all augmentations, before torch.from_numpy:
        image_np = np.ascontiguousarray(image_np)
        mask_np  = np.ascontiguousarray(mask_np)

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        mask_tensor  = torch.from_numpy(mask_np).unsqueeze(0)


        # Convert to tensors with shape (1, H, W)
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        return image_tensor, mask_tensor
    
    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply paired augmentations to image and mask.
        
        Args:
            image: Image array of shape (H, W)
            mask: Mask array of shape (H, W)
            
        Returns:
            Augmented image and mask
        """
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if np.random.random() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random brightness adjustment (image only)
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.9, 1.1)
            image = np.clip(image * brightness_factor, 0.0, 1.0)
        
        return image, mask


def make_dataloaders(
    data_dir: str,
    batch_size: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with deterministic split.

    This implementation avoids toggling augment on a shared dataset instance by
    creating two base datasets (one for training with augmentations enabled and
    one for evaluation without augmentations) and then using Subset to select the
    appropriate indices for each split. The index order is randomized once using
    the provided seed so that splits are reproducible.
    
    Args:
        data_dir: Root directory containing dataset
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation (default 0.1)
        test_split: Fraction of data for testing (default 0.1)
        seed: Random seed for deterministic splitting
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # first, determine how many samples we have
    base_dataset = SegmentationDataset2D(data_dir, augment=False, seed=seed)
    total_size = len(base_dataset)
    if total_size == 0:
        raise ValueError(f"No samples found in dataset at {data_dir}")
    
    # compute split sizes
    test_size = int(total_size * test_split)
    val_size = int((total_size - test_size) * val_split)
    train_size = total_size - val_size - test_size
    
    # create a deterministic permutation of indices
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=gen).tolist()
    
    train_indices = perm[:train_size]
    val_indices = perm[train_size:train_size + val_size]
    test_indices = perm[train_size + val_size:]
    
    from torch.utils.data import Subset
    
    # separate base datasets for augmentation control
    train_base = SegmentationDataset2D(data_dir, augment=True, seed=seed)
    eval_base = SegmentationDataset2D(data_dir, augment=False, seed=seed)
    
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(eval_base, val_indices)
    test_dataset = Subset(eval_base, test_indices)
    
    # build loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset and dataloaders
    print("Testing SegmentationDataset2D...")
    
    # Check if test data exists
    test_dir = "./test_data"
    if not os.path.exists(test_dir):
        print(f"Test data directory {test_dir} not found. Skipping test.")
    else:
        # Create dataloaders
        train_loader, val_loader, test_loader = make_dataloaders(
            test_dir,
            batch_size=4,
            val_split=0.2,
            test_split=0.2,
            seed=42,
            num_workers=0
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        # Get a batch and print info
        for images, masks in train_loader:
            print(f"\nBatch from train loader:")
            print(f"  Image shape: {images.shape}, dtype: {images.dtype}")
            print(f"  Image min: {images.min():.4f}, max: {images.max():.4f}")
            print(f"  Mask shape: {masks.shape}, dtype: {masks.dtype}")
            print(f"  Mask unique values: {torch.unique(masks).tolist()}")
            break
        
        print("\nTest completed successfully!")

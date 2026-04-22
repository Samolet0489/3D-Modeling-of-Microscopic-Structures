"""
Training script for 2D U-Net segmentation.

Trains UNet2D on paired image-mask dataset with BCEWithLogitsLoss.
Includes mixed precision support, checkpoint saving, and prediction snapshots.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # needed for rebuilding loader
# use the generic amp API (works for cuda, cpu in newer PyTorch versions)
from torch.amp import autocast, GradScaler

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from src.dataset2d import SegmentationDataset2D, make_dataloaders
from src.unet2d import UNet2D
from src.metrics import threshold_logits, dice_coeff, iou_score


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train UNet2D for 2D image segmentation"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset folder (with images/ and masks/ subdirs)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm; <=0 to disable (default: 1.0)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/train_2d",
        help="Output directory for checkpoints and logs (default: runs/train_2d)"
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Limit training to first N samples (for debug/overfit testing)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision when using CUDA"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save prediction snapshots every N epochs (default: 5)"
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive class weight for BCEWithLogitsLoss; computed from train data if unspecified"
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.5,
        help="Threshold for binary predictions (default: 0.5)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers (default: 2)"
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """
    Get device based on argument.
    
    Args:
        device_arg: 'auto', 'cpu', or 'cuda'
        
    Returns:
        torch.device object
    """
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    return device


def print_device_info(device: torch.device):
    """Print device information."""
    print(f"Device: {device}")
    if device.type == "cuda":
        # torch.cuda.get_device_name expects an integer index, not a torch.device
        idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = f"<unknown, idx={idx}>"
        print(f"GPU: {name}")
        print(f"CUDA version: {torch.version.cuda}")


def create_pred_snapshot(
    model: nn.Module,
    val_loader,
    device: torch.device,
    epoch: int = None,
    out_dir: Path = None,
    threshold: float = 0.5,
    num_samples: int = 6,
    suffix: str = None
):
    """
    Create and save a visualization grid of predictions.
    
    Args:
        model: Segmentation model
        val_loader: Validation dataloader
        device: Device to use
        epoch: Current epoch number
        out_dir: Output directory
        threshold: Threshold for binary predictions
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    samples_collected = 0
    images_list = []
    targets_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            preds = threshold_logits(logits, thr=threshold)
            
            # Collect samples
            batch_size = images.shape[0]
            for i in range(batch_size):
                if samples_collected >= num_samples:
                    break
                
                images_list.append(images[i:i+1])
                targets_list.append(masks[i:i+1])
                preds_list.append(preds[i:i+1])
                samples_collected += 1
            
            if samples_collected >= num_samples:
                break
    
    # Create grid visualization
    grid_width = 3 * 128 + 20  # 3 images + padding
    grid_height = num_samples * 128 + 20 * (num_samples - 1)
    grid_image = Image.new('L', (grid_width, grid_height), color=255)
    
    y_offset = 10
    for idx in range(len(images_list)):
        # Image (normalize from [0,1] to [0,255])
        img = (images_list[idx].squeeze().cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        grid_image.paste(img_pil, (10, y_offset))
        
        # Target mask (convert from [0,1] to [0,255])
        target = (targets_list[idx].squeeze().cpu().numpy() * 255).astype(np.uint8)
        target_pil = Image.fromarray(target)
        grid_image.paste(target_pil, (138, y_offset))
        
        # Predicted mask (already binary, convert to [0,255])
        pred = (preds_list[idx].squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred)
        grid_image.paste(pred_pil, (266, y_offset))
        
        y_offset += 128 + 20
    
    # Save grid with appropriate filename
    if suffix is not None:
        output_path = out_dir / f"preds_{suffix}.png"
    else:
        output_path = out_dir / f"preds_epoch_{epoch:03d}.png"
    grid_image.save(output_path)
    print(f"  Saved prediction snapshot to {output_path}")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler = None,
    grad_clip: float = 1.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Segmentation model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        use_amp: Use automatic mixed precision
        scaler: GradScaler for AMP
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, masks)
        else:
            logits = model(images)
            loss = criterion(logits, masks)
        
        # check for non-finite loss
        if not torch.isfinite(loss):
            print(f"Warning: non-finite loss ({loss.item()}); skipping batch")
            continue
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # gradient clipping
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Validate model.
    
    Args:
        model: Segmentation model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to use
        threshold: Threshold for binary predictions
        
    Returns:
        Tuple of (val_loss, val_dice, val_iou)
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            # Get predictions
            preds = threshold_logits(logits, thr=threshold)
            
            # Compute metrics
            dice = dice_coeff(preds, masks)
            iou = iou_score(preds, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_iou += iou.item()
            num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    is_best: bool,
    out_dir: Path,
    metric_value: float = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        is_best: Whether this is the best checkpoint
        out_dir: Output directory
        metric_value: Metric value (for logging)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save last checkpoint
    last_path = out_dir / "checkpoint_last.pt"
    torch.save(checkpoint, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = out_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        if metric_value is not None:
            print(f"  Saved best checkpoint (metric: {metric_value:.6f})")


def limit_train_dataset(dataset, limit: int):
    """
    Limit training dataset to first N samples.
    
    Args:
        dataset: PyTorch Subset or similar
        limit: Maximum number of samples
        
    Returns:
        Limited dataset
    """
    from torch.utils.data import Subset
    
    if hasattr(dataset, 'indices'):
        # It's a Subset
        limited_indices = dataset.indices[:limit]
        return Subset(dataset.dataset, limited_indices)
    else:
        # Direct dataset
        return Subset(dataset, range(min(limit, len(dataset))))


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Get device
    device = get_device(args.device)
    print_device_info(device)
    
    # Set cuDNN benchmark for faster training
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")
    
    # Create dataloaders
    print(f"\nLoading dataset from {args.data}...")
    train_loader, val_loader, test_loader = make_dataloaders(
        args.data,
        batch_size=args.batch_size,
        val_split=0.1,
        test_split=0.1,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Apply training limit if specified
    if args.limit_train is not None:
        print(f"\n*** OVERFIT MODE: Limiting training to {args.limit_train} samples ***")
        from torch.utils.data import Subset
        # build evaluation dataset without augment to inspect masks
        eval_base = SegmentationDataset2D(args.data, augment=False, seed=args.seed)
        limited_indices = []
        for i in range(len(eval_base)):
            _, mask = eval_base[i]
            if mask.sum() > 0:
                limited_indices.append(i)
                if len(limited_indices) >= args.limit_train:
                    break
        # fallback if not enough non-empty masks
        if len(limited_indices) < args.limit_train:
            limited_indices = list(range(min(args.limit_train, len(eval_base))))
        # compute coverage mean
        if limited_indices:
            coverages = [eval_base[i][1].mean().item() for i in limited_indices]
            mean_cov = sum(coverages) / len(coverages)
        else:
            mean_cov = 0.0
        print(f"Limited subset mask coverage mean: {mean_cov:.6f}")
        
        # rebuild loaders using the same limited indices
        train_base = SegmentationDataset2D(args.data, augment=True, seed=args.seed)
        train_ds = Subset(train_base, limited_indices)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_ds = Subset(eval_base, limited_indices)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        # optionally restrict test as well
        test_ds = Subset(eval_base, limited_indices)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Train batches after limit: {len(train_loader)}")
        print(f"Val batches after limit: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating UNet2D model...")
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # determine positive weight
    if args.pos_weight is not None:
        effective_pos_weight = args.pos_weight
        print(f"Using provided pos_weight={effective_pos_weight}")
    else:
        # estimate from training data (first few batches)
        pos = 0.0
        neg = 0.0
        seen = 0
        for images, masks in train_loader:
            masks = masks.to(device)
            pos += masks.sum().item()
            neg += masks.numel() - masks.sum().item()
            seen += 1
            if seen >= 10:
                break
        if pos + neg == 0:
            effective_pos_weight = 1.0
        else:
            effective_pos_weight = neg / (pos + 1e-6)
            effective_pos_weight = float(min(effective_pos_weight, 100.0))
        print(f"Estimated pos_weight from train data: {effective_pos_weight:.6f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([effective_pos_weight], device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Mixed precision scaler (for CUDA only and if requested)
    use_amp = (device.type == "cuda" and args.amp)

    if use_amp:
        try:
            # newer API style
            scaler = GradScaler("cuda")
        except TypeError:
            try:
                # older API style
                scaler = GradScaler()
            except TypeError:
                scaler = None
                use_amp = False
                print("Warning: AMP GradScaler not supported in this PyTorch version. Falling back to full precision.")
    else:
        scaler = None


    # # Mixed precision scaler (for CUDA only and if requested)  - REWORKED VERSION
    # use_amp = (device.type == "cuda" and args.amp)

    # if use_amp:
    #     try:
    #         # newer API style
    #         scaler = GradScaler("cuda")
    #     except TypeError:
    #         try:
    #             # older API style
    #             scaler = GradScaler()
    #         except TypeError:
    #             scaler = None
    #             use_amp = False
    #             print("Warning: AMP GradScaler not supported in this PyTorch version. Falling back to full precision.")
    # else:
    #     scaler = None

    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    best_val_dice = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler,
            grad_clip=args.grad_clip
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, args.thr
        )
        
        # Log
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Dice: {val_dice:.6f} | "
            f"Val IoU: {val_iou:.6f}"
        )
        
        # Save checkpoint
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
        
        if epoch % args.save_every == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, is_best, out_dir, val_dice)
        
        # Save prediction snapshot
        if epoch % args.save_every == 0:
            create_pred_snapshot(model, val_loader, device, epoch, out_dir, args.thr)
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice: {best_val_dice:.6f}")
    print(f"Checkpoints saved to {out_dir}")
    
    # evaluate best checkpoint on test set
    best_path = out_dir / "checkpoint_best.pt"
    if best_path.exists():
        print("\nLoading best checkpoint for test evaluation...")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        test_loss, test_dice, test_iou = validate(model, test_loader, criterion, device, args.thr)
        print(f"Test Dice: {test_dice:.6f} | Test IoU: {test_iou:.6f}")
        # save snapshot
        create_pred_snapshot(model, test_loader, device, epoch=0, out_dir=out_dir, threshold=args.thr, suffix='test')


if __name__ == "__main__":
    main()

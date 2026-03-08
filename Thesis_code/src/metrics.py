"""
Segmentation metrics for 2D image segmentation.

Metrics include Dice coefficient, IoU score, and logit thresholding.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def threshold_logits(
    logits: torch.Tensor,
    thr: float = 0.5
) -> torch.Tensor:
    """
    Convert logits to binary predictions using sigmoid and threshold.
    
    Args:
        logits: Logit tensor of shape (B, 1, H, W)
        thr: Threshold value (default 0.5)
        
    Returns:
        Binary predictions of shape (B, 1, H, W) with values {0, 1}
    """
    probs = torch.sigmoid(logits)
    return (probs > thr).float()


def dice_coeff(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient (F1 score) for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        pred: Predicted mask of shape (B, 1, H, W) with values {0, 1}
        target: Target mask of shape (B, 1, H, W) with values {0, 1}
        eps: Small epsilon for numerical stability (default 1e-6)
        
    Returns:
        Dice coefficient as a scalar tensor (0 to 1)
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Compute intersection and sizes
    intersection = (pred_flat * target_flat).sum()
    
    # Dice coefficient
    dice = (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)
    
    return dice


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute Intersection over Union (Jaccard index) for segmentation.
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        pred: Predicted mask of shape (B, 1, H, W) with values {0, 1}
        target: Target mask of shape (B, 1, H, W) with values {0, 1}
        eps: Small epsilon for numerical stability (default 1e-6)
        
    Returns:
        IoU score as a scalar tensor (0 to 1)
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = (pred_flat + target_flat - pred_flat * target_flat).sum()
    
    # IoU score
    iou = (intersection + eps) / (union + eps)
    
    return iou


if __name__ == "__main__":
    print("Testing segmentation metrics...")
    
    # Create test tensors
    batch_size = 2
    height, width = 64, 64
    
    # Simulate logits
    logits = torch.randn(batch_size, 1, height, width)
    
    # Simulate perfect prediction
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    pred_perfect = target.clone()
    
    # Simulate random prediction
    pred_random = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test threshold_logits
    print("\n--- threshold_logits ---")
    binary_pred = threshold_logits(logits, thr=0.5)
    print(f"Binary predictions shape: {binary_pred.shape}")
    print(f"Unique values: {torch.unique(binary_pred).tolist()}")
    
    # Test dice_coeff
    print("\n--- dice_coeff ---")
    dice_perfect = dice_coeff(pred_perfect, target)
    dice_random = dice_coeff(pred_random, target)
    print(f"Dice (perfect prediction): {dice_perfect:.6f}")
    print(f"Dice (random prediction): {dice_random:.6f}")
    
    # Test iou_score
    print("\n--- iou_score ---")
    iou_perfect = iou_score(pred_perfect, target)
    iou_random = iou_score(pred_random, target)
    print(f"IoU (perfect prediction): {iou_perfect:.6f}")
    print(f"IoU (random prediction): {iou_random:.6f}")
    
    # Test on GPU if available
    print("\n--- GPU Test ---")
    if torch.cuda.is_available():
        print("Testing on GPU...")
        logits_gpu = logits.cuda()
        target_gpu = target.cuda()
        pred_perfect_gpu = pred_perfect.cuda()
        
        binary_pred_gpu = threshold_logits(logits_gpu, thr=0.5)
        dice_gpu = dice_coeff(pred_perfect_gpu, target_gpu)
        iou_gpu = iou_score(pred_perfect_gpu, target_gpu)
        
        print(f"GPU - Binary predictions shape: {binary_pred_gpu.shape}")
        print(f"GPU - Dice (perfect): {dice_gpu:.6f}")
        print(f"GPU - IoU (perfect): {iou_gpu:.6f}")
    else:
        print("CUDA not available, skipping GPU test")
    
    print("\nTest completed successfully!")

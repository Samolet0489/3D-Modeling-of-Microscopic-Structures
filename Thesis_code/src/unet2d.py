"""
2D U-Net implementation for image segmentation.

Standard U-Net architecture with skip connections for 2D segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv2d -> ReLU -> Conv2d -> ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net for image segmentation.
    
    Architecture:
        - Encoder: Progressive downsampling with skip connections
        - Bottleneck: Deepest convolutional block
        - Decoder: Progressive upsampling with skip connections
    
    Input shape: (B, in_channels, H, W)
    Output shape: (B, out_channels, H, W) - logits, no activation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32
    ):
        """
        Initialize U-Net.
        
        Args:
            in_channels: Number of input channels (default 1 for grayscale)
            out_channels: Number of output channels (default 1 for binary segmentation)
            base_channels: Number of channels in first conv block (default 32)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W)
            
        Returns:
            Output tensor of shape (B, out_channels, H, W) - logits
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)           # (B, C, H, W)
        x = self.pool1(enc1)          # (B, C, H/2, W/2)
        
        enc2 = self.enc2(x)           # (B, 2C, H/2, W/2)
        x = self.pool2(enc2)          # (B, 2C, H/4, W/4)
        
        enc3 = self.enc3(x)           # (B, 4C, H/4, W/4)
        x = self.pool3(enc3)          # (B, 4C, H/8, W/8)
        
        enc4 = self.enc4(x)           # (B, 8C, H/8, W/8)
        x = self.pool4(enc4)          # (B, 8C, H/16, W/16)
        
        # Bottleneck
        x = self.bottleneck(x)        # (B, 16C, H/16, W/16)
        
        # Decoder with skip connections
        x = self.upconv4(x)           # (B, 8C, H/8, W/8)
        x = torch.cat([x, enc4], dim=1)  # (B, 16C, H/8, W/8)
        x = self.dec4(x)              # (B, 8C, H/8, W/8)
        
        x = self.upconv3(x)           # (B, 4C, H/4, W/4)
        x = torch.cat([x, enc3], dim=1)  # (B, 8C, H/4, W/4)
        x = self.dec3(x)              # (B, 4C, H/4, W/4)
        
        x = self.upconv2(x)           # (B, 2C, H/2, W/2)
        x = torch.cat([x, enc2], dim=1)  # (B, 4C, H/2, W/2)
        x = self.dec2(x)              # (B, 2C, H/2, W/2)
        
        x = self.upconv1(x)           # (B, C, H, W)
        x = torch.cat([x, enc1], dim=1)  # (B, 2C, H, W)
        x = self.dec1(x)              # (B, C, H, W)
        
        # Final output (logits, no activation)
        x = self.final_conv(x)        # (B, out_channels, H, W)
        
        return x


if __name__ == "__main__":
    print("Testing UNet2D...")
    
    # Create model
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32)
    print(f"Model created: {model.__class__.__name__}")
    
    # Create random input
    x = torch.randn(2, 1, 128, 128)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Verify output is float (logits)
    print(f"Output dtype: {y.dtype}")
    print(f"Output min: {y.min():.4f}, max: {y.max():.4f}")
    
    print("\nTest completed successfully!")

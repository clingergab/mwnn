"""Adaptive fusion modules for Single-Output Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SingleOutputStage(nn.Module):
    """Stage with single-output multi-weight processing."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 num_blocks: int, downsample: bool,
                 adaptive: bool = False):
        super().__init__()
        
        self.adaptive = adaptive
        stride = 2 if downsample else 1
        
        # Multi-weight blocks
        self.blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            if i == 0:
                block = MultiWeightResidualBlock(
                    in_channels, out_channels, stride, adaptive
                )
            else:
                block = MultiWeightResidualBlock(
                    out_channels, out_channels, 1, adaptive
                )
            self.blocks.append(block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stage."""
        for block in self.blocks:
            x = block(x)
        return x


class MultiWeightResidualBlock(nn.Module):
    """Residual block with multi-weight processing."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, adaptive: bool = False):
        super().__init__()
        
        self.adaptive = adaptive
        
        # First convolution (standard)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Multi-weight convolution
        self.mw_conv = MultiWeightConv2d(
            out_channels, out_channels, 3, 1, 1,
            adaptive=adaptive
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through block."""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.mw_conv(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class MultiWeightConv2d(nn.Module):
    """2D Convolution with separate weights for color and brightness."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 adaptive: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.adaptive = adaptive
        
        # Separate convolutions for color and brightness
        self.color_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, bias=False
        )
        self.brightness_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, bias=False
        )
        
        if adaptive:
            # Adaptive fusion network
            self.fusion_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels // 4),
                nn.ReLU(),
                nn.Linear(in_channels // 4, 2),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-weight convolution."""
        # Separate into color and brightness components
        # This is a simplified separation - in practice would use proper extraction
        brightness = torch.mean(x, dim=1, keepdim=True)
        brightness = brightness.expand_as(x)
        color = x - brightness + 0.5  # Normalize around 0.5
        
        # Process with separate convolutions
        color_out = self.color_conv(color)
        brightness_out = self.brightness_conv(brightness)
        
        if self.adaptive:
            # Adaptive fusion
            weights = self.fusion_net(x)
            weights = weights.view(-1, 2, 1, 1, 1)
            output = (weights[:, 0] * color_out + 
                     weights[:, 1] * brightness_out)
        else:
            # Simple addition
            output = color_out + brightness_out
        
        return output


class AdaptiveFusionBlock(nn.Module):
    """Adaptive fusion block for combining features."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, color_features: torch.Tensor, 
                brightness_features: torch.Tensor) -> torch.Tensor:
        """Adaptively fuse color and brightness features."""
        # Concatenate features
        combined = torch.cat([color_features, brightness_features], dim=1)
        
        # Compute fusion weights
        weights = self.gate_network(combined)
        weights = weights.view(-1, 2, 1, 1)
        
        # Apply weights
        fused = (weights[:, 0] * color_features + 
                weights[:, 1] * brightness_features)
        
        return fused
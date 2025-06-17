"""Shared building blocks for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class ConvBlock(nn.Module):
    """Standard convolutional block with batch norm and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, activation: str = 'relu'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, activation: str = 'relu'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = self._get_activation(activation)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class AttentionBlock(nn.Module):
    """Self-attention block for feature refinement."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FusionBlock(nn.Module):
    """Block for fusing color and brightness features."""
    
    def __init__(self, channels: int, fusion_type: str = 'adaptive'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'adaptive':
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 2, 1),
                nn.Softmax(dim=1)
            )
        elif fusion_type == 'attention':
            self.color_attention = AttentionBlock(channels)
            self.brightness_attention = AttentionBlock(channels)
    
    def forward(self, color: torch.Tensor, brightness: torch.Tensor) -> torch.Tensor:
        """Fuse color and brightness features."""
        if self.fusion_type == 'concatenate':
            return torch.cat([color, brightness], dim=1)
        
        elif self.fusion_type == 'add':
            return color + brightness
        
        elif self.fusion_type == 'adaptive':
            combined = torch.cat([color, brightness], dim=1)
            weights = self.gate(combined)
            # weights has shape (B, 2, H, W) after softmax
            return weights[:, 0:1, :, :] * color + weights[:, 1:2, :, :] * brightness
        
        elif self.fusion_type == 'attention':
            color_refined = self.color_attention(color)
            brightness_refined = self.brightness_attention(brightness)
            return color_refined + brightness_refined
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class GlobalContextBlock(nn.Module):
    """Global context aggregation block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LayerNorm([channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Global average pooling
        context = x.view(b, c, -1).mean(-1, keepdim=True)
        context = context.view(b, c, 1, 1)
        
        # Transform
        context = self.channel_add_conv(context)
        
        # Broadcast and add
        return x + context
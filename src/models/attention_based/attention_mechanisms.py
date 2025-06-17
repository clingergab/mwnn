"""Attention mechanisms for the Attention-Based Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from ..components.blocks import ResidualBlock


class CrossModalAttention(nn.Module):
    """Cross-modal attention module with multi-head support."""
    
    def __init__(self, channels: int, attention_dim: int = 64, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.channels = channels
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention components
        self.color_qkv = nn.Linear(channels, attention_dim * 3)
        self.brightness_qkv = nn.Linear(channels, attention_dim * 3)
        
        # Output projections
        self.color_output = nn.Linear(attention_dim, channels)
        self.brightness_output = nn.Linear(attention_dim, channels)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.color_norm = nn.LayerNorm(channels)
        self.brightness_norm = nn.LayerNorm(channels)
        
        self.last_attention_weights = None
    
    def forward(self, color: torch.Tensor, 
                brightness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention."""
        B = color.shape[0]
        
        # For 4D inputs, reshape to (B, C, H*W) then transpose
        if len(color.shape) == 4:
            _, C, H, W = color.shape
            color_flat = color.flatten(2).transpose(1, 2)  # (B, H*W, C)
            brightness_flat = brightness.flatten(2).transpose(1, 2)
            spatial_shape = (H, W)
        elif len(color.shape) == 2:
            # For 2D inputs (B, C), add sequence dimension
            color_flat = color.unsqueeze(1)  # (B, 1, C)
            brightness_flat = brightness.unsqueeze(1)
            spatial_shape = None
        else:
            raise ValueError(f"Expected 2D or 4D input, got {len(color.shape)}D")
        
        # Compute queries, keys, values for color
        color_qkv = self.color_qkv(color_flat).reshape(
            B, -1, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        color_q, color_k, color_v = color_qkv[0], color_qkv[1], color_qkv[2]
        
        # Compute queries, keys, values for brightness
        brightness_qkv = self.brightness_qkv(brightness_flat).reshape(
            B, -1, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        brightness_q, brightness_k, brightness_v = brightness_qkv[0], brightness_qkv[1], brightness_qkv[2]
        
        # Cross-modal attention: color attending to brightness
        color_attn = (color_q @ brightness_k.transpose(-2, -1)) * self.scale
        color_attn = F.softmax(color_attn, dim=-1)
        color_attn = self.attn_dropout(color_attn)
        color_attended = (color_attn @ brightness_v).transpose(1, 2).reshape(
            B, -1, self.attention_dim
        )
        
        # Cross-modal attention: brightness attending to color
        brightness_attn = (brightness_q @ color_k.transpose(-2, -1)) * self.scale
        brightness_attn = F.softmax(brightness_attn, dim=-1)
        brightness_attn = self.attn_dropout(brightness_attn)
        brightness_attended = (brightness_attn @ color_v).transpose(1, 2).reshape(
            B, -1, self.attention_dim
        )
        
        # Store attention weights for visualization
        self.last_attention_weights = {
            'color_to_brightness': color_attn.detach().mean(dim=1),  # Average over heads
            'brightness_to_color': brightness_attn.detach().mean(dim=1)
        }
        
        # Project back to original dimension
        color_out = self.proj_dropout(self.color_output(color_attended))
        brightness_out = self.proj_dropout(self.brightness_output(brightness_attended))
        
        # Reshape back to original spatial dimensions if needed
        if spatial_shape is not None:
            H, W = spatial_shape
            color_out = color_out.transpose(1, 2).reshape(B, self.channels, H, W)
            brightness_out = brightness_out.transpose(1, 2).reshape(B, self.channels, H, W)
            
            # Residual connection for conv features
            color_out = color + color_out
            brightness_out = brightness + brightness_out
        else:
            # For 1D features
            color_out = color_out.squeeze(1)
            brightness_out = brightness_out.squeeze(1)
            
            # Residual and norm
            color_out = self.color_norm(color + color_out)
            brightness_out = self.brightness_norm(brightness + brightness_out)
        
        return color_out, brightness_out
    
    def get_last_attention_weights(self):
        """Get the last computed attention weights."""
        return self.last_attention_weights


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling layer."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        
        self.value_conv = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention pooling."""
        B, C, H, W = x.shape
        
        # Compute attention weights
        attention_weights = self.attention(x)
        attention_weights = attention_weights.view(B, C, -1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Compute values
        values = self.value_conv(x).view(B, C, -1)
        
        # Apply attention
        attended = torch.bmm(attention_weights, values.transpose(1, 2))
        attended = attended.mean(dim=-1)  # Global aggregation
        
        return attended


class AttentionStage(nn.Module):
    """Stage with attention-based cross-modal processing."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 num_blocks: int, downsample: bool,
                 attention_dim: int = 64, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_blocks = num_blocks
        stride = 2 if downsample else 1
        
        # Residual blocks for each pathway
        self.color_blocks = nn.ModuleList()
        self.brightness_blocks = nn.ModuleList()
        
        # Cross-modal attention modules
        self.attention_modules = nn.ModuleList()
        
        for i in range(num_blocks):
            # First block handles downsampling
            if i == 0:
                self.color_blocks.append(
                    ResidualBlock(in_channels, out_channels, stride)
                )
                self.brightness_blocks.append(
                    ResidualBlock(in_channels, out_channels, stride)
                )
                current_channels = out_channels
            else:
                self.color_blocks.append(
                    ResidualBlock(out_channels, out_channels, 1)
                )
                self.brightness_blocks.append(
                    ResidualBlock(out_channels, out_channels, 1)
                )
                current_channels = out_channels
            
            # Add cross-modal attention every other block
            if i % 2 == 1 or i == num_blocks - 1:
                self.attention_modules.append(
                    SpatialCrossModalAttention(
                        current_channels, attention_dim, num_heads, dropout
                    )
                )
            else:
                self.attention_modules.append(None)
        
        self.last_attention_maps = None
    
    def forward(self, color: torch.Tensor, 
                brightness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention stage."""
        attention_maps = []
        
        for i, (color_block, brightness_block, attention) in enumerate(
            zip(self.color_blocks, self.brightness_blocks, self.attention_modules)
        ):
            # Process through blocks
            color = color_block(color)
            brightness = brightness_block(brightness)
            
            # Apply cross-modal attention if available
            if attention is not None:
                color, brightness = attention(color, brightness)
                
                # Store attention maps
                if hasattr(attention, 'last_attention_weights'):
                    attention_maps.append(attention.last_attention_weights)
        
        self.last_attention_maps = attention_maps
        return color, brightness
    
    def get_last_attention_maps(self):
        """Get the last computed attention maps."""
        return self.last_attention_maps


class SpatialCrossModalAttention(nn.Module):
    """Spatial cross-modal attention for feature maps."""
    
    def __init__(self, channels: int, attention_dim: int = 64,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Spatial attention components
        self.color_spatial_attn = nn.Sequential(
            nn.Conv2d(channels, attention_dim, 1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self.brightness_spatial_attn = nn.Sequential(
            nn.Conv2d(channels, attention_dim, 1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = ChannelCrossModalAttention(
            channels, num_heads, dropout
        )
        
        self.last_attention_weights = None
    
    def forward(self, color: torch.Tensor,
                brightness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial and channel cross-modal attention."""
        # Spatial attention
        color_spatial_attn = self.color_spatial_attn(brightness)
        brightness_spatial_attn = self.brightness_spatial_attn(color)
        
        # Apply spatial attention
        color_spatially_attended = color * color_spatial_attn
        brightness_spatially_attended = brightness * brightness_spatial_attn
        
        # Channel attention
        color_out, brightness_out = self.channel_attention(
            color_spatially_attended, brightness_spatially_attended
        )
        
        # Store attention maps
        self.last_attention_weights = {
            'color_spatial': color_spatial_attn.detach(),
            'brightness_spatial': brightness_spatial_attn.detach()
        }
        
        return color_out, brightness_out


class ChannelCrossModalAttention(nn.Module):
    """Channel-wise cross-modal attention."""
    
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = channels ** 0.5
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter
        self.temperature_param = nn.Parameter(torch.ones(1))
    
    def forward(self, color: torch.Tensor,
                brightness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply channel-wise cross-modal attention."""
        B, C, H, W = color.shape
        
        # Global average pooling for channel descriptors
        color_desc = F.adaptive_avg_pool2d(color, 1).view(B, C)
        brightness_desc = F.adaptive_avg_pool2d(brightness, 1).view(B, C)
        
        # Compute channel attention
        color_attn = torch.matmul(color_desc, brightness_desc.transpose(0, 1))
        color_attn = F.softmax(color_attn / (self.temperature * self.temperature_param), dim=-1)
        color_attn = self.dropout(color_attn)
        
        brightness_attn = torch.matmul(brightness_desc, color_desc.transpose(0, 1))
        brightness_attn = F.softmax(brightness_attn / (self.temperature * self.temperature_param), dim=-1)
        brightness_attn = self.dropout(brightness_attn)
        
        # Apply channel attention
        color_out = color * color_attn.view(B, C, 1, 1)
        brightness_out = brightness * brightness_attn.view(B, C, 1, 1)
        
        return color_out, brightness_out
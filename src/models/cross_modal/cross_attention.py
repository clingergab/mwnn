"""Cross-attention modules for Cross-Modal Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..components.blocks import ResidualBlock


class CrossModalStage(nn.Module):
    """Stage with cross-modal influence between pathways."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 num_blocks: int, downsample: bool,
                 cross_influence: float = 0.1):
        super().__init__()
        
        self.cross_influence = cross_influence
        stride = 2 if downsample else 1
        
        # Build blocks for each pathway
        self.color_blocks = nn.ModuleList()
        self.brightness_blocks = nn.ModuleList()
        self.cross_attention_modules = nn.ModuleList()
        
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
            
            # Add cross-attention module
            self.cross_attention_modules.append(
                CrossAttentionModule(current_channels, cross_influence)
            )
    
    def forward(self, color: torch.Tensor, 
                brightness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through cross-modal stage."""
        for i, (color_block, brightness_block, cross_attention) in enumerate(
            zip(self.color_blocks, self.brightness_blocks, self.cross_attention_modules)
        ):
            # Process through blocks
            color = color_block(color)
            brightness = brightness_block(brightness)
            
            # Apply cross-modal attention
            color, brightness = cross_attention(color, brightness)
        
        return color, brightness
    
    def get_cross_weights(self, direction: str) -> float:
        """Get average cross-influence weights."""
        weights = []
        for module in self.cross_attention_modules:
            if direction == 'cb':
                weights.append(module.color_to_brightness_weight.mean().item())
            else:
                weights.append(module.brightness_to_color_weight.mean().item())
        return sum(weights) / len(weights) if weights else 0.0


class CrossAttentionModule(nn.Module):
    """Cross-attention module for cross-modal influence."""
    
    def __init__(self, channels: int, cross_influence: float = 0.1):
        super().__init__()
        
        # Learnable cross-influence weights
        self.color_to_brightness_weight = nn.Parameter(
            torch.ones(1) * cross_influence
        )
        self.brightness_to_color_weight = nn.Parameter(
            torch.ones(1) * cross_influence
        )
        
        # Attention computation
        self.color_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        self.brightness_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, color: torch.Tensor, 
                brightness: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention."""
        # Compute attention maps
        color_att = self.color_attention(color)
        brightness_att = self.brightness_attention(brightness)
        
        # Apply cross-influence
        color_influenced = color + self.brightness_to_color_weight * brightness * color_att
        brightness_influenced = brightness + self.color_to_brightness_weight * color * brightness_att
        
        return color_influenced, brightness_influenced


class CrossModalLinear(nn.Module):
    """Linear layer with cross-modal influence."""
    
    def __init__(self, in_features: int, out_features: int,
                 cross_influence: float = 0.1, activation: str = 'relu', 
                 bias: bool = True):
        super().__init__()
        
        # Direct pathway weights
        self.color_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        # Cross-influence weights
        self.cross_weights_cb = nn.Parameter(torch.randn(out_features, in_features) * cross_influence)
        self.cross_weights_bc = nn.Parameter(torch.randn(out_features, in_features) * cross_influence)
        
        if bias:
            self.color_bias = nn.Parameter(torch.zeros(out_features))
            self.brightness_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('color_bias', None)
            self.register_parameter('brightness_bias', None)
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, color_input: torch.Tensor, 
                brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-modal influence."""
        # Direct pathways
        color_direct = F.linear(color_input, self.color_weights)
        brightness_direct = F.linear(brightness_input, self.brightness_weights)
        
        # Cross-influence pathways
        color_from_brightness = F.linear(brightness_input, self.cross_weights_cb)
        brightness_from_color = F.linear(color_input, self.cross_weights_bc)
        
        # Combine with cross-influence
        color_output = color_direct + color_from_brightness
        brightness_output = brightness_direct + brightness_from_color
        
        if self.color_bias is not None:
            color_output = color_output + self.color_bias
            brightness_output = brightness_output + self.brightness_bias
        
        # Apply activation
        color_output = self.activation(color_output)
        brightness_output = self.activation(brightness_output)
        
        return color_output, brightness_output
"""Integration modules for Continuous Integration Model - GPU Optimized."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..components.blocks import ResidualBlock


class IntegrationModule(nn.Module):
    """GPU-optimized learnable integration module for combining features."""
    
    def __init__(self, color_channels: int, brightness_channels: int, 
                 memory_efficient: bool = True):
        super().__init__()
        
        self.memory_efficient = memory_efficient
        
        # Learnable integration weights - optimized initialization
        self.color_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.brightness_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.integrated_weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        # GPU-optimized non-linear integration with fused operations
        self.integration_mlp = nn.Sequential(
            nn.Conv2d(color_channels + brightness_channels, color_channels, 1, bias=False),
            nn.BatchNorm2d(color_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(color_channels, color_channels, 1, bias=False),
            nn.BatchNorm2d(color_channels)
        )
        
        # For final integration (FC layers) - optimized for GPU
        self.fc_integration = nn.Sequential(
            nn.Linear(color_channels + brightness_channels, color_channels, bias=False),
            nn.BatchNorm1d(color_channels),
            nn.ReLU(inplace=True),
            nn.Linear(color_channels, color_channels, bias=False)
        )
        
        # Cache for weight computation to reduce redundant softmax calls
        self._cached_weights = None
        self._weight_cache_valid = False
    
    def forward(self, color: torch.Tensor, brightness: torch.Tensor,
                integrated: Optional[torch.Tensor] = None) -> torch.Tensor:
        """GPU-optimized learnable integration with minimal bottlenecks."""
        # Skip unnecessary contiguity checks for better performance
        # Tensors should already be contiguous from previous operations
        
        # Get normalized weights with caching for efficiency
        has_integrated = integrated is not None and integrated.shape == color.shape
        weights = self._get_cached_weights(has_integrated)
        
        # GPU-optimized non-linear integration (do this first to overlap with weight computation)
        if len(color.shape) == 4:  # Conv features
            # Fused concatenation and MLP processing
            concat_features = torch.cat([color, brightness], dim=1)
            non_linear = self.integration_mlp(concat_features)
        else:  # FC features
            concat_features = torch.cat([color, brightness], dim=1)
            non_linear = self.fc_integration(concat_features)
        
        # Optimized linear combination using efficient tensor operations
        if has_integrated:
            # Three-way integration with broadcasting - keep on GPU
            # Don't use .item() to avoid CPU/GPU transfers
            linear_combination = (weights[0] * color + 
                                weights[1] * brightness + 
                                weights[2] * integrated)
        else:
            # Two-way integration - keep on GPU
            linear_combination = (weights[0] * color + 
                                weights[1] * brightness)
        
        # Final combination
        output = linear_combination + non_linear
        
        return output
    
    def _get_cached_weights(self, has_integrated: bool):
        """Get normalized weights - no caching during training to avoid gradient issues."""
        # Don't cache during training to avoid gradient graph conflicts
        if has_integrated:
            weights = F.softmax(torch.stack([
                self.color_weight, self.brightness_weight, self.integrated_weight
            ]), dim=0)
        else:
            weights = F.softmax(torch.stack([
                self.color_weight, self.brightness_weight
            ]), dim=0)
        
        return weights
    
    def invalidate_weight_cache(self):
        """Invalidate weight cache when parameters are updated."""
        self._weight_cache_valid = False
    
    def get_weights(self) -> dict:
        """Get current integration weights with GPU optimization."""
        # Use cached weights if available, but minimize CPU transfers
        if hasattr(self, 'integrated_weight'):
            weights = self._get_cached_weights(True)
            # Only transfer to CPU when actually needed for reporting
            with torch.no_grad():
                return {
                    'color': weights[0].detach().cpu().item(),
                    'brightness': weights[1].detach().cpu().item(),
                    'integrated': weights[2].detach().cpu().item() if len(weights) > 2 else 0.0
                }
        else:
            weights = self._get_cached_weights(False)
            with torch.no_grad():
                return {
                    'color': weights[0].detach().cpu().item(),
                    'brightness': weights[1].detach().cpu().item()
                }


class IntegrationStage(nn.Module):
    """GPU-optimized processing stage with separate pathways and optional integration."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 num_blocks: int, downsample: bool,
                 memory_efficient: bool = True):
        super().__init__()
        
        self.memory_efficient = memory_efficient
        self.gradient_checkpointing = False  # Can be enabled for memory savings
        
        stride = 2 if downsample else 1
        
        # GPU-optimized color pathway blocks
        self.color_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.color_blocks.append(
                    ResidualBlock(in_channels, out_channels, stride)
                )
            else:
                self.color_blocks.append(
                    ResidualBlock(out_channels, out_channels, 1)
                )
        
        # GPU-optimized brightness pathway blocks
        self.brightness_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.brightness_blocks.append(
                    ResidualBlock(in_channels, out_channels, stride)
                )
            else:
                self.brightness_blocks.append(
                    ResidualBlock(out_channels, out_channels, 1)
                )
        
        # Integration pathway blocks (if integrated features exist)
        self.integration_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.integration_blocks.append(
                    ResidualBlock(in_channels, out_channels, stride)
                )
            else:
                self.integration_blocks.append(
                    ResidualBlock(out_channels, out_channels, 1)
                )
    
    def forward(self, color: torch.Tensor, brightness: torch.Tensor,
                integrated: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """GPU-optimized forward pass through the stage."""
        
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency during training
            color = self._checkpointed_forward(self.color_blocks, color)
            brightness = self._checkpointed_forward(self.brightness_blocks, brightness)
            if integrated is not None:
                integrated = self._checkpointed_forward(self.integration_blocks, integrated)
        else:
            # Standard forward pass optimized for GPU
            color = self._forward_blocks(self.color_blocks, color)
            brightness = self._forward_blocks(self.brightness_blocks, brightness)
            if integrated is not None:
                integrated = self._forward_blocks(self.integration_blocks, integrated)
        
        return color, brightness, integrated
    
    def _forward_blocks(self, blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass through blocks."""
        for block in blocks:
            x = block(x)
        return x
    
    def _checkpointed_forward(self, blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward with gradient checkpointing."""
        def run_blocks(tensor):
            for block in blocks:
                tensor = block(tensor)
            return tensor
        
        # Use recommended use_reentrant=False for better memory efficiency
        return torch.utils.checkpoint.checkpoint(run_blocks, x, use_reentrant=False)


class AdaptiveIntegrationGate(nn.Module):
    """GPU-optimized adaptive gating mechanism for integration."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Optimized gate network with reduced parameters
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 3, channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 3, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Cache for gate computations
        self._gate_cache = None
        self._input_hash = None
    
    def forward(self, color: torch.Tensor, brightness: torch.Tensor,
                integrated: torch.Tensor) -> torch.Tensor:
        """Apply GPU-optimized adaptive gating."""
        # Memory-efficient concatenation
        combined = torch.cat([color, brightness, integrated], dim=1)
        
        # Compute gates with potential caching for repeated inputs
        gates = self.gate_network(combined)
        gates = gates.view(-1, 3, 1, 1)
        
        # GPU-optimized gated combination
        output = (gates[:, 0:1] * color + 
                 gates[:, 1:2] * brightness + 
                 gates[:, 2:3] * integrated)
        
        return output
"""Single-Output Multi-Weight Neural Network implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..base import BaseMultiWeightModel
from ..components.blocks import ConvBlock, ResidualBlock
from ..components.layers import PracticalMultiWeightLinear, AdaptiveMultiWeightLinear
from ..components.neurons import PracticalMultiWeightNeuron, AdaptiveMultiWeightNeuron
from .adaptive_fusion import SingleOutputStage, MultiWeightResidualBlock


class SingleOutputModel(BaseMultiWeightModel):
    """Single-output model with specialized weights for color and brightness."""
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 feature_extraction_method: str = 'hsv',
                 normalize_features: bool = True,
                 base_channels: int = 64,
                 depth: str = 'medium',
                 dropout_rate: float = 0.2,
                 adaptive_fusion: bool = False):
        
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.adaptive_fusion = adaptive_fusion
        
        # Architecture configurations
        self.depth_configs = {
            'shallow': {'blocks': [2, 2], 'channels': [64, 128]},
            'medium': {'blocks': [2, 2, 2], 'channels': [64, 128, 256]},
            'deep': {'blocks': [3, 4, 6, 3], 'channels': [64, 128, 256, 512]}
        }
        
        super().__init__(input_channels, num_classes,
                        feature_extraction_method, normalize_features)
    
    def _build_model(self):
        """Build the single-output multi-weight architecture."""
        config = self.depth_configs[self.depth]
        
        # Initial feature processing
        self.initial_conv = ConvBlock(3, self.base_channels)  # Process full input
        
        # Build multi-weight stages
        self.stages = nn.ModuleList()
        in_channels = self.base_channels
        
        for i, (out_channels, num_blocks) in enumerate(zip(config['channels'], config['blocks'])):
            stage = SingleOutputStage(
                in_channels, out_channels, num_blocks,
                downsample=(i > 0), adaptive=self.adaptive_fusion
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final multi-weight layers
        final_channels = config['channels'][-1]
        
        if self.adaptive_fusion:
            self.final_layer = AdaptiveMultiWeightLinear(
                final_channels, final_channels
            )
        else:
            self.final_layer = PracticalMultiWeightLinear(
                final_channels, final_channels
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels // 2, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through single-output model."""
        # Initial processing
        x = self.initial_conv(x)
        
        # Process through multi-weight stages
        for stage in self.stages:
            x = stage(x)
        
        # Global pooling
        x = self.global_pool(x).flatten(1)
        
        # Final multi-weight processing
        x = self.final_layer(x)
        
        # Classification
        output = self.classifier(x)
        
        return output
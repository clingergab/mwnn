"""Cross-Modal Neural Network implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..base import BaseMultiWeightModel
from ..components.blocks import ConvBlock, ResidualBlock
from ..components.neurons import CrossModalMultiWeightNeuron
from .cross_attention import CrossModalStage, CrossAttentionModule, CrossModalLinear


class CrossModalModel(BaseMultiWeightModel):
    """Model with cross-modal influence between color and brightness pathways."""
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 feature_extraction_method: str = 'hsv',
                 normalize_features: bool = True,
                 base_channels: int = 64,
                 depth: str = 'medium',
                 cross_influence: float = 0.1,
                 dropout_rate: float = 0.2):
        
        self.base_channels = base_channels
        self.depth = depth
        self.cross_influence = cross_influence
        self.dropout_rate = dropout_rate
        
        # Architecture configurations
        self.depth_configs = {
            'shallow': {'blocks': [2, 2], 'channels': [64, 128]},
            'medium': {'blocks': [2, 2, 2], 'channels': [64, 128, 256]},
            'deep': {'blocks': [3, 4, 6, 3], 'channels': [64, 128, 256, 512]}
        }
        
        super().__init__(input_channels, num_classes,
                        feature_extraction_method, normalize_features)
    
    def _build_model(self):
        """Build the cross-modal architecture."""
        config = self.depth_configs[self.depth]
        
        # Initial processing
        self.initial_color = ConvBlock(2, self.base_channels)
        self.initial_brightness = ConvBlock(1, self.base_channels)
        
        # Build cross-modal stages
        self.stages = nn.ModuleList()
        in_channels = self.base_channels
        
        for i, (out_channels, num_blocks) in enumerate(zip(config['channels'], config['blocks'])):
            stage = CrossModalStage(
                in_channels, out_channels, num_blocks,
                downsample=(i > 0), cross_influence=self.cross_influence
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        final_channels = config['channels'][-1]
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels * 2, final_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through cross-modal model."""
        # Extract features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Process through cross-modal stages
        for stage in self.stages:
            color, brightness = stage(color, brightness)
        
        # Global pooling
        color = self.global_pool(color).flatten(1)
        brightness = self.global_pool(brightness).flatten(1)
        
        # Concatenate and classify (no need for final cross attention on pooled features)
        combined = torch.cat([color, brightness], dim=1)
        output = self.classifier(combined)
        
        return output
    
    def get_cross_influence_weights(self) -> dict:
        """Get cross-influence weights for analysis."""
        weights = {}
        
        for i, stage in enumerate(self.stages):
            weights[f'stage_{i}'] = {
                'color_to_brightness': stage.get_cross_weights('cb'),
                'brightness_to_color': stage.get_cross_weights('bc')
            }
        
        return weights
    
    def get_pathway_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate outputs from color and brightness pathways."""
        # Extract features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Process through cross-modal stages
        for stage in self.stages:
            color, brightness = stage(color, brightness)
        
        # Global pooling
        color = self.global_pool(color).flatten(1)
        brightness = self.global_pool(brightness).flatten(1)
        
        return color, brightness
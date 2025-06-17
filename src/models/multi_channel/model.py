"""Multi-Channel Neural Network implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..base import BaseMultiWeightModel
from ..components.blocks import ConvBlock, ResidualBlock, FusionBlock, GlobalContextBlock
from ..components.neurons import MultiChannelNeuron


class MultiChannelModel(BaseMultiWeightModel):
    """Multi-Channel model with separate color and brightness pathways."""
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 feature_extraction_method: str = 'hsv',
                 normalize_features: bool = True,
                 base_channels: int = 64,
                 depth: str = 'medium',
                 dropout_rate: float = 0.2,
                 fusion_method: str = 'adaptive'):
        
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.fusion_method = fusion_method
        
        # Set architecture parameters based on depth
        self.depth_configs = {
            'shallow': {'blocks': [2, 2], 'channels': [64, 128]},
            'medium': {'blocks': [2, 2, 2], 'channels': [64, 128, 256]},
            'deep': {'blocks': [3, 4, 6, 3], 'channels': [64, 128, 256, 512]}
        }
        
        super().__init__(input_channels, num_classes, 
                        feature_extraction_method, normalize_features)
    
    def _build_model(self):
        """Build the multi-channel architecture."""
        config = self.depth_configs[self.depth]
        
        # Initial feature extraction
        self.initial_color = ConvBlock(2, self.base_channels)  # HS channels
        self.initial_brightness = ConvBlock(1, self.base_channels)  # V channel
        
        # Build color pathway
        self.color_pathway = self._build_pathway(
            self.base_channels, config['channels'], config['blocks'], 'color'
        )
        
        # Build brightness pathway
        self.brightness_pathway = self._build_pathway(
            self.base_channels, config['channels'], config['blocks'], 'brightness'
        )
        
        # Global pooling and context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.color_context = GlobalContextBlock(config['channels'][-1])
        self.brightness_context = GlobalContextBlock(config['channels'][-1])
        
        # Fusion layer
        final_channels = config['channels'][-1]
        self.fusion = FusionBlock(final_channels, self.fusion_method)
        
        # Determine fusion output size
        if self.fusion_method == 'concatenate':
            fusion_output_size = final_channels * 2
        else:
            fusion_output_size = final_channels
        
        # Classification head - input size depends on fusion method
        if self.fusion_method == 'concatenate':
            classifier_input_size = final_channels * 2
        else:
            classifier_input_size = final_channels
            
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(classifier_input_size, final_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels, self.num_classes)
        )
    
    def _build_pathway(self, in_channels: int, channel_list: list, 
                      block_list: list, pathway_name: str) -> nn.ModuleList:
        """Build a processing pathway (color or brightness)."""
        layers = nn.ModuleList()
        
        for i, (out_channels, num_blocks) in enumerate(zip(channel_list, block_list)):
            # First block might have stride 2 for downsampling
            stride = 2 if i > 0 else 1
            
            # Add residual blocks
            for j in range(num_blocks):
                if j == 0:
                    layers.append(
                        ResidualBlock(in_channels, out_channels, stride)
                    )
                else:
                    layers.append(
                        ResidualBlock(out_channels, out_channels, 1)
                    )
                in_channels = out_channels
        
        return layers
    
    def forward_pathway(self, x: torch.Tensor, pathway: nn.ModuleList) -> torch.Tensor:
        """Forward pass through a pathway."""
        for layer in pathway:
            x = layer(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multi-channel model."""
        # Extract color and brightness features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Process through pathways
        color = self.forward_pathway(color, self.color_pathway)
        brightness = self.forward_pathway(brightness, self.brightness_pathway)
        
        # Apply global context
        color = self.color_context(color)
        brightness = self.brightness_context(brightness)
        
        # Global pooling
        color = self.global_pool(color).flatten(1)
        brightness = self.global_pool(brightness).flatten(1)
        
        # Fuse features - handle based on fusion method
        if self.fusion_method == 'concatenate':
            fused = torch.cat([color, brightness], dim=1)
        elif self.fusion_method == 'add':
            fused = color + brightness
        elif self.fusion_method == 'adaptive':
            # Use the fusion block for adaptive fusion
            # Need to add spatial dimensions back for fusion block
            color_4d = color.unsqueeze(-1).unsqueeze(-1)
            brightness_4d = brightness.unsqueeze(-1).unsqueeze(-1)
            fused = self.fusion(color_4d, brightness_4d)
            fused = fused.squeeze(-1).squeeze(-1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Classification
        output = self.classifier(fused)
        
        return output
    
    def get_pathway_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate outputs from color and brightness pathways."""
        # Extract features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Process through pathways
        color = self.forward_pathway(color, self.color_pathway)
        brightness = self.forward_pathway(brightness, self.brightness_pathway)
        
        # Apply global context
        color = self.color_context(color)
        brightness = self.brightness_context(brightness)
        
        # Global pooling
        color = self.global_pool(color).flatten(1)
        brightness = self.global_pool(brightness).flatten(1)
        
        return color, brightness
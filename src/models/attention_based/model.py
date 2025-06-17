"""Attention-Based Multi-Weight Neural Network implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..base import BaseMultiWeightModel
from ..components.blocks import ConvBlock, ResidualBlock
from ..components.neurons import AttentionMultiWeightNeuron
from .attention_mechanisms import CrossModalAttention, GlobalAttentionPooling, AttentionStage


class AttentionBasedModel(BaseMultiWeightModel):
    """Model using attention mechanisms for cross-modal processing."""
    
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 feature_extraction_method: str = 'hsv',
                 normalize_features: bool = True,
                 base_channels: int = 64,
                 depth: str = 'medium',
                 attention_dim: int = 64,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.2,
                 attention_dropout: float = 0.1):
        
        self.base_channels = base_channels
        self.depth = depth
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        
        # Architecture configurations
        self.depth_configs = {
            'shallow': {'blocks': [2, 2], 'channels': [64, 128]},
            'medium': {'blocks': [2, 2, 2], 'channels': [64, 128, 256]},
            'deep': {'blocks': [3, 4, 6, 3], 'channels': [64, 128, 256, 512]}
        }
        
        super().__init__(input_channels, num_classes,
                        feature_extraction_method, normalize_features)
    
    def _build_model(self):
        """Build the attention-based architecture."""
        config = self.depth_configs[self.depth]
        
        # Initial processing
        self.initial_color = ConvBlock(2, self.base_channels)
        self.initial_brightness = ConvBlock(1, self.base_channels)
        
        # Build attention-based stages
        self.stages = nn.ModuleList()
        in_channels = self.base_channels
        
        for i, (out_channels, num_blocks) in enumerate(zip(config['channels'], config['blocks'])):
            stage = AttentionStage(
                in_channels, out_channels, num_blocks,
                downsample=(i > 0), 
                attention_dim=self.attention_dim,
                num_heads=self.num_attention_heads,
                dropout=self.attention_dropout
            )
            self.stages.append(stage)
            in_channels = out_channels
        
        # Global attention pooling
        self.color_global_pool = GlobalAttentionPooling(in_channels)
        self.brightness_global_pool = GlobalAttentionPooling(in_channels)
        
        # Final cross-modal attention
        self.final_attention = CrossModalAttention(
            in_channels, self.attention_dim, 
            self.num_attention_heads, self.attention_dropout
        )
        
        # Classification head with self-attention
        self.pre_classifier_attention = nn.MultiheadAttention(
            in_channels * 2, self.num_attention_heads, 
            dropout=self.attention_dropout, batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_channels, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention-based model."""
        # Extract features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Process through attention stages
        for stage in self.stages:
            color, brightness = stage(color, brightness)
        
        # Global attention pooling
        color = self.color_global_pool(color)
        brightness = self.brightness_global_pool(brightness)
        
        # Final cross-modal attention on pooled features
        color_attended, brightness_attended = self.final_attention(color, brightness)
        
        # Concatenate features
        combined = torch.cat([color_attended, brightness_attended], dim=1)
        
        # Self-attention before classification
        combined = combined.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.pre_classifier_attention(combined, combined, combined)
        attended = attended.squeeze(1)
        
        # Classification
        output = self.classifier(attended)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """Get attention maps for visualization."""
        attention_maps = {}
        
        # Extract features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Collect attention maps from each stage
        for i, stage in enumerate(self.stages):
            color, brightness = stage(color, brightness)
            attention_maps[f'stage_{i}'] = stage.get_last_attention_maps()
        
        # Get final attention maps
        color_pooled = self.color_global_pool(color)
        brightness_pooled = self.brightness_global_pool(brightness)
        
        _, _ = self.final_attention(color_pooled, brightness_pooled)
        attention_maps['final'] = self.final_attention.get_last_attention_weights()
        
        return attention_maps
    
    def get_pathway_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate outputs from color and brightness pathways."""
        # Extract features
        color, brightness = self.extract_features(x)
        
        # Initial processing
        color = self.initial_color(color)
        brightness = self.initial_brightness(brightness)
        
        # Process through attention stages
        for stage in self.stages:
            color, brightness = stage(color, brightness)
        
        # Global attention pooling
        color = self.color_global_pool(color)
        brightness = self.brightness_global_pool(brightness)
        
        return color, brightness
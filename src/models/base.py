"""Base model class for all Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging

try:
    from ..preprocessing.color_extractors import FeatureExtractor
except ImportError:
    from preprocessing.color_extractors import FeatureExtractor


logger = logging.getLogger(__name__)


class BaseMultiWeightModel(nn.Module, ABC):
    """Abstract base class for all multi-weight models."""
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 10,
                 feature_extraction_method: str = 'hsv',
                 normalize_features: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feature_extraction_method = feature_extraction_method
        self.normalize_features = normalize_features
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            method=feature_extraction_method,
            normalize=normalize_features
        )
        
        # Model name for logging
        self.model_name = self.__class__.__name__
        
        # Initialize architecture
        self._build_model()
        
        # Log model info
        self._log_model_info()
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Must be implemented by subclasses."""
        pass
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color and brightness features from input."""
        return self.feature_extractor(x)
    
    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Feature extraction method: {self.feature_extraction_method}")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters by category."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by type
        color_params = sum(p.numel() for n, p in self.named_parameters() 
                          if 'color' in n and p.requires_grad)
        brightness_params = sum(p.numel() for n, p in self.named_parameters() 
                               if 'brightness' in n and p.requires_grad)
        integration_params = sum(p.numel() for n, p in self.named_parameters() 
                                if 'integration' in n and p.requires_grad)
        cross_params = sum(p.numel() for n, p in self.named_parameters() 
                          if 'cross' in n and p.requires_grad)
        attention_params = sum(p.numel() for n, p in self.named_parameters() 
                              if 'attention' in n and p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'color': color_params,
            'brightness': brightness_params,
            'integration': integration_params,
            'cross_modal': cross_params,
            'attention': attention_params,
            'other': trainable_params - color_params - brightness_params - integration_params - cross_params - attention_params
        }
    
    def freeze_color_weights(self):
        """Freeze all color-related weights."""
        for name, param in self.named_parameters():
            if 'color' in name:
                param.requires_grad = False
                logger.info(f"Frozen: {name}")
    
    def freeze_brightness_weights(self):
        """Freeze all brightness-related weights."""
        for name, param in self.named_parameters():
            if 'brightness' in name:
                param.requires_grad = False
                logger.info(f"Frozen: {name}")
    
    def unfreeze_all_weights(self):
        """Unfreeze all weights."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All weights unfrozen")
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded pretrained weights from {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       additional_info: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_parameters': self.get_num_parameters(),
            'feature_extraction_method': self.feature_extraction_method,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def get_pathway_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get separate outputs from color and brightness pathways.
        
        This is an optional method that models with separate pathways can override.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (color_features, brightness_features) if applicable
            
        Raises:
            NotImplementedError: If the model doesn't have separate pathways
        """
        raise NotImplementedError(
            f"{self.model_name} does not have separate color/brightness pathways. "
            "This method is only available for models like MultiChannelModel, "
            "ContinuousIntegrationModel, CrossModalModel, and AttentionBasedModel."
        )
"""
Configuration management for ImageNet preprocessing pipeline.

This module provides configuration classes and utilities for managing ImageNet-1K
preprocessing settings, including preset configurations for different use cases.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ImageNetPreprocessingConfig:
    """
    Configuration class for ImageNet preprocessing pipeline.
    
    This dataclass holds all configuration parameters for ImageNet data loading,
    preprocessing, and augmentation for Multi-Weight Neural Networks.
    """
    
    # Dataset paths
    data_dir: str
    devkit_dir: str
    
    # Feature extraction
    feature_method: str = 'rgb_luminance'  # Default to RGB+Luminance approach
    input_size: int = 224
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    
    # Train/validation split
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Subset loading for development
    load_subset: Optional[int] = None
    
    # Augmentation settings
    augment: bool = True
    augment_features: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    
    # Normalization
    normalize_features: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Caching
    cache_labels: bool = True
    cache_features: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not 0 <= self.train_split <= 1:
            raise ValueError("train_split must be between 0 and 1")
            
        if not 0 <= self.val_split <= 1:
            raise ValueError("val_split must be between 0 and 1")
            
        if abs(self.train_split + self.val_split - 1.0) > 1e-6:
            raise ValueError("train_split + val_split must equal 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_dir': self.data_dir,
            'devkit_dir': self.devkit_dir,
            'feature_method': self.feature_method,
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'drop_last': self.drop_last,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'load_subset': self.load_subset,
            'augment': self.augment,
            'augment_features': self.augment_features,
            'horizontal_flip_prob': self.horizontal_flip_prob,
            'color_jitter_brightness': self.color_jitter_brightness,
            'color_jitter_contrast': self.color_jitter_contrast,
            'color_jitter_saturation': self.color_jitter_saturation,
            'color_jitter_hue': self.color_jitter_hue,
            'normalize_features': self.normalize_features,
            'mean': self.mean,
            'std': self.std,
            'cache_labels': self.cache_labels,
            'cache_features': self.cache_features
        }
    
    def to_dataset_kwargs(self) -> Dict[str, Any]:
        """Get dataset creation keyword arguments."""
        return {
            'feature_method': self.feature_method,
            'augment': self.augment,
            'load_subset': self.load_subset,
            'cache_labels': self.cache_labels,
            'input_size': self.input_size
        }
    
    def to_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get DataLoader creation keyword arguments."""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'drop_last': self.drop_last
        }
    
    def save_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'ImageNetPreprocessingConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


def get_preset_config(preset: str, data_dir: str, devkit_dir: str) -> ImageNetPreprocessingConfig:
    """
    Get a preset configuration for common use cases.
    
    Args:
        preset: Configuration preset ('development', 'training', 'evaluation', 'research')
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to ImageNet devkit directory
    
    Returns:
        ImageNetPreprocessingConfig: Configured instance
    """
    
    if preset == 'development':
        return ImageNetPreprocessingConfig(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            feature_method='rgb_luminance',
            batch_size=16,
            num_workers=2,
            load_subset=100,
            augment=False,
            cache_labels=True,
            cache_features=False
        )
    
    elif preset == 'training':
        return ImageNetPreprocessingConfig(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            feature_method='rgb_luminance',
            batch_size=32,
            num_workers=4,
            augment=True,
            augment_features=True,
            cache_labels=True,
            cache_features=True,
            persistent_workers=True
        )
    
    elif preset == 'evaluation':
        return ImageNetPreprocessingConfig(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            feature_method='rgb_luminance',
            batch_size=64,
            num_workers=4,
            augment=False,
            augment_features=False,
            cache_labels=True,
            cache_features=True,
            drop_last=False
        )
    
    elif preset == 'research':
        return ImageNetPreprocessingConfig(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            feature_method='rgb_luminance',
            batch_size=32,
            num_workers=6,
            input_size=256,
            augment=True,
            augment_features=True,
            cache_labels=True,
            cache_features=True,
            horizontal_flip_prob=0.5,
            color_jitter_brightness=0.3,
            color_jitter_contrast=0.3,
            color_jitter_saturation=0.3,
            color_jitter_hue=0.15
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset}. Choose from: development, training, evaluation, research")


def create_default_configs(output_dir: str = "configs/preprocessing"):
    """
    Create default configuration files for all presets.
    
    Args:
        output_dir: Directory to save configuration files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Placeholder paths - users should update these
    data_dir = "/path/to/your/ImageNet-1K"
    devkit_dir = "/path/to/your/ImageNet-1K/ILSVRC2013_devkit"
    
    presets = ['development', 'training', 'evaluation', 'research']
    
    for preset in presets:
        config = get_preset_config(preset, data_dir, devkit_dir)
        config.save_yaml(output_path / f"imagenet_{preset}.yaml")
        print(f"Created {preset} configuration: {output_path / f'imagenet_{preset}.yaml'}")
    
    # Also create a template
    template_config = ImageNetPreprocessingConfig(
        data_dir=data_dir,
        devkit_dir=devkit_dir
    )
    template_config.save_yaml(output_path / "imagenet_template.yaml")
    print(f"Created template configuration: {output_path / 'imagenet_template.yaml'}")
    
    print(f"\nConfiguration files created in: {output_path}")
    print("Update the data_dir and devkit_dir paths in each file to match your setup.")
"""Data loaders for Multi-Weight Neural Networks."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Callable, List
import numpy as np
from pathlib import Path
import os
from PIL import Image

from .color_extractors import FeatureExtractor, AugmentedFeatureExtractor


class MWNNDataset(Dataset):
    """Base dataset class for Multi-Weight Neural Networks."""
    
    def __init__(self, 
                 base_dataset: Dataset,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 feature_method: str = 'hsv',
                 augment: bool = False):
        """
        Args:
            base_dataset: Base PyTorch dataset (e.g., CIFAR10, ImageNet)
            feature_extractor: Feature extractor instance
            feature_method: Method for extracting color/brightness features
            augment: Whether to apply augmentation
        """
        self.base_dataset = base_dataset
        
        if feature_extractor is None:
            if augment:
                self.feature_extractor = AugmentedFeatureExtractor(
                    method=feature_method,
                    normalize=True,
                    augment_color=True,
                    augment_brightness=True
                )
            else:
                self.feature_extractor = FeatureExtractor(
                    method=feature_method,
                    normalize=True
                )
        else:
            self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get image and label from base dataset
        image, label = self.base_dataset[idx]
        
        # Extract color and brightness features
        # Note: In practice, feature extraction happens in the model
        # This is for datasets that pre-separate features
        
        return image, label


class MultiModalDataset(Dataset):
    """Dataset for multi-modal inputs (e.g., RGB + Depth/NIR)."""
    
    def __init__(self,
                 rgb_dir: str,
                 auxiliary_dir: str,
                 labels_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 auxiliary_transform: Optional[Callable] = None,
                 extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')):
        """
        Args:
            rgb_dir: Directory containing RGB images
            auxiliary_dir: Directory containing auxiliary modality (depth, NIR, etc.)
            labels_file: Optional CSV file with labels
            transform: Transform for RGB images
            auxiliary_transform: Transform for auxiliary modality
            extensions: Valid file extensions
        """
        self.rgb_dir = Path(rgb_dir)
        self.auxiliary_dir = Path(auxiliary_dir)
        self.transform = transform
        self.auxiliary_transform = auxiliary_transform
        
        # Find all valid image files
        self.samples = []
        self.labels = {}
        
        # Load labels if provided
        if labels_file:
            import pandas as pd
            df = pd.read_csv(labels_file)
            self.labels = dict(zip(df['filename'], df['label']))
        
        # Scan directories
        for rgb_path in sorted(self.rgb_dir.iterdir()):
            if rgb_path.suffix.lower() in extensions:
                aux_path = self.auxiliary_dir / rgb_path.name
                if aux_path.exists():
                    label = self.labels.get(rgb_path.stem, 0)
                    self.samples.append((rgb_path, aux_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        rgb_path, aux_path, label = self.samples[idx]
        
        # Load images
        rgb_image = Image.open(rgb_path).convert('RGB')
        aux_image = Image.open(aux_path)
        
        # Apply transforms
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        if self.auxiliary_transform:
            aux_image = self.auxiliary_transform(aux_image)
        elif self.transform:
            # Use same transform if not specified
            aux_image = self.transform(aux_image)
        
        return (rgb_image, aux_image), label


class SyntheticMWNNDataset(Dataset):
    """Synthetic dataset for testing MWNN architectures."""
    
    def __init__(self,
                 num_samples: int = 1000,
                 image_size: Tuple[int, int] = (32, 32),
                 num_classes: int = 10,
                 color_noise: float = 0.1,
                 brightness_noise: float = 0.1):
        """
        Create synthetic dataset with controlled color/brightness variations.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images
            num_classes: Number of classes
            color_noise: Amount of color noise
            brightness_noise: Amount of brightness noise
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            # Generate class-specific pattern
            label = i % num_classes
            
            # Base pattern (class-specific)
            pattern = self._generate_pattern(label)
            
            # Add color variation
            color = self._generate_color(label, color_noise)
            
            # Add brightness variation
            brightness = self._generate_brightness(label, brightness_noise)
            
            # Combine to create image
            image = pattern * color * brightness
            
            self.images.append(torch.FloatTensor(image))
            self.labels.append(label)
    
    def _generate_pattern(self, label: int) -> np.ndarray:
        """Generate class-specific pattern."""
        h, w = self.image_size
        pattern = np.zeros((3, h, w))
        
        # Simple patterns based on class
        if label == 0:  # Horizontal stripes
            pattern[:, ::4, :] = 1
        elif label == 1:  # Vertical stripes
            pattern[:, :, ::4] = 1
        elif label == 2:  # Diagonal stripes
            for i in range(h):
                for j in range(w):
                    if (i + j) % 8 < 4:
                        pattern[:, i, j] = 1
        elif label == 3:  # Circles
            center = (h // 2, w // 2)
            for i in range(h):
                for j in range(w):
                    if np.sqrt((i - center[0])**2 + (j - center[1])**2) < min(h, w) // 4:
                        pattern[:, i, j] = 1
        else:  # Random patterns for other classes
            np.random.seed(label)
            pattern = np.random.rand(3, h, w) > 0.5
        
        return pattern
    
    def _generate_color(self, label: int, noise: float) -> np.ndarray:
        """Generate class-specific color with noise."""
        # Base colors for each class
        base_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.5, 0.5, 0.5],  # Gray
            [1.0, 1.0, 1.0],  # White
        ]
        
        color = np.array(base_colors[label % len(base_colors)])
        color = color + np.random.randn(3) * noise
        color = np.clip(color, 0, 1)
        
        return color.reshape(3, 1, 1)
    
    def _generate_brightness(self, label: int, noise: float) -> float:
        """Generate class-specific brightness with noise."""
        base_brightness = 0.5 + 0.05 * label
        brightness = base_brightness + np.random.randn() * noise
        return np.clip(brightness, 0.1, 1.0)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def create_mwnn_dataset(base_dataset: Dataset,
                       feature_method: str = 'hsv',
                       augment: bool = True) -> MWNNDataset:
    """Create MWNN dataset from a base dataset."""
    return MWNNDataset(
        base_dataset=base_dataset,
        feature_method=feature_method,
        augment=augment
    )


def get_data_loader(dataset: Dataset,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4,
                   pin_memory: bool = True,
                   drop_last: bool = False) -> DataLoader:
    """Create a DataLoader with optimal settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )
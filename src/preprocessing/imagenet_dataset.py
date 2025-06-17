"""
ImageNet-1K Dataset preprocessing and loading utilities for Multi-Weight Neural Networks.

This module provides comprehensive preprocessing and loading utilities for the ImageNet-1K
validation dataset, specifically designed to work with Multi-Weight Neural Networks (MWNNs).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Dict, Callable
from pathlib import Path
from PIL import Image
import pickle
from collections import defaultdict
import logging

from .color_extractors import FeatureExtractor, AugmentedFeatureExtractor

logger = logging.getLogger(__name__)


class ImageNetMWNNDataset(Dataset):
    """
    ImageNet-1K dataset optimized for Multi-Weight Neural Networks.
    
    This dataset class handles the ImageNet-1K validation set with proper label mapping
    and supports the multi-weight neural network architecture with color/brightness features.
    """
    
    def __init__(self,
                 data_dir: str,
                 devkit_dir: str,
                 split: str = 'val',
                 transform: Optional[Callable] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 feature_method: str = 'hsv',
                 augment: bool = False,
                 num_classes: int = 1000,
                 load_subset: Optional[int] = None,
                 cache_labels: bool = True):
        """
        Initialize ImageNet dataset for MWNN.
        
        Args:
            data_dir: Path to directory containing val_images folder
            devkit_dir: Path to ILSVRC2013_devkit directory
            split: Dataset split ('val' for validation)
            transform: Image transformations
            feature_extractor: Custom feature extractor
            feature_method: Method for extracting color/brightness features ('hsv', 'lab', etc.)
            augment: Whether to apply augmentation during feature extraction
            num_classes: Number of classes (1000 for full ImageNet)
            load_subset: If specified, only load this many samples (for testing)
            cache_labels: Whether to cache label mappings
        """
        self.data_dir = Path(data_dir)
        self.devkit_dir = Path(devkit_dir)
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.load_subset = load_subset
        self.cache_labels = cache_labels
        
        # Set up feature extractor
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
        
        # Initialize paths
        self.images_dir = self.data_dir / f"{split}_images"
        if split == 'val':
            self.ground_truth_file = self.devkit_dir / "data" / "ILSVRC2013_clsloc_validation_ground_truth.txt"
        else:
            self.ground_truth_file = self.devkit_dir / "data" / f"ILSVRC2013_clsloc_{split}_ground_truth.txt"
        
        # Load class mappings and labels
        self._load_class_mappings()
        self._load_image_labels()
        
        # Build file list
        self._build_file_list()
        
        logger.info(f"Loaded ImageNet {split} dataset with {len(self.samples)} samples")
    
    def _load_class_mappings(self):
        """Load class mappings from devkit metadata."""
        try:
            # Try to load cached mappings first
            cache_file = self.devkit_dir / "class_mappings.pkl"
            if self.cache_labels and cache_file.exists():
                with open(cache_file, 'rb') as f:
                    mappings = pickle.load(f)
                    self.synset_to_idx = mappings['synset_to_idx']
                    self.idx_to_synset = mappings['idx_to_synset']
                    self.synset_to_name = mappings['synset_to_name']
                    logger.info("Loaded cached class mappings")
                    return
            
            # Load from MATLAB metadata (if available)
            meta_file = self.devkit_dir / "data" / "meta_clsloc.mat"
            if meta_file.exists():
                try:
                    import scipy.io
                    meta = scipy.io.loadmat(str(meta_file))
                    
                    # Extract synsets and class names
                    synsets = meta['synsets']
                    self.synset_to_idx = {}
                    self.idx_to_synset = {}
                    self.synset_to_name = {}
                    
                    for i, synset_data in enumerate(synsets[0]):
                        synset_id = synset_data[1][0]  # WNID
                        class_name = synset_data[2][0]  # Name
                        
                        self.synset_to_idx[synset_id] = i
                        self.idx_to_synset[i] = synset_id
                        self.synset_to_name[synset_id] = class_name
                    
                    logger.info(f"Loaded {len(self.synset_to_idx)} class mappings from meta_clsloc.mat")
                
                except ImportError:
                    logger.warning("scipy not available, using fallback class mapping")
                    self._create_fallback_mappings()
                except Exception as e:
                    logger.warning(f"Error loading meta_clsloc.mat: {e}, using fallback")
                    self._create_fallback_mappings()
            else:
                logger.warning("meta_clsloc.mat not found, using fallback class mapping")
                self._create_fallback_mappings()
            
            # Cache the mappings
            if self.cache_labels:
                mappings = {
                    'synset_to_idx': self.synset_to_idx,
                    'idx_to_synset': self.idx_to_synset,
                    'synset_to_name': self.synset_to_name
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(mappings, f)
                logger.info("Cached class mappings")
        
        except Exception as e:
            logger.error(f"Error loading class mappings: {e}")
            self._create_fallback_mappings()
    
    def _create_fallback_mappings(self):
        """Create fallback class mappings when metadata is not available."""
        # Create basic mappings for 1000 classes
        self.synset_to_idx = {}
        self.idx_to_synset = {}
        self.synset_to_name = {}
        
        # Generate placeholder synset IDs and names
        for i in range(self.num_classes):
            synset_id = f"n{i:08d}"
            class_name = f"class_{i:03d}"
            
            self.synset_to_idx[synset_id] = i
            self.idx_to_synset[i] = synset_id
            self.synset_to_name[synset_id] = class_name
        
        logger.info(f"Created fallback mappings for {self.num_classes} classes")
    
    def _load_image_labels(self):
        """Load image labels from ground truth file."""
        try:
            with open(self.ground_truth_file, 'r') as f:
                # Each line contains the class index (0-indexed in the file, but 1-indexed in practice)
                self.image_labels = [int(line.strip()) - 1 for line in f.readlines()]
            
            logger.info(f"Loaded {len(self.image_labels)} image labels")
            
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {self.ground_truth_file}")
            # Create dummy labels
            self.image_labels = list(range(self.num_classes)) * (50000 // self.num_classes)
            self.image_labels = self.image_labels[:50000]
            logger.warning("Using dummy labels")
        
        except Exception as e:
            logger.error(f"Error loading image labels: {e}")
            self.image_labels = [0] * 50000
    
    def _build_file_list(self):
        """Build list of image files with their labels."""
        self.samples = []
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Get all JPEG files
        image_files = sorted([f for f in self.images_dir.iterdir() 
                             if f.suffix.lower() in ['.jpg', '.jpeg']])
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {self.images_dir}")
        
        # Match files with labels
        for idx, image_path in enumerate(image_files):
            if idx < len(self.image_labels):
                label = self.image_labels[idx]
                
                # Validate label range
                if 0 <= label < self.num_classes:
                    self.samples.append((image_path, label))
                else:
                    logger.warning(f"Invalid label {label} for image {image_path.name}")
            
            # Apply subset limit if specified
            if self.load_subset and len(self.samples) >= self.load_subset:
                break
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found")
        
        logger.info(f"Built file list with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        image_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Convert to tensor if not already
            if not isinstance(image, torch.Tensor):
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
            
            return image, label
        
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                # Try to infer expected size from transform
                dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = self.transform(dummy_image)
            else:
                image = torch.zeros(3, 224, 224)
            
            return image, label
    
    def get_class_name(self, label: int) -> str:
        """Get human-readable class name for a label."""
        if label < 0 or label >= self.num_classes:
            return f"unknown_{label}"
        
        synset_id = self.idx_to_synset.get(label, f"n{label:08d}")
        return self.synset_to_name.get(synset_id, f"class_{label}")
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in the dataset."""
        distribution = defaultdict(int)
        for _, label in self.samples:
            distribution[label] += 1
        return dict(distribution)


class ImageNetRGBLuminanceDataset(ImageNetMWNNDataset):
    """
    ImageNet dataset that returns 4-channel RGB+Luminance format.
    
    This dataset class implements the new MWNN design approach where:
    - Channels 0-2: Original RGB data (preserved)
    - Channel 3: ITU-R BT.709 luminance
    
    This enables separate color and brightness processing pathways while
    maintaining zero information loss from the original RGB data.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with RGB+Luminance as default method."""
        # Force feature_method to rgb_luminance
        kwargs['feature_method'] = 'rgb_luminance'
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        """Get a sample with 4-channel RGB+Luminance format."""
        # Get the standard RGB image and label
        image, label = super().__getitem__(idx)
        
        # Convert RGB to RGB+Luminance format
        if image.dim() == 3 and image.size(0) == 3:  # (3, H, W)
            image = image.unsqueeze(0)  # Add batch dimension: (1, 3, H, W)
            
            # Import the utility function
            from .color_extractors import rgb_to_rgb_luminance
            
            # Convert to 4-channel format
            image_4ch = rgb_to_rgb_luminance(image)
            
            # Remove batch dimension: (4, H, W)
            image = image_4ch.squeeze(0)
        
        return image, label
    
    def get_color_brightness_channels(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract color and brightness channels from the 4-channel format.
        
        Args:
            image: 4-channel image tensor (4, H, W) or (B, 4, H, W)
            
        Returns:
            color: RGB channels for color pathway
            brightness: Luminance channel for brightness pathway
        """
        from .color_extractors import extract_color_brightness_from_rgb_luminance
        
        if image.dim() == 3:  # (4, H, W)
            image = image.unsqueeze(0)  # Add batch dimension
            color, brightness = extract_color_brightness_from_rgb_luminance(image)
            return color.squeeze(0), brightness.squeeze(0)  # Remove batch dimension
        elif image.dim() == 4:  # (B, 4, H, W)
            return extract_color_brightness_from_rgb_luminance(image)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")


def get_imagenet_transforms(input_size: int = 224,
                           mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                           std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                           augment: bool = False) -> transforms.Compose:
    """
    Get standard ImageNet preprocessing transforms.
    
    Args:
        input_size: Target input size for the model
        mean: ImageNet mean values for normalization
        std: ImageNet std values for normalization  
        augment: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    if augment:
        transform_list = [
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    else:
        transform_list = [
            transforms.Resize(int(input_size * 1.143)),  # 256 for 224
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    
    return transforms.Compose(transform_list)


def create_imagenet_mwnn_dataset(data_dir: str,
                                devkit_dir: str,
                                split: str = 'val',
                                input_size: int = 224,
                                feature_method: str = 'hsv',
                                augment: bool = False,
                                load_subset: Optional[int] = None) -> ImageNetMWNNDataset:
    """
    Create ImageNet dataset optimized for MWNN training/evaluation.
    
    Args:
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to ILSVRC2013_devkit directory
        split: Dataset split ('val')
        input_size: Input image size
        feature_method: Color feature extraction method
        augment: Whether to apply data augmentation
        load_subset: Number of samples to load (None for all)
        
    Returns:
        ImageNet dataset for MWNN
    """
    # Get appropriate transforms
    transform = get_imagenet_transforms(
        input_size=input_size,
        augment=augment
    )
    
    # Create dataset
    dataset = ImageNetMWNNDataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        split=split,
        transform=transform,
        feature_method=feature_method,
        augment=augment,
        load_subset=load_subset
    )
    
    return dataset


def create_imagenet_rgb_luminance_dataset(data_dir: str,
                                         devkit_dir: str,
                                         split: str = 'val',
                                         input_size: int = 224,
                                         augment: bool = False,
                                         load_subset: Optional[int] = None,
                                         **kwargs) -> ImageNetRGBLuminanceDataset:
    """
    Create ImageNet dataset with RGB+Luminance format.
    
    This function creates a dataset that returns 4-channel tensors with:
    - Channels 0-2: Original RGB data
    - Channel 3: ITU-R BT.709 luminance
    
    Args:
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to devkit directory  
        split: Dataset split
        input_size: Target image size
        augment: Whether to apply data augmentation
        load_subset: Number of samples to load (None for all)
        **kwargs: Additional arguments
        
    Returns:
        ImageNet dataset with 4-channel RGB+Luminance format
    """
    # Get transforms
    transforms = get_imagenet_transforms(
        input_size=input_size,
        augment=augment
    )
    
    # Create dataset
    dataset = ImageNetRGBLuminanceDataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        split=split,
        transform=transforms,
        augment=augment,
        load_subset=load_subset,
        **kwargs
    )
    
    return dataset


def create_imagenet_dataloaders(data_dir: str,
                               devkit_dir: str,
                               batch_size: int = 32,
                               input_size: int = 224,
                               feature_method: str = 'hsv',
                               num_workers: int = 4,
                               load_subset: Optional[int] = None,
                               val_split: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for ImageNet.
    
    Note: This creates a train/val split from the validation set since we only have
    the validation set available.
    
    Args:
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to ILSVRC2013_devkit directory
        batch_size: Batch size for dataloaders
        input_size: Input image size
        feature_method: Color feature extraction method
        num_workers: Number of worker processes
        load_subset: Number of samples to load (None for all)
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        split='val',
        input_size=input_size,
        feature_method=feature_method,
        augment=False,  # We'll handle augmentation separately
        load_subset=load_subset
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create augmented version for training
    train_dataset_aug = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        split='val',
        input_size=input_size,
        feature_method=feature_method,
        augment=True,
        load_subset=load_subset
    )
    
    # Apply the same split to augmented dataset
    train_dataset_aug, _ = torch.utils.data.random_split(
        train_dataset_aug,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


def create_imagenet_rgb_luminance_dataloaders(data_dir: str,
                                             devkit_dir: str,
                                             batch_size: int = 32,
                                             input_size: int = 224,
                                             num_workers: int = 4,
                                             val_split: float = 0.1,
                                             load_subset: Optional[int] = None,
                                             **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for RGB+Luminance format ImageNet data.
    
    Returns train and validation DataLoaders that provide 4-channel tensors
    with RGB data in channels 0-2 and luminance in channel 3.
    
    Args:
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to devkit directory
        batch_size: Batch size for dataloaders
        input_size: Target image size
        num_workers: Number of worker processes
        val_split: Fraction of data to use for validation
        load_subset: Optional subset size for testing
        **kwargs: Additional arguments
        
    Returns:
        (train_loader, val_loader): DataLoaders with 4-channel RGB+Luminance data
    """
    # Create full dataset without augmentation
    full_dataset = create_imagenet_rgb_luminance_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        input_size=input_size,
        augment=False,
        load_subset=load_subset,
        **kwargs
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split) if val_split > 0 else 0
    train_size = total_size - val_size
    
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        val_dataset = full_dataset  # Use same dataset for evaluation
    
    # Create augmented version for training
    train_dataset_aug = create_imagenet_rgb_luminance_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        input_size=input_size,
        augment=True,
        load_subset=load_subset,
        **kwargs
    )
    
    # Apply the same split to augmented dataset if needed
    if val_size > 0:
        train_dataset_aug, _ = torch.utils.data.random_split(
            train_dataset_aug,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader


def create_imagenet_separate_pathway_dataloaders(data_dir: str,
                                                devkit_dir: str,
                                                batch_size: int = 32,
                                                input_size: int = 224,
                                                num_workers: int = 4,
                                                val_split: float = 0.1,
                                                load_subset: Optional[int] = None,
                                                **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders that return separate RGB and brightness tensors.
    
    Returns train and validation DataLoaders that provide separate tensors:
    - RGB data: (B, 3, H, W) 
    - Brightness data: (B, 1, H, W)
    
    This is optimized for Multi-Weight Neural Networks with separate pathways.
    
    Args:
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to devkit directory
        batch_size: Batch size for dataloaders
        input_size: Target image size
        num_workers: Number of worker processes
        val_split: Fraction of data to use for validation
        load_subset: Optional subset size for testing
        **kwargs: Additional arguments
        
    Returns:
        (train_loader, val_loader): DataLoaders returning (rgb_tensor, brightness_tensor, labels)
    """
    # First get the RGB+Luminance dataloaders
    train_loader_4ch, val_loader_4ch = create_imagenet_rgb_luminance_dataloaders(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=num_workers,
        val_split=val_split,
        load_subset=load_subset,
        **kwargs
    )
    
    class SeparatePathwayWrapper:
        """Wrapper to split 4-channel data into separate RGB and brightness tensors."""
        
        def __init__(self, dataloader):
            self.dataloader = dataloader
            
        def __iter__(self):
            for batch_4ch, labels in self.dataloader:
                # Split 4-channel tensor into RGB (channels 0-2) and brightness (channel 3)
                rgb_data = batch_4ch[:, :3, :, :]      # Shape: (B, 3, H, W)
                brightness_data = batch_4ch[:, 3:4, :, :] # Shape: (B, 1, H, W)
                yield (rgb_data, brightness_data), labels
                
        def __len__(self):
            return len(self.dataloader)
            
        @property
        def dataset(self):
            return self.dataloader.dataset
    
    # Wrap the dataloaders
    train_loader = SeparatePathwayWrapper(train_loader_4ch)
    val_loader = SeparatePathwayWrapper(val_loader_4ch)
    
    return train_loader, val_loader


def analyze_imagenet_dataset(data_dir: str, devkit_dir: str, sample_size: int = 1000):
    """
    Analyze the ImageNet dataset and provide statistics.
    
    Args:
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to ILSVRC2013_devkit directory
        sample_size: Number of samples to analyze
    """
    print("Analyzing ImageNet-1K Dataset...")
    print("=" * 50)
    
    # Create dataset
    dataset = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        load_subset=sample_size
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Class distribution
    distribution = dataset.get_class_distribution()
    print("\nClass distribution (top 10):")
    sorted_classes = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_classes[:10]:
        class_name = dataset.get_class_name(label)
        print(f"  Class {label:3d} ({class_name}): {count} samples")
    
    # Sample images
    print("\nSample images:")
    for i in range(min(5, len(dataset))):
        image, label = dataset[i]
        class_name = dataset.get_class_name(label)
        print(f"  Sample {i}: Shape {image.shape}, Label {label} ({class_name})")
    
    print("\nDataset analysis complete!")


if __name__ == "__main__":
    # Example usage
    data_dir = "/Users/gclinger/Documents/projects/mwnn/multi-weight-neural-networks/data/ImageNet-1K"
    devkit_dir = "/Users/gclinger/Documents/projects/mwnn/multi-weight-neural-networks/data/ImageNet-1K/ILSVRC2013_devkit"
    
    # Analyze dataset
    analyze_imagenet_dataset(data_dir, devkit_dir, sample_size=100)
    
    # Create dataset and dataloaders
    dataset = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        load_subset=100
    )
    
    train_loader, val_loader = create_imagenet_dataloaders(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        batch_size=16,
        load_subset=100
    )
    
    print(f"Created dataloaders: Train {len(train_loader)} batches, Val {len(val_loader)} batches")

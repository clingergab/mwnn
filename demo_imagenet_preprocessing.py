#!/usr/bin/env python3
"""
Example script demonstrating ImageNet-1K preprocessing for Multi-Weight Neural Networks.

This script shows how to set up and use the ImageNet preprocessing pipeline
for training and evaluating Multi-Weight Neural Networks.
"""

import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.imagenet_dataset import (
    create_imagenet_mwnn_dataset,
    create_imagenet_rgb_luminance_dataset,
    create_imagenet_dataloaders,
    create_imagenet_rgb_luminance_dataloaders,
    analyze_imagenet_dataset
)
from src.preprocessing.imagenet_config import (
    ImageNetPreprocessingConfig,
    get_preset_config
)


def demonstrate_basic_usage():
    """Demonstrate basic ImageNet dataset usage."""
    print("="*60)
    print("BASIC IMAGENET DATASET USAGE")
    print("="*60)
    
    # Set paths (modify these to match your setup)
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    print(f"Data directory: {data_dir}")
    print(f"Devkit directory: {devkit_dir}")
    
    # Create a small RGB+Luminance dataset for demonstration
    print("\n1. Creating ImageNet RGB+Luminance dataset...")
    dataset = create_imagenet_rgb_luminance_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        input_size=224,
        augment=False,
        load_subset=20  # Only load 20 samples for demo
    )
    
    print(f"   Dataset created with {len(dataset)} samples")
    
    # Test loading samples
    print("\n2. Testing RGB+Luminance sample loading...")
    for i in range(min(3, len(dataset))):
        image, label = dataset[i]
        class_name = dataset.get_class_name(label)
        print(f"   Sample {i}: Shape {image.shape} (4-channel RGB+L), Label {label} ({class_name})")
        print(f"   Channels: [R, G, B, Luminance]")
    
    # Create dataloader
    print("\n3. Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    print(f"   Dataloader created with {len(dataloader)} batches")
    
    # Test batch loading
    print("\n4. Testing batch loading...")
    for i, (images, labels) in enumerate(dataloader):
        print(f"   Batch {i}: Images {images.shape}, Labels {labels.shape}")
        print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   Sample labels: {labels.tolist()}")
        if i >= 2:  # Only show first 3 batches
            break


def demonstrate_configuration_system():
    """Demonstrate the configuration system."""
    print("\n" + "="*60)
    print("CONFIGURATION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Show different presets
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    presets = ['development', 'training', 'evaluation', 'research']
    
    for preset in presets:
        print(f"\n{preset.upper()} PRESET:")
        config = get_preset_config(
            preset=preset,
            data_dir=data_dir,
            devkit_dir=devkit_dir
        )
        
        print(f"   Input size: {config.input_size}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Augmentation: {config.augment}")
        print(f"   Load subset: {config.load_subset}")
        print(f"   Feature method: {config.feature_method}")
        print(f"   Workers: {config.num_workers}")


def demonstrate_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("\n" + "="*60)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("="*60)
    
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    feature_methods = ['hsv', 'lab', 'yuv']
    
    for method in feature_methods:
        print(f"\n{method.upper()} FEATURE EXTRACTION:")
        
        # Create dataset with specific feature method
        dataset = create_imagenet_mwnn_dataset(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            feature_method=method,
            augment=False,
            load_subset=5
        )
        
        # Load a sample
        image, label = dataset[0]
        class_name = dataset.get_class_name(label)
        
        print(f"   Sample: Shape {image.shape}, Label {label}")
        print(f"   Class: {class_name}")
        print(f"   Image stats: mean={image.mean():.3f}, std={image.std():.3f}")


def demonstrate_performance_benchmark():
    """Demonstrate performance benchmarking."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    # Test different configurations
    configs = [
        ('Small batch, few workers', {'batch_size': 8, 'num_workers': 1}),
        ('Medium batch, more workers', {'batch_size': 16, 'num_workers': 2}),
        ('Large batch, many workers', {'batch_size': 32, 'num_workers': 4}),
    ]
    
    for config_name, config_params in configs:
        print(f"\n{config_name}:")
        
        # Create dataloader
        train_loader, _ = create_imagenet_dataloaders(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            load_subset=100,  # Small subset for benchmarking
            **config_params
        )
        
        # Benchmark loading time
        start_time = time.time()
        total_samples = 0
        
        for i, (images, labels) in enumerate(train_loader):
            total_samples += images.size(0)
            if i >= 5:  # Only test first 5 batches
                break
        
        elapsed_time = time.time() - start_time
        samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
        
        print(f"   Batches: {min(6, len(train_loader))}")
        print(f"   Total samples: {total_samples}")
        print(f"   Time: {elapsed_time:.2f}s")
        print(f"   Samples/sec: {samples_per_sec:.1f}")


def demonstrate_data_augmentation():
    """Demonstrate data augmentation effects."""
    print("\n" + "="*60)
    print("DATA AUGMENTATION DEMONSTRATION")
    print("="*60)
    
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    # Compare without and with augmentation
    print("\nCreating datasets:")
    
    # Without augmentation
    dataset_no_aug = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        augment=False,
        load_subset=10
    )
    
    # With augmentation
    dataset_with_aug = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        augment=True,
        load_subset=10
    )
    
    print(f"   No augmentation: {len(dataset_no_aug)} samples")
    print(f"   With augmentation: {len(dataset_with_aug)} samples")
    
    # Compare same image with and without augmentation
    print("\nComparing first image:")
    
    # Load same image multiple times with augmentation
    image_no_aug, label = dataset_no_aug[0]
    print(f"   No augmentation: shape {image_no_aug.shape}")
    print(f"   Stats: mean={image_no_aug.mean():.3f}, std={image_no_aug.std():.3f}")
    
    # Load same image with augmentation (may vary each time)
    for i in range(3):
        image_aug, _ = dataset_with_aug[0]
        print(f"   Augmentation {i+1}: mean={image_aug.mean():.3f}, std={image_aug.std():.3f}")


def demonstrate_rgb_luminance_features():
    """Demonstrate RGB+Luminance feature extraction."""
    print("\n" + "="*60)
    print("RGB+LUMINANCE FEATURE EXTRACTION")
    print("="*60)
    
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    print("\nCreating RGB+Luminance dataset...")
    dataset = create_imagenet_rgb_luminance_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        load_subset=5
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test sample loading
    print("\nTesting 4-channel RGB+Luminance data:")
    for i in range(min(3, len(dataset))):
        image, label = dataset[i]
        class_name = dataset.get_class_name(label)
        
        print(f"\nSample {i}:")
        print(f"   Shape: {image.shape} (4 channels: R, G, B, Luminance)")
        print(f"   Label: {label} ({class_name})")
        print("   RGB channels stats:")
        print(f"     R: mean={image[0].mean():.3f}, std={image[0].std():.3f}")
        print(f"     G: mean={image[1].mean():.3f}, std={image[1].std():.3f}")
        print(f"     B: mean={image[2].mean():.3f}, std={image[2].std():.3f}")
        print("   Luminance channel stats:")
        print(f"     L: mean={image[3].mean():.3f}, std={image[3].std():.3f}")
    
    # Create RGB+Luminance dataloaders
    print("\nCreating RGB+Luminance dataloaders...")
    train_loader, val_loader = create_imagenet_rgb_luminance_dataloaders(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        batch_size=4,
        load_subset=20,
        train_split=0.7
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test batch loading
    print("\nTesting batch loading:")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}: Images {images.shape} (4-channel), Labels {labels.shape}")
        if i >= 1:  # Only show first 2 batches
            break


def main():
    """Main demonstration function."""
    print("ImageNet-1K Preprocessing for Multi-Weight Neural Networks")
    print("Demo Script")
    print("="*60)
    
    try:
        # Check if data directory exists
        data_dir = Path("data/ImageNet-1K")
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            print("Please make sure ImageNet-1K data is available in the data/ directory")
            return
        
        # Run demonstrations
        demonstrate_basic_usage()
        demonstrate_configuration_system()
        demonstrate_feature_extraction()
        demonstrate_performance_benchmark()
        demonstrate_data_augmentation()
        demonstrate_rgb_luminance_features()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Modify the configuration files in configs/preprocessing/")
        print("2. Update data paths to match your setup")
        print("3. Use the preprocessing utilities for your MWNN training")
        print("4. Check the documentation in configs/preprocessing/README.md")
        
    except Exception as e:
        print(f"ERROR during demonstration: {e}")
        print("Make sure all dependencies are installed and data paths are correct")


if __name__ == "__main__":
    main()

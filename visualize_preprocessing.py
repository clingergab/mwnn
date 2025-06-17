#!/usr/bin/env python3
"""
ImageNet RGB+Luminance Preprocessing Visualization

This script demonstrates the RGB+Luminance preprocessing pipeline by:
1. Loading original ImageNet images
2. Applying RGB+Luminance transformation
3. Visualizing the before/after data
4. Showing channel separation for MWNN processing
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

# Add project root to path
sys.path.append('.')

from src.preprocessing.imagenet_dataset import (
    create_imagenet_rgb_luminance_dataset,
    create_imagenet_rgb_luminance_dataloaders
)
from src.preprocessing.color_extractors import (
    rgb_to_rgb_luminance,
    extract_color_brightness_from_rgb_luminance
)
from src.preprocessing.imagenet_config import get_preset_config

def visualize_rgb_luminance_transformation():
    """Visualize the RGB to RGB+Luminance transformation."""
    print("🎨 ImageNet RGB+Luminance Preprocessing Visualization")
    print("=" * 60)
    
    # 1. Setup configuration
    print("\n1️⃣ Setting up configuration...")
    data_dir = "data/ImageNet-1K"
    devkit_dir = "data/ImageNet-1K/ILSVRC2013_devkit"
    
    config = get_preset_config('development', data_dir, devkit_dir)
    print(f"   ✅ Using {config.feature_method} feature extraction")
    print(f"   ✅ Batch size: {config.batch_size}")
    print(f"   ✅ Input size: {config.input_size}")
    
    # 2. Create dataset and dataloader
    print("\n2️⃣ Creating RGB+Luminance dataset...")
    try:
        train_loader, val_loader = create_imagenet_rgb_luminance_dataloaders(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            batch_size=4,  # Small batch for visualization
            load_subset=20,  # Just a few samples
            val_split=0.2
        )
        print(f"   ✅ Train loader: {len(train_loader)} batches")
        print(f"   ✅ Val loader: {len(val_loader)} batches")
    except Exception as e:
        print(f"   ❌ Error creating dataloaders: {e}")
        return False
    
    # 3. Get a batch of data
    print("\n3️⃣ Loading sample batch...")
    try:
        # Get a batch from the dataloader
        images, labels = next(iter(train_loader))
        print(f"   ✅ Loaded batch shape: {images.shape}")
        print(f"   ✅ Labels shape: {labels.shape}")
        print(f"   ✅ Image data type: {images.dtype}")
        print(f"   ✅ Value range: [{images.min():.3f}, {images.max():.3f}]")
        
        # The images are already in RGB+Luminance format (4 channels)
        assert images.shape[1] == 4, f"Expected 4 channels, got {images.shape[1]}"
        print("   ✅ Confirmed 4-channel RGB+Luminance data")
        
    except Exception as e:
        print(f"   ❌ Error loading batch: {e}")
        return False
    
    # 4. Extract color and brightness pathways
    print("\n4️⃣ Extracting color and brightness pathways...")
    try:
        color_pathway, brightness_pathway = extract_color_brightness_from_rgb_luminance(images)
        print(f"   ✅ Color pathway shape: {color_pathway.shape}")
        print(f"   ✅ Brightness pathway shape: {brightness_pathway.shape}")
        print(f"   ✅ Color range: [{color_pathway.min():.3f}, {color_pathway.max():.3f}]")
        print(f"   ✅ Brightness range: [{brightness_pathway.min():.3f}, {brightness_pathway.max():.3f}]")
    except Exception as e:
        print(f"   ❌ Error extracting pathways: {e}")
        return False
    
    # 5. Create visualizations
    print("\n5️⃣ Creating visualizations...")
    try:
        # Select first image for detailed visualization
        sample_idx = 0
        rgb_luminance_img = images[sample_idx]  # Shape: (4, H, W)
        color_img = color_pathway[sample_idx]   # Shape: (3, H, W)
        brightness_img = brightness_pathway[sample_idx]  # Shape: (1, H, W)
        
        # Convert to numpy and proper format for matplotlib
        def tensor_to_numpy(tensor):
            """Convert tensor to numpy array for visualization."""
            if tensor.dim() == 3:
                # CHW to HWC for matplotlib
                return tensor.permute(1, 2, 0).cpu().numpy()
            elif tensor.dim() == 2:
                return tensor.cpu().numpy()
            else:
                return tensor.squeeze().cpu().numpy()
        
        rgb_img_np = tensor_to_numpy(color_img)
        brightness_img_np = tensor_to_numpy(brightness_img)
        luminance_channel_np = tensor_to_numpy(rgb_luminance_img[3])  # Just the luminance channel
        
        # Ensure values are in [0, 1] range for visualization
        rgb_img_np = np.clip(rgb_img_np, 0, 1)
        brightness_img_np = np.clip(brightness_img_np, 0, 1)
        luminance_channel_np = np.clip(luminance_channel_np, 0, 1)
        
        # Create the visualization with enhanced layout
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('ImageNet RGB+Luminance Preprocessing Visualization', fontsize=18, fontweight='bold')
        
        # Row 1: Original RGB and luminance components
        axes[0, 0].imshow(rgb_img_np)
        axes[0, 0].set_title('Original RGB Image\n(Color Pathway)', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(brightness_img_np, cmap='gray')
        axes[0, 1].set_title('Computed Luminance\n(Brightness Pathway)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(luminance_channel_np, cmap='gray')
        axes[0, 2].set_title('Luminance Channel\n(ITU-R BT.709)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Individual RGB channels
        axes[1, 0].imshow(tensor_to_numpy(color_img[0]), cmap='Reds')
        axes[1, 0].set_title('Red Channel', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(tensor_to_numpy(color_img[1]), cmap='Greens')
        axes[1, 1].set_title('Green Channel', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(tensor_to_numpy(color_img[2]), cmap='Blues')
        axes[1, 2].set_title('Blue Channel', fontweight='bold')
        axes[1, 2].axis('off')
        
        # Row 3: Data distribution histograms
        axes[2, 0].hist(rgb_img_np.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[2, 0].set_title('RGB Pixel Distribution', fontweight='bold')
        axes[2, 0].set_xlabel('Pixel Value')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].hist(brightness_img_np.flatten(), bins=50, alpha=0.7, color='gray', edgecolor='black')
        axes[2, 1].set_title('Luminance Distribution', fontweight='bold')
        axes[2, 1].set_xlabel('Luminance Value')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Channel comparison
        for i, (channel, color, name) in enumerate([(color_img[0], 'red', 'R'), 
                                                    (color_img[1], 'green', 'G'), 
                                                    (color_img[2], 'blue', 'B')]):
            channel_np = tensor_to_numpy(channel)
            axes[2, 2].hist(channel_np.flatten(), bins=30, alpha=0.6, 
                           color=color, label=f'{name} Channel', edgecolor='black')
        
        axes[2, 2].set_title('RGB Channel Comparison', fontweight='bold')
        axes[2, 2].set_xlabel('Pixel Value')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = 'rgb_luminance_visualization.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Show comprehensive data statistics
    print("\n6️⃣ Comprehensive Data Statistics:")
    print(f"   📊 4-Channel Data Shape: {images.shape}")
    print(f"   📊 RGB Channels (0-2): {color_pathway.shape}")
    print(f"   📊 Luminance Channel (3): {brightness_pathway.shape}")
    
    # Format RGB channel statistics properly
    rgb_means = color_pathway.mean(dim=(0,2,3))
    rgb_stds = color_pathway.std(dim=(0,2,3))
    rgb_mins = color_pathway.min()
    rgb_maxs = color_pathway.max()
    
    rgb_mean_str = ', '.join([f"{val:.3f}" for val in rgb_means])
    rgb_std_str = ', '.join([f"{val:.3f}" for val in rgb_stds])
    
    print(f"   📊 RGB mean: [{rgb_mean_str}]")
    print(f"   📊 RGB std: [{rgb_std_str}]")
    print(f"   📊 RGB range: [{rgb_mins:.3f}, {rgb_maxs:.3f}]")
    print(f"   📊 Luminance mean: {brightness_pathway.mean():.3f}")
    print(f"   📊 Luminance std: {brightness_pathway.std():.3f}")
    print(f"   📊 Luminance range: [{brightness_pathway.min():.3f}, {brightness_pathway.max():.3f}]")
    
    # Additional insights
    print(f"   🔍 Data type: {images.dtype}")
    print(f"   🔍 Memory usage: {images.element_size() * images.nelement() / 1024**2:.2f} MB")
    print("   🔍 Normalization applied: Yes (ImageNet standards)")
    print("   🔍 ITU-R BT.709 weights: R=0.2126, G=0.7152, B=0.0722")
    
    # 7. Demonstrate manual transformation
    print("\n7️⃣ Demonstrating manual RGB to RGB+Luminance transformation...")
    try:
        # Create a simple RGB tensor
        sample_rgb = torch.rand(1, 3, 64, 64)
        rgb_lum_manual = rgb_to_rgb_luminance(sample_rgb)
        
        print(f"   ✅ Input RGB shape: {sample_rgb.shape}")
        print(f"   ✅ Output RGB+Luminance shape: {rgb_lum_manual.shape}")
        
        # Verify luminance calculation
        r, g, b = sample_rgb[0, 0], sample_rgb[0, 1], sample_rgb[0, 2]
        manual_luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        computed_luminance = rgb_lum_manual[0, 3]
        
        diff = torch.abs(manual_luminance - computed_luminance).max()
        print(f"   ✅ Luminance calculation accuracy: {diff:.6f} (should be ~0)")
        
    except Exception as e:
        print(f"   ❌ Error in manual transformation: {e}")
        return False
    
    print("\n🎉 RGB+Luminance Preprocessing Visualization Complete!")
    print("✅ Successfully processed ImageNet data with RGB+Luminance approach")
    print("✅ Visualization saved as: rgb_luminance_visualization.png")
    print("✅ Key benefit: Zero information loss (RGB preserved + luminance added)")
    
    return True

if __name__ == "__main__":
    # Run the main visualization
    success = visualize_rgb_luminance_transformation()
    
    if success:        
        print("\n" + "="*60)
        print("🎯 Key Takeaways:")
        print("✅ RGB+Luminance provides 4-channel data (R, G, B, L)")
        print("✅ Zero information loss from original RGB images")
        print("✅ Clean separation for MWNN color/brightness pathways")
        print("✅ ITU-R BT.709 standard luminance calculation")
        print("✅ Superior to traditional color space transformations")
    else:
        print("\n❌ Visualization failed. Check error messages above.")

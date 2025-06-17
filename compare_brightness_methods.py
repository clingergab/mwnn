#!/usr/bin/env python3
"""
Brightness Extraction Method Comparison

Compares different approaches to extracting brightness from RGB images:
1. RGB + ITU-R BT.709 Luminance (current approach)
2. RGB + YUV Y-channel Brightness (alternative approach)

Shows 5 images with side-by-side brightness comparisons.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('.')

from src.preprocessing.imagenet_dataset import create_imagenet_rgb_luminance_dataloaders
from src.preprocessing.color_extractors import extract_color_brightness_from_rgb_luminance

def rgb_to_yuv_brightness(rgb_tensor):
    """Extract YUV Y-channel (brightness) from RGB tensor."""
    # YUV conversion matrix (ITU-R BT.601)
    # Y = 0.299*R + 0.587*G + 0.114*B
    r, g, b = rgb_tensor[:, 0, :, :], rgb_tensor[:, 1, :, :], rgb_tensor[:, 2, :, :]
    y_channel = 0.299 * r + 0.587 * g + 0.114 * b
    return y_channel.unsqueeze(1)  # Add channel dimension

def compare_brightness_methods():
    """Compare ITU-R BT.709 luminance vs YUV Y-channel brightness."""
    print("🔬 Brightness Extraction Method Comparison")
    print("=" * 60)
    print("Method 1: RGB + ITU-R BT.709 Luminance (R=0.2126, G=0.7152, B=0.0722)")
    print("Method 2: RGB + YUV Y-channel (R=0.299, G=0.587, B=0.114)")
    print("=" * 60)
    
    # Load data
    print("Loading ImageNet samples...")
    try:
        train_loader, _ = create_imagenet_rgb_luminance_dataloaders(
            data_dir="data/ImageNet-1K",
            devkit_dir="data/ImageNet-1K/ILSVRC2013_devkit",
            batch_size=5,
            load_subset=5,
            val_split=0.0
        )
        
        # Get batch of 5 images (already in 4-channel RGB+Luminance format)
        images_4ch, labels = next(iter(train_loader))
        print(f"✅ Loaded {images_4ch.shape[0]} images")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Extract RGB and current luminance
    color_pathway, luminance_bt709 = extract_color_brightness_from_rgb_luminance(images_4ch)
    
    # Extract YUV Y-channel brightness from RGB
    yuv_brightness = rgb_to_yuv_brightness(color_pathway)
    
    print(f"📊 RGB shape: {color_pathway.shape}")
    print(f"📊 BT.709 Luminance shape: {luminance_bt709.shape}")
    print(f"📊 YUV Y-channel shape: {yuv_brightness.shape}")
    
    # Convert tensors for visualization
    def tensor_to_display(tensor):
        """Convert tensor to displayable numpy array."""
        if tensor.dim() == 3:
            array = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            array = tensor.cpu().numpy()
        
        # Normalize to [0, 1] for display
        array = (array - array.min()) / (array.max() - array.min())
        return np.clip(array, 0, 1)
    
    # Create comparison visualization: 5 rows x 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle('Brightness Extraction Method Comparison\n' +
                 'Original RGB | BT.709 Luminance | YUV Y-channel | Difference Map', 
                 fontsize=16, fontweight='bold')
    
    for i in range(5):
        # Convert to display format
        rgb_img = tensor_to_display(color_pathway[i])
        bt709_img = tensor_to_display(luminance_bt709[i].squeeze())
        yuv_img = tensor_to_display(yuv_brightness[i].squeeze())
        
        # Calculate difference map
        diff_map = np.abs(bt709_img - yuv_img)
        
        # Display images
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'Image {i+1}: Original RGB', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(bt709_img, cmap='gray')
        axes[i, 1].set_title('BT.709 Luminance\n(Current Method)', fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(yuv_img, cmap='gray')
        axes[i, 2].set_title('YUV Y-channel\n(Alternative Method)', fontweight='bold')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(diff_map, cmap='hot')
        axes[i, 3].set_title(f'Difference Map\n(Max: {diff_map.max():.3f})', fontweight='bold')
        axes[i, 3].axis('off')
        
        # Print detailed comparison to terminal
        print(f"\nImage {i+1} Analysis (Label: {labels[i].item()}):")
        print(f"  RGB range: [{color_pathway[i].min():.3f}, {color_pathway[i].max():.3f}]")
        print(f"  BT.709 Luminance: [{luminance_bt709[i].min():.3f}, {luminance_bt709[i].max():.3f}], mean: {luminance_bt709[i].mean():.3f}")
        print(f"  YUV Y-channel:   [{yuv_brightness[i].min():.3f}, {yuv_brightness[i].max():.3f}], mean: {yuv_brightness[i].mean():.3f}")
        print(f"  Difference stats: mean: {diff_map.mean():.4f}, max: {diff_map.max():.4f}")
        
        # Calculate correlation
        bt709_flat = bt709_img.flatten()
        yuv_flat = yuv_img.flatten()
        correlation = np.corrcoef(bt709_flat, yuv_flat)[0, 1]
        print(f"  Correlation: {correlation:.4f}")
    
    plt.tight_layout()
    plt.savefig('brightness_method_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Comparison saved to: brightness_method_comparison.png")
    
    # Overall comparison
    print("\n" + "=" * 60)
    print("🎯 Method Comparison Summary:")
    print("ITU-R BT.709 Luminance (Current):")
    print("  ✅ Perceptually accurate (matches human vision)")
    print("  ✅ Industry standard for display/broadcast")
    print("  ✅ Optimized for human visual system")
    print("  📊 Weights: R=21.26%, G=71.52%, B=7.22%")
    
    print("\nYUV Y-channel (Alternative):")
    print("  ✅ Simpler calculation")
    print("  ✅ Video processing standard")
    print("  ⚠️  Less perceptually accurate")
    print("  📊 Weights: R=29.9%, G=58.7%, B=11.4%")
    
    print("\n🔬 Key Differences:")
    print("  • BT.709 emphasizes green more (human eye sensitivity)")
    print("  • YUV gives more weight to red and blue")
    print("  • BT.709 is newer standard (1990s vs 1950s)")
    print("  • Both produce similar but slightly different brightness maps")
    print("=" * 60)

if __name__ == "__main__":
    compare_brightness_methods()

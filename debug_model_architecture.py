#!/usr/bin/env python3
"""
Debug script to check model architecture and channel flow
Run this in Colab to verify the model is built correctly
"""

import torch
import sys
import os

# Add the project to Python path
sys.path.append('/content/drive/MyDrive/mwnn/multi-weight-neural-networks')

from src.models.continuous_integration.model import ContinuousIntegrationModel

def debug_model_architecture(base_channels=24):
    """Debug the model architecture to find channel mismatches"""
    print(f"🔍 Debugging model with base_channels={base_channels}")
    print("=" * 50)
    
    # Create model
    model = ContinuousIntegrationModel(
        num_classes=1000,
        depth='deep',
        base_channels=base_channels,
        dropout_rate=0.2,
        integration_points=['early', 'middle', 'late'],
        enable_mixed_precision=True,
        memory_efficient=True
    )
    
    print(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check depth config
    print(f"\n🏗️  Depth config for 'deep':")
    config = model.depth_configs['deep']
    print(f"   Blocks: {config['blocks']}")
    print(f"   Channels: {config['channels']}")
    
    # Check initial layers
    print(f"\n🎯 Initial layers:")
    print(f"   RGB input: 3 → {model.initial_color.conv.out_channels}")
    print(f"   Brightness input: 1 → {model.initial_brightness.conv.out_channels}")
    
    # Check stages
    print(f"\n🏭 Stages:")
    for i, stage in enumerate(model.stages):
        print(f"   Stage {i}:")
        # Check first block of color pathway
        first_color_block = stage.color_blocks[0]
        print(f"     Color: {first_color_block.conv1.in_channels} → {first_color_block.conv1.out_channels}")
        # Check first block of brightness pathway  
        first_brightness_block = stage.brightness_blocks[0]
        print(f"     Brightness: {first_brightness_block.conv1.in_channels} → {first_brightness_block.conv1.out_channels}")
    
    # Test forward pass with dummy data
    print(f"\n🧪 Testing forward pass...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create test inputs
        batch_size = 2
        rgb_data = torch.randn(batch_size, 3, 224, 224).to(device)
        brightness_data = torch.randn(batch_size, 1, 224, 224).to(device)
        
        print(f"   Input shapes: RGB {rgb_data.shape}, Brightness {brightness_data.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(rgb_data, brightness_data)
        
        print(f"   ✅ Success! Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_different_base_channels():
    """Test different base_channels values to find working configuration"""
    print("🧪 Testing different base_channels values...")
    print("=" * 50)
    
    for base_channels in [16, 20, 24, 28, 32]:
        print(f"\n📊 Testing base_channels = {base_channels}")
        try:
            success = debug_model_architecture(base_channels)
            if success:
                print(f"   ✅ base_channels={base_channels} WORKS!")
            else:
                print(f"   ❌ base_channels={base_channels} FAILED")
        except Exception as e:
            print(f"   ❌ base_channels={base_channels} ERROR: {e}")

if __name__ == "__main__":
    print("🔍 MWNN Model Architecture Debug")
    print("=" * 40)
    
    # Test current configuration
    debug_model_architecture(24)
    
    print("\n" + "=" * 40)
    
    # Test different configurations
    test_different_base_channels()

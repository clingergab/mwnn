#!/usr/bin/env python3
"""
Simple test for continuous integration model to isolate tensor stride issues
"""

import sys
import torch

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def simple_model_test():
    """Simple test to identify the tensor stride issue."""
    print("🔍 Simple Model Test")
    print("=" * 30)
    
    # Detect device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Create a simple model without optimizations first
    print("\n1. Testing basic model creation...")
    try:
        model = ContinuousIntegrationModel(
            num_classes=10,  # Small number of classes
            base_channels=32,  # Smaller channels
            depth='shallow',  # Simplest architecture
            enable_mixed_precision=False,  # Disable mixed precision
            memory_efficient=False  # Disable memory optimizations
        )
        print("✅ Basic model created successfully")
    except Exception as e:
        print(f"❌ Basic model creation failed: {e}")
        return
    
    # Move to device
    print("\n2. Testing device transfer...")
    try:
        model = model.to(device)
        print("✅ Model moved to device successfully")
    except Exception as e:
        print(f"❌ Device transfer failed: {e}")
        return
    
    # Create simple input tensors
    print("\n3. Testing input tensor creation...")
    try:
        batch_size = 2  # Very small batch
        rgb_data = torch.randn(batch_size, 3, 64, 64, device=device)  # Small resolution
        brightness_data = torch.randn(batch_size, 1, 64, 64, device=device)
        print(f"✅ Input tensors created: RGB {rgb_data.shape}, Brightness {brightness_data.shape}")
    except Exception as e:
        print(f"❌ Input tensor creation failed: {e}")
        return
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(rgb_data, brightness_data)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        
        # Try to identify which part fails
        print("\n🔍 Debugging forward pass components...")
        
        try:
            # Test initial processing
            color = model.initial_color(rgb_data)
            brightness = model.initial_brightness(brightness_data)
            print(f"✅ Initial processing: color {color.shape}, brightness {brightness.shape}")
            
            # Test first stage
            stage = model.stages[0]
            color_out, brightness_out, integrated_out = stage(color, brightness, None)
            print(f"✅ First stage: color {color_out.shape}, brightness {brightness_out.shape}")
            
        except Exception as stage_e:
            print(f"❌ Stage processing failed: {stage_e}")
        
        return
    
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    simple_model_test()

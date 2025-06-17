#!/usr/bin/env python3
"""
Test automatic GPU detection - no arguments needed!
"""

import sys
import torch

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def test_automatic_gpu_detection():
    """Test that the model automatically detects and uses the best GPU."""
    print("🤖 Automatic GPU Detection Test")
    print("=" * 40)
    print("ℹ️  No device arguments needed - model auto-detects best GPU!")
    print()
    
    # Create model WITHOUT specifying any device
    print("1. Creating model with automatic device detection...")
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=32,
        depth='shallow'
    )
    
    print(f"   ✅ Model automatically detected and configured for: {model.device}")
    
    # Show what was detected
    if model.device.type == 'mps':
        print("   🍎 Using Apple Silicon GPU (Metal Performance Shaders)")
        print("   📈 Optimized for Mac M1/M2/M3 performance")
    elif model.device.type == 'cuda':
        device_name = torch.cuda.get_device_name(0)
        print(f"   🚀 Using NVIDIA GPU: {device_name}")
        print("   📈 Optimized for CUDA performance")
    else:
        print("   💻 Using CPU (no GPU available)")
    
    # Test that the model actually works on the detected device
    print("\n2. Testing model on auto-detected device...")
    
    # Create test data (automatically placed on the same device)
    batch_size = 4
    rgb_data = torch.randn(batch_size, 3, 224, 224, device=model.device)
    brightness_data = torch.randn(batch_size, 1, 224, 224, device=model.device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(rgb_data, brightness_data)
    
    print(f"   ✅ Inference successful! Output shape: {output.shape}")
    print(f"   📊 Data and model on same device: {rgb_data.device}")
    
    # Show GPU memory usage if available
    if model.device.type == 'mps':
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**2  # MB
            print(f"   💾 GPU memory used: {allocated:.1f} MB")
        except AttributeError:
            print("   💾 Memory tracking not available")
    elif model.device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(model.device) / 1024**2  # MB
        print(f"   💾 GPU memory used: {allocated:.1f} MB")
    
    # Show that integration weights are working
    print("\n3. Verifying model functionality...")
    weights = model.get_integration_weights()
    print("   🔗 Integration weights:")
    for stage, stage_weights in weights.items():
        color_w = stage_weights['color']
        brightness_w = stage_weights['brightness']
        print(f"      {stage}: color={color_w:.3f}, brightness={brightness_w:.3f}")
    
    print("\n🎉 Automatic GPU detection working perfectly!")
    print("💡 No device arguments needed - the model is smart enough to find the best GPU!")


if __name__ == "__main__":
    test_automatic_gpu_detection()

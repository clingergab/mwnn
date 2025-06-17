#!/usr/bin/env python3
"""
Simple debug script to test the integration module specifically
"""

import torch
import sys
sys.path.append('/content/drive/MyDrive/mwnn/multi-weight-neural-networks')

from src.models.continuous_integration.integration_module import IntegrationModule

def test_integration_module():
    """Test the IntegrationModule with different channel sizes"""
    print("🧪 Testing IntegrationModule...")
    
    # Test case 1: 56 channels (from base_channels=28, stage 0)
    print("\n📊 Test 1: 56 channels")
    channels = 56
    module = IntegrationModule(channels, channels)
    
    # Create test tensors
    batch_size = 2
    height, width = 56, 56  # After some downsampling
    
    color = torch.randn(batch_size, channels, height, width)
    brightness = torch.randn(batch_size, channels, height, width)
    
    print(f"  Color shape: {color.shape}")
    print(f"  Brightness shape: {brightness.shape}")
    print(f"  Expected concat shape: {[batch_size, channels*2, height, width]}")
    
    try:
        # Test concatenation
        concat = torch.cat([color, brightness], dim=1)
        print(f"  Actual concat shape: {concat.shape}")
        
        # Test the integration MLP
        output = module.integration_mlp(concat)
        print(f"  MLP output shape: {output.shape}")
        
        # Test full forward pass
        result = module(color, brightness)
        print(f"  ✅ Success! Final output shape: {result.shape}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        
        # Debug the integration_mlp layers
        print(f"  🔍 Integration MLP layers:")
        for i, layer in enumerate(module.integration_mlp):
            print(f"    Layer {i}: {layer}")

def test_model_step_by_step():
    """Test model step by step to find where the mismatch occurs"""
    print("\n🔍 Step-by-step model debug...")
    
    from src.models.continuous_integration.model import ContinuousIntegrationModel
    
    # Use CPU to avoid device issues during debugging
    device = torch.device('cpu')
    
    model = ContinuousIntegrationModel(
        num_classes=1000,
        depth='deep', 
        base_channels=28,
        dropout_rate=0.2,
        integration_points=['early', 'middle', 'late'],
        enable_mixed_precision=False,  # Disable for debugging
        memory_efficient=True
    )
    model = model.to(device)
    
    # Test inputs on CPU
    batch_size = 2
    rgb_data = torch.randn(batch_size, 3, 224, 224).to(device)
    brightness_data = torch.randn(batch_size, 1, 224, 224).to(device)
    
    print(f"Input shapes - RGB: {rgb_data.shape}, Brightness: {brightness_data.shape}")
    print(f"Device: {device}")
    
    try:
        # Step 1: Initial processing
        color = model.initial_color(rgb_data)
        brightness = model.initial_brightness(brightness_data)
        print(f"After initial: Color {color.shape}, Brightness {brightness.shape}")
        
        # Step 2: Process through stages
        integrated = None
        
        for i, stage in enumerate(model.stages):
            print(f"\n🏭 Stage {i}:")
            print(f"  Input: Color {color.shape}, Brightness {brightness.shape}")
            
            try:
                color, brightness, stage_integrated = stage(color, brightness, integrated)
                print(f"  After stage: Color {color.shape}, Brightness {brightness.shape}")
                
                # Check integration
                stage_name = f'stage_{i}'
                if stage_name in model.integration_modules:
                    print(f"  🔗 Applying integration for {stage_name}")
                    integrated = model.integration_modules[stage_name](
                        color, brightness, stage_integrated
                    )
                    print(f"  After integration: {integrated.shape}")
                    
            except Exception as e:
                print(f"  ❌ Error in stage {i}: {e}")
                # Print more debug info
                print(f"  🔍 Stage details:")
                print(f"    Expected in_channels: {stage.color_blocks[0].conv1.in_channels}")
                print(f"    Expected out_channels: {stage.color_blocks[0].conv1.out_channels}")
                print(f"    Actual color channels: {color.shape[1]}")
                print(f"    Actual brightness channels: {brightness.shape[1]}")
                break
                
    except Exception as e:
        print(f"❌ Error in initial processing: {e}")

if __name__ == "__main__":
    test_integration_module()
    test_model_step_by_step()

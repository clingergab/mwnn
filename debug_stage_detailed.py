#!/usr/bin/env python3
"""
Detailed debug script to trace the exact error location
"""

import torch
import sys
sys.path.append('/content/drive/MyDrive/mwnn/multi-weight-neural-networks')

from src.models.continuous_integration.model import ContinuousIntegrationModel

def debug_stage_by_stage():
    """Debug each stage individually"""
    print("🔍 Detailed Stage-by-Stage Debug")
    print("=" * 50)
    
    # Use CPU to avoid device issues
    device = torch.device('cpu')
    
    model = ContinuousIntegrationModel(
        num_classes=1000,
        depth='deep', 
        base_channels=28,
        dropout_rate=0.2,
        integration_points=['early', 'middle', 'late'],
        enable_mixed_precision=False,
        memory_efficient=True
    )
    model = model.to(device)
    
    # Test inputs
    batch_size = 2
    rgb_data = torch.randn(batch_size, 3, 224, 224).to(device)
    brightness_data = torch.randn(batch_size, 1, 224, 224).to(device)
    
    print(f"Input shapes - RGB: {rgb_data.shape}, Brightness: {brightness_data.shape}")
    
    # Initial processing
    color = model.initial_color(rgb_data)
    brightness = model.initial_brightness(brightness_data)
    print(f"After initial: Color {color.shape}, Brightness {brightness.shape}")
    
    # Debug each stage separately
    for i, stage in enumerate(model.stages):
        print(f"\n🏭 Stage {i} Detailed Debug:")
        print(f"  Input: Color {color.shape}, Brightness {brightness.shape}")
        
        # Check stage configuration
        print(f"  Stage config:")
        print(f"    Color blocks: {len(stage.color_blocks)}")
        print(f"    Brightness blocks: {len(stage.brightness_blocks)}")
        print(f"    Integration blocks: {len(stage.integration_blocks)}")
        
        # Check first block in each pathway
        first_color_block = stage.color_blocks[0]
        first_brightness_block = stage.brightness_blocks[0]
        
        print(f"    First color block: {first_color_block.conv1.in_channels} → {first_color_block.conv1.out_channels}")
        print(f"    First brightness block: {first_brightness_block.conv1.in_channels} → {first_brightness_block.conv1.out_channels}")
        
        try:
            # Process each pathway separately to isolate the error
            print(f"  🎨 Processing color pathway...")
            color_result = color
            for j, block in enumerate(stage.color_blocks):
                print(f"    Block {j}: input {color_result.shape}")
                color_result = block(color_result)
                print(f"    Block {j}: output {color_result.shape}")
            
            print(f"  💡 Processing brightness pathway...")
            brightness_result = brightness
            for j, block in enumerate(stage.brightness_blocks):
                print(f"    Block {j}: input {brightness_result.shape}")
                brightness_result = block(brightness_result)
                print(f"    Block {j}: output {brightness_result.shape}")
            
            # Update for next stage
            color = color_result
            brightness = brightness_result
            
            print(f"  ✅ Stage {i} completed: Color {color.shape}, Brightness {brightness.shape}")
            
            # Check integration
            stage_name = f'stage_{i}'
            if stage_name in model.integration_modules:
                print(f"  🔗 Testing integration for {stage_name}")
                integrated = model.integration_modules[stage_name](color, brightness, None)
                print(f"  🔗 Integration result: {integrated.shape}")
            
        except Exception as e:
            print(f"  ❌ Error in stage {i}: {e}")
            print(f"  🔍 Error details:")
            print(f"    Current color shape: {color.shape}")
            print(f"    Current brightness shape: {brightness.shape}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    debug_stage_by_stage()

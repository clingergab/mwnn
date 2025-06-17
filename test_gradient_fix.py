#!/usr/bin/env python3
"""
Test the gradient fix for continuous_integration model.
"""

import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def test_gradient_fix():
    """Test that gradients work correctly without graph conflicts."""
    print("🔍 Testing gradient computation fix...")
    
    # Create model
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=32,  # Smaller for faster testing
        depth='medium',
        enable_mixed_precision=False,  # Disable for clearer debugging
        memory_efficient=True
    )
    
    print(f"✅ Model created on device: {model.device}")
    
    # Create test data
    batch_size = 4
    rgb_data = torch.randn(batch_size, 3, 224, 224, device=model.device, requires_grad=True)
    brightness_data = torch.randn(batch_size, 1, 224, 224, device=model.device, requires_grad=True)
    targets = torch.randint(0, 1000, (batch_size,), device=model.device)
    
    # Set up training
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("🔥 Testing multiple backward passes...")
    
    # Test multiple training steps to ensure no gradient graph conflicts
    for step in range(5):
        print(f"  Step {step + 1}/5")
        
        try:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(rgb_data, brightness_data)
            loss = criterion(outputs, targets)
            
            print(f"    Forward pass: ✅ Loss = {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            print(f"    Backward pass: ✅")
            
            # Optimizer step
            optimizer.step()
            print(f"    Optimizer step: ✅")
            
            # Check gradients exist
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
            
            print(f"    Gradients computed: ✅ ({grad_count} parameters)")
            
        except RuntimeError as e:
            print(f"    ❌ Error in step {step + 1}: {e}")
            return False
    
    print("✅ All gradient tests passed!")
    
    # Test integration weights
    print("\n🔗 Testing integration weights...")
    weights = model.get_integration_weights()
    for stage, stage_weights in weights.items():
        print(f"  {stage}: {stage_weights}")
    
    print("\n✅ Gradient fix test complete - ready for training!")
    return True


if __name__ == "__main__":
    test_gradient_fix()

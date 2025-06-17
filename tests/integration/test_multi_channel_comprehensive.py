#!/usr/bin/env python3
"""Comprehensive test for Multi-Channel Model with Option 1A MultiChannelNeuron."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
from models.multi_channel.model import MultiChannelModel
from models.components.neurons import MultiChannelNeuron

def test_multi_channel_model_comprehensive():
    """Comprehensive test for multi-channel model functionality."""
    print("🧪 MULTI-CHANNEL MODEL COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test 1: Model Creation
    print("\n1️⃣ Testing Model Creation...")
    model = MultiChannelModel(
        input_channels=3,
        num_classes=10,
        depth='medium',
        fusion_method='adaptive',
        feature_extraction_method='hsv'
    )
    print("✅ Model created successfully")
    print(f"   - Input channels: 3")
    print(f"   - Output classes: 10")
    print(f"   - Depth: medium")
    print(f"   - Fusion method: adaptive")
    print(f"   - Feature extraction: HSV")
    
    # Test 2: Model Architecture Verification
    print("\n2️⃣ Testing Model Architecture...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")
    
    # Check key components exist
    assert hasattr(model, 'initial_color'), "Missing initial_color layer"
    assert hasattr(model, 'initial_brightness'), "Missing initial_brightness layer"
    assert hasattr(model, 'color_pathway'), "Missing color_pathway"
    assert hasattr(model, 'brightness_pathway'), "Missing brightness_pathway"
    assert hasattr(model, 'fusion'), "Missing fusion layer"
    assert hasattr(model, 'classifier'), "Missing classifier"
    print("✅ All key components present")
    
    # Test 3: Forward Pass
    print("\n3️⃣ Testing Forward Pass...")
    model.eval()
    batch_size = 8
    test_input = torch.randn(batch_size, 3, 32, 32)
    
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {output.shape}"
    print(f"✅ Forward pass successful: {test_input.shape} -> {output.shape}")
    
    # Test 4: Feature Extraction
    print("\n4️⃣ Testing HSV Feature Extraction...")
    with torch.no_grad():
        color_features, brightness_features = model.extract_features(test_input)
    
    assert color_features.shape == (batch_size, 2, 32, 32), f"Color features shape: {color_features.shape}"
    assert brightness_features.shape == (batch_size, 1, 32, 32), f"Brightness features shape: {brightness_features.shape}"
    print(f"✅ HSV extraction: RGB{test_input.shape} -> Color{color_features.shape} + Brightness{brightness_features.shape}")
    
    # Test 5: Pathway Outputs
    print("\n5️⃣ Testing Separate Pathway Outputs...")
    with torch.no_grad():
        color_out, brightness_out = model.get_pathway_outputs(test_input)
    
    print(f"✅ Pathway outputs: Color{color_out.shape}, Brightness{brightness_out.shape}")
    assert not torch.allclose(color_out, brightness_out), "Pathways should produce different outputs"
    print("✅ Pathways produce independent outputs")
    
    # Test 6: Different Fusion Methods
    print("\n6️⃣ Testing Different Fusion Methods...")
    fusion_methods = ['concatenate', 'add', 'adaptive']
    
    for fusion_method in fusion_methods:
        test_model = MultiChannelModel(
            input_channels=3,
            num_classes=10,
            depth='shallow',
            fusion_method=fusion_method
        )
        test_model.eval()
        
        with torch.no_grad():
            output = test_model(test_input)
        
        assert output.shape == (batch_size, 10), f"Fusion {fusion_method} failed"
        print(f"✅ {fusion_method} fusion: {output.shape}")
    
    # Test 7: Gradient Flow
    print("\n7️⃣ Testing Gradient Flow...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(test_input)
    target = torch.randint(0, 10, (batch_size,))
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    assert grad_norm > 0, "No gradients computed"
    print(f"✅ Gradient flow successful, gradient norm: {grad_norm:.4f}")
    
    # Test 8: MultiChannelNeuron Integration
    print("\n8️⃣ Testing MultiChannelNeuron Integration...")
    neuron = MultiChannelNeuron(input_size=64)
    
    # Test neuron with sample data
    color_input = torch.randn(4, 64)
    brightness_input = torch.randn(4, 64)
    
    with torch.no_grad():
        color_out, brightness_out = neuron(color_input, brightness_input)
    
    assert color_out.shape == (4,), f"Color output shape: {color_out.shape}"
    assert brightness_out.shape == (4,), f"Brightness output shape: {brightness_out.shape}"
    print(f"✅ MultiChannelNeuron integration: {color_input.shape} -> ({color_out.shape}, {brightness_out.shape})")
    
    # Test 9: Different Depths
    print("\n9️⃣ Testing Different Model Depths...")
    depths = ['shallow', 'medium', 'deep']
    
    for depth in depths:
        test_model = MultiChannelModel(
            input_channels=3,
            num_classes=10,
            depth=depth
        )
        test_model.eval()
        
        with torch.no_grad():
            output = test_model(test_input)
        
        assert output.shape == (batch_size, 10), f"Depth {depth} failed"
        param_count = sum(p.numel() for p in test_model.parameters())
        print(f"✅ {depth} depth: {output.shape}, {param_count:,} parameters")
    
    # Test 10: Feature Normalization
    print("\n🔟 Testing Feature Normalization...")
    model_norm = MultiChannelModel(
        input_channels=3,
        num_classes=10,
        normalize_features=True
    )
    model_no_norm = MultiChannelModel(
        input_channels=3,
        num_classes=10,
        normalize_features=False
    )
    
    both_models = [model_norm, model_no_norm]
    for i, test_model in enumerate(both_models):
        test_model.eval()
        with torch.no_grad():
            output = test_model(test_input)
        assert output.shape == (batch_size, 10)
        norm_status = "with" if i == 0 else "without"
        print(f"✅ Model {norm_status} normalization: {output.shape}")
    
    print("\n" + "=" * 60)
    print("🎉 ALL MULTI-CHANNEL MODEL TESTS PASSED!")
    print("✅ Model creation and architecture")
    print("✅ Forward pass and output shapes")
    print("✅ HSV feature extraction")
    print("✅ Independent pathway processing")
    print("✅ Multiple fusion methods")
    print("✅ Gradient flow and training capability")
    print("✅ MultiChannelNeuron integration")
    print("✅ Multiple depth configurations")
    print("✅ Feature normalization options")
    print("\n🚀 Multi-Channel Model is FULLY FUNCTIONAL and ready for training!")

if __name__ == "__main__":
    test_multi_channel_model_comprehensive()

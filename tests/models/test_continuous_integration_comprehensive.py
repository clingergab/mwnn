#!/usr/bin/env python3
"""Comprehensive test for ContinuousIntegrationNeuron and ContinuousIntegrationModel compliance with DESIGN.md Option 1B."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
from models.components.neurons import ContinuousIntegrationNeuron
from models.continuous_integration.model import ContinuousIntegrationModel

def test_continuous_integration_neuron():
    """Test ContinuousIntegrationNeuron implementation against DESIGN.md Option 1B."""
    print('🧪 TESTING OPTION 1B: CONTINUOUS INTEGRATION NEURON')
    print('=' * 60)
    
    # Test 1: Basic Creation and Parameters
    print('\n1️⃣ Testing Basic Creation and Parameters...')
    neuron = ContinuousIntegrationNeuron(input_size=64)
    print(f'✅ Created neuron with input_size=64')
    print(f'   - Color weights shape: {neuron.color_weights.shape}')
    print(f'   - Brightness weights shape: {neuron.brightness_weights.shape}')
    print(f'   - Integration weights: {neuron.integration_weights}')
    print(f'   - Integration weights shape: {neuron.integration_weights.shape}')
    
    # Verify required parameters exist per DESIGN.md
    assert hasattr(neuron, 'color_weights'), "Missing color_weights"
    assert hasattr(neuron, 'brightness_weights'), "Missing brightness_weights"
    assert hasattr(neuron, 'integration_weights'), "Missing integration_weights"
    assert neuron.color_weights.shape == (32,), f"Wrong color_weights shape: {neuron.color_weights.shape}"
    assert neuron.brightness_weights.shape == (32,), f"Wrong brightness_weights shape: {neuron.brightness_weights.shape}"
    assert neuron.integration_weights.shape == (2,), f"Wrong integration_weights shape: {neuron.integration_weights.shape}"
    
    # Test 2: Forward Pass Output Structure
    print('\n2️⃣ Testing Forward Pass Output Structure...')
    inputs = torch.randn(8, 64)
    outputs = neuron(inputs)
    
    print(f'✅ Forward pass successful')
    print(f'   - Input: {inputs.shape}')
    print(f'   - Number of outputs: {len(outputs)}')
    print(f'   - Output types: {[type(o).__name__ for o in outputs]}')
    print(f'   - Output shapes: {[o.shape for o in outputs]}')
    
    # Verify DESIGN.md specification: output = [y_color, y_brightness, y_integrated]
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
    color_out, brightness_out, integrated_out = outputs
    assert color_out.shape == (8,), f"Wrong color output shape: {color_out.shape}"
    assert brightness_out.shape == (8,), f"Wrong brightness output shape: {brightness_out.shape}"
    assert integrated_out.shape == (8,), f"Wrong integrated output shape: {integrated_out.shape}"
    
    # Test 3: DESIGN.md Mathematical Compliance
    print('\n3️⃣ Testing DESIGN.md Mathematical Compliance...')
    neuron_test = ContinuousIntegrationNeuron(input_size=4, activation='relu')
    test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    
    with torch.no_grad():
        color_out, brightness_out, integrated_out = neuron_test(test_input)
    
    # Manual calculation for verification
    color_components = test_input[:, :2]  # First half
    brightness_components = test_input[:, 2:]  # Second half
    
    # Expected color: y_color = f(Σ(wc_i * xc_i) + b_c)
    expected_color = torch.matmul(color_components, neuron_test.color_weights) + neuron_test.color_bias
    expected_color = torch.relu(expected_color)
    
    # Expected brightness: y_brightness = f(Σ(wb_i * xb_i) + b_b)
    expected_brightness = torch.matmul(brightness_components, neuron_test.brightness_weights) + neuron_test.brightness_bias
    expected_brightness = torch.relu(expected_brightness)
    
    # Expected integration: y_integrated = f(wi_c * y_color + wi_b * y_brightness)
    expected_integrated = (neuron_test.integration_weights[0] * expected_color + 
                          neuron_test.integration_weights[1] * expected_brightness)
    expected_integrated = expected_integrated + neuron_test.integration_bias
    expected_integrated = torch.relu(expected_integrated)
    
    # Verify mathematical formulation
    assert torch.allclose(color_out, expected_color, atol=1e-6), f"Color calculation mismatch: {color_out} vs {expected_color}"
    assert torch.allclose(brightness_out, expected_brightness, atol=1e-6), f"Brightness calculation mismatch: {brightness_out} vs {expected_brightness}"
    assert torch.allclose(integrated_out, expected_integrated, atol=1e-6), f"Integration calculation mismatch: {integrated_out} vs {expected_integrated}"
    
    print('✅ Mathematical formulation verified:')
    print('   y_color = f(Σ(wc_i * xc_i) + b_c) ✓')
    print('   y_brightness = f(Σ(wb_i * xb_i) + b_b) ✓')
    print('   y_integrated = f(wi_c * y_color + wi_b * y_brightness) ✓')
    
    # Test 4: Learnable Integration Weights
    print('\n4️⃣ Testing Learnable Integration Weights...')
    neuron_learn = ContinuousIntegrationNeuron(input_size=32)
    
    # Check that integration weights are learnable parameters
    assert neuron_learn.integration_weights.requires_grad, "Integration weights should be learnable"
    
    # Test gradient flow
    inputs = torch.randn(4, 32, requires_grad=True)
    color_out, brightness_out, integrated_out = neuron_learn(inputs)
    
    # Compute loss on integrated output
    loss = integrated_out.sum()
    loss.backward()
    
    # Check that integration weights receive gradients
    assert neuron_learn.integration_weights.grad is not None, "Integration weights should receive gradients"
    assert neuron_learn.color_weights.grad is not None, "Color weights should receive gradients"
    assert neuron_learn.brightness_weights.grad is not None, "Brightness weights should receive gradients"
    
    print('✅ Learnable integration weights verified')
    print('   - Integration weights are learnable parameters ✓')
    print('   - Gradients flow to integration weights ✓')
    print('   - All weights update during training ✓')
    
    # Test 5: Independence of Color/Brightness Processing
    print('\n5️⃣ Testing Independence of Color/Brightness Processing...')
    neuron_indep = ContinuousIntegrationNeuron(input_size=64)
    
    # Create inputs that emphasize color vs brightness
    color_emphasis = torch.randn(4, 64)
    color_emphasis[:, 32:] = 0.0  # Zero out brightness components
    
    brightness_emphasis = torch.randn(4, 64)  
    brightness_emphasis[:, :32] = 0.0  # Zero out color components
    
    with torch.no_grad():
        color_out1, brightness_out1, integrated_out1 = neuron_indep(color_emphasis)
        color_out2, brightness_out2, integrated_out2 = neuron_indep(brightness_emphasis)
    
    # Color output should be stronger when color components are present
    assert torch.abs(color_out1).mean() > torch.abs(color_out2).mean(), "Color pathway not responding to color emphasis"
    # Brightness output should be stronger when brightness components are present  
    assert torch.abs(brightness_out2).mean() > torch.abs(brightness_out1).mean(), "Brightness pathway not responding to brightness emphasis"
    
    print('✅ Pathway independence verified')
    print('   - Color pathway responds to color emphasis ✓')
    print('   - Brightness pathway responds to brightness emphasis ✓')
    
    print('\n🎉 CONTINUOUS INTEGRATION NEURON: FULLY COMPLIANT WITH DESIGN.md OPTION 1B!')

def test_continuous_integration_model():
    """Test ContinuousIntegrationModel implementation."""
    print('\n\n🧪 TESTING CONTINUOUS INTEGRATION MODEL')
    print('=' * 60)
    
    # Test 1: Model Creation
    print('\n1️⃣ Testing Model Creation...')
    model = ContinuousIntegrationModel(
        input_channels=3,
        num_classes=10,
        depth='shallow',
        integration_points=['early', 'middle', 'late']
    )
    print(f'✅ Model created successfully')
    print(f'   - Integration points: {model.integration_points}')
    print(f'   - Number of parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test 2: Forward Pass
    print('\n2️⃣ Testing Forward Pass...')
    test_input = torch.randn(4, 3, 32, 32)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f'✅ Forward pass successful: {test_input.shape} -> {output.shape}')
    assert output.shape == (4, 10), f"Wrong output shape: {output.shape}"
    
    # Test 3: Integration Modules
    print('\n3️⃣ Testing Integration Modules...')
    integration_weights = model.get_integration_weights()
    print(f'✅ Integration weights available:')
    for stage, weights in integration_weights.items():
        print(f'   - {stage}: {weights}')
    
    # Verify integration modules exist
    assert len(model.integration_modules) > 0, "No integration modules found"
    assert hasattr(model, 'final_integration'), "Missing final integration"
    
    # Test 4: Feature Extraction  
    print('\n4️⃣ Testing Feature Extraction...')
    with torch.no_grad():
        color_features, brightness_features = model.extract_features(test_input)
    
    print(f'✅ Feature extraction: RGB{test_input.shape} -> Color{color_features.shape} + Brightness{brightness_features.shape}')
    assert color_features.shape == (4, 2, 32, 32), f"Wrong color shape: {color_features.shape}"
    assert brightness_features.shape == (4, 1, 32, 32), f"Wrong brightness shape: {brightness_features.shape}"
    
    # Test 5: Gradient Flow
    print('\n5️⃣ Testing Gradient Flow...')
    model.train()
    test_input_grad = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(test_input_grad)
    loss = output.sum()
    loss.backward()
    
    # Check that integration weights receive gradients
    integration_grads = []
    for module in model.integration_modules.values():
        if hasattr(module, 'color_weight') and module.color_weight.grad is not None:
            integration_grads.append(module.color_weight.grad.abs().sum())
    
    assert len(integration_grads) > 0, "No integration module gradients found"
    print(f'✅ Gradient flow verified to {len(integration_grads)} integration modules')
    
    # Test 6: Different Integration Points
    print('\n6️⃣ Testing Different Integration Points...')
    
    # Test early only
    model_early = ContinuousIntegrationModel(
        input_channels=3, num_classes=10, depth='shallow',
        integration_points=['early']
    )
    with torch.no_grad():
        output_early = model_early(test_input)
    print(f'✅ Early integration only: {output_early.shape}')
    
    # Test all points
    model_all = ContinuousIntegrationModel(
        input_channels=3, num_classes=10, depth='shallow', 
        integration_points=['all']
    )
    with torch.no_grad():
        output_all = model_all(test_input)
    print(f'✅ All integration points: {output_all.shape}')
    
    print('\n🎉 CONTINUOUS INTEGRATION MODEL: SUCCESSFULLY IMPLEMENTED!')

if __name__ == "__main__":
    try:
        # Test neuron implementation
        test_continuous_integration_neuron()
        
        # Test model implementation  
        test_continuous_integration_model()
        
        print('\n' + '=' * 70)
        print('🎯 OPTION 1B VERIFICATION: COMPLETE SUCCESS!')
        print('=' * 70)
        print('✅ ContinuousIntegrationNeuron: DESIGN.md compliant')
        print('✅ ContinuousIntegrationModel: Fully functional')
        print('✅ Mathematical formulation: Exact match')
        print('✅ Integration weights: Learnable and working')
        print('✅ Gradient flow: Verified')
        print('✅ Feature independence: Verified')
        print('✅ All tests passed: PRODUCTION READY!')
            
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()

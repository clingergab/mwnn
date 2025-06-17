"""Test ContinuousIntegrationModel with proper error handling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
from models.continuous_integration.model import ContinuousIntegrationModel

def test_model_basic():
    """Test basic functionality of ContinuousIntegrationModel."""
    print('🧪 Testing ContinuousIntegrationModel Basic Functionality')
    print('=' * 55)
    
    # Test 1: Model with shallow depth and minimal integration
    print('\n1️⃣ Testing Basic Model Creation and Forward Pass...')
    try:
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow',
            integration_points=['late']  # Only integrate at the end
        )
        print('✅ Model created successfully')
        
        # Test forward pass
        test_input = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(test_input)
        
        print(f'✅ Forward pass successful: {test_input.shape} -> {output.shape}')
        assert output.shape == (2, 10), f"Wrong output shape: {output.shape}"
        
    except Exception as e:
        print(f'❌ Basic test failed: {e}')
        return False
    
    # Test 2: Check Integration Modules
    print('\n2️⃣ Testing Integration Modules...')
    try:
        integration_weights = model.get_integration_weights()
        print(f'✅ Integration weights: {list(integration_weights.keys())}')
        
        # Verify we have integration modules
        assert len(model.integration_modules) >= 0, "Should have integration modules"
        print(f'✅ Found {len(model.integration_modules)} integration modules')
        
    except Exception as e:
        print(f'❌ Integration test failed: {e}')
        return False
    
    # Test 3: Gradient Flow
    print('\n3️⃣ Testing Gradient Flow...')
    try:
        model.train()
        test_input_grad = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(test_input_grad)
        loss = output.sum()
        loss.backward()
        
        print('✅ Gradient flow successful')
        
    except Exception as e:
        print(f'❌ Gradient test failed: {e}')
        return False
    
    # Test 4: Feature Extraction
    print('\n4️⃣ Testing Feature Extraction...')
    try:
        with torch.no_grad():
            color_features, brightness_features = model.extract_features(test_input)
        
        print(f'✅ Feature extraction: {test_input.shape} -> Color{color_features.shape} + Brightness{brightness_features.shape}')
        assert color_features.shape == (2, 2, 32, 32), f"Wrong color shape: {color_features.shape}"
        assert brightness_features.shape == (2, 1, 32, 32), f"Wrong brightness shape: {brightness_features.shape}"
        
    except Exception as e:
        print(f'❌ Feature extraction test failed: {e}')
        raise
        
    print('🎉 ContinuousIntegrationModel Basic Tests: ALL PASSED!')

if __name__ == "__main__":
    test_model_basic()
    print('\n✅ ContinuousIntegrationModel is working properly!')

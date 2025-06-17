"""Tests for shared layer components."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch
import torch.nn as nn
from models.components.layers import (
    MultiChannelLinear, PracticalMultiWeightLinear,
    AdaptiveMultiWeightLinear
)


class TestMultiChannelLinear:
    """Test MultiChannelLinear implementation."""
    
    def test_creation(self):
        """Test layer creation."""
        layer = MultiChannelLinear(64, 32)  # in_features per pathway, out_features
        assert layer.in_features == 64
        assert layer.out_features == 32
        assert layer.color_weights.shape == (32, 64)  # (out_features, in_features)
        assert layer.brightness_weights.shape == (32, 64)
    
    def test_forward_pass(self):
        """Test forward pass."""
        layer = MultiChannelLinear(50, 30)  # 50 features per pathway, 30 output features
        color = torch.randn(16, 50)  # Each pathway gets full feature size
        brightness = torch.randn(16, 50)
        
        color_out, brightness_out = layer(color, brightness)
        assert color_out.shape == (16, 30)
        assert brightness_out.shape == (16, 30)
        assert not torch.allclose(color_out, brightness_out)
    
    def test_bias_handling(self):
        """Test bias parameter handling."""
        # With bias
        layer_bias = MultiChannelLinear(64, 32, bias=True)
        assert hasattr(layer_bias, 'color_bias')
        assert hasattr(layer_bias, 'brightness_bias')
        
        # Without bias
        layer_no_bias = MultiChannelLinear(64, 32, bias=False)
        assert not hasattr(layer_no_bias, 'color_bias')
        assert not hasattr(layer_no_bias, 'brightness_bias')
    
    def test_gradient_flow(self):
        """Test gradients flow properly."""
        layer = MultiChannelLinear(64, 32)
        color = torch.randn(8, 64, requires_grad=True)  # Fixed: should be 64 to match in_features
        brightness = torch.randn(8, 64, requires_grad=True)  # Fixed: should be 64 to match in_features
        
        color_out, brightness_out = layer(color, brightness)
        loss = color_out.sum() + brightness_out.sum()
        loss.backward()
        
        assert color.grad is not None
        assert brightness.grad is not None
        assert layer.color_weights.grad is not None
        assert layer.brightness_weights.grad is not None


class TestPracticalMultiWeightLinear:
    """Test PracticalMultiWeightLinear implementation."""
    
    def test_creation(self):
        """Test layer creation."""
        layer = PracticalMultiWeightLinear(128, 64)
        assert layer.color_weights.shape == (64, 128)  # (out_features, in_features)
        assert layer.brightness_weights.shape == (64, 128)  # (out_features, in_features)
    
    def test_forward_pass(self):
        """Test forward pass with single output."""
        layer = PracticalMultiWeightLinear(100, 50)
        inputs = torch.randn(16, 100)
        
        output = layer(inputs)
        assert output.shape == (16, 50)
    
    def test_feature_extraction(self):
        """Test internal feature extraction."""
        layer = PracticalMultiWeightLinear(64, 32)
        inputs = torch.randn(8, 64)
        
        color, brightness = layer.extract_features(inputs)
        assert color.shape == (8, 32)
        assert brightness.shape == (8, 32)
        assert not torch.allclose(color, brightness)
    
    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ['relu', 'sigmoid', 'tanh', 'gelu']:
            layer = PracticalMultiWeightLinear(64, 32, activation=activation)
            inputs = torch.randn(4, 64)
            output = layer(inputs)
            assert output.shape == (4, 32)


class TestAdaptiveMultiWeightLinear:
    """Test AdaptiveMultiWeightLinear implementation."""
    
    def test_creation(self):
        """Test layer creation with adaptive selection."""
        layer = AdaptiveMultiWeightLinear(128, 64)
        assert hasattr(layer, 'output_selector')
        assert isinstance(layer.output_selector, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass with adaptive selection."""
        layer = AdaptiveMultiWeightLinear(100, 50)
        inputs = torch.randn(16, 100)
        
        output = layer(inputs)
        assert output.shape == (16, 50)
    
    def test_output_selection(self):
        """Test that output selection produces valid weights."""
        layer = AdaptiveMultiWeightLinear(64, 32)
        inputs = torch.randn(8, 64)
        
        # Get selector output
        selector_out = layer.output_selector(inputs)
        selector_out = selector_out.view(-1, layer.out_features, 2)
        weights = torch.softmax(selector_out, dim=-1)
        
        # Check weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(8, layer.out_features), atol=1e-6)
        
        # Check weights are in [0, 1]
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
    
    def test_gradient_flow(self):
        """Test gradients flow through adaptive selection."""
        layer = AdaptiveMultiWeightLinear(64, 32)
        inputs = torch.randn(8, 64, requires_grad=True)
        
        output = layer(inputs)
        loss = output.sum()
        loss.backward()
        
        assert inputs.grad is not None
        assert layer.color_weights.grad is not None
        assert layer.brightness_weights.grad is not None
        
        # Check selector network gets gradients
        for param in layer.output_selector.parameters():
            assert param.grad is not None


class TestLayerIntegration:
    """Test layer integration with models."""
    
    def test_multi_channel_linear_in_sequential(self):
        """Test MultiChannelLinear in a sequential model."""
        # This is a special case - typically needs custom handling
        layer = MultiChannelLinear(64, 32)
        
        color = torch.randn(8, 64)  # Fixed: should be 64 to match in_features
        brightness = torch.randn(8, 64)  # Fixed: should be 64 to match in_features
        
        color_out, brightness_out = layer(color, brightness)
        assert color_out.shape == (8, 32)
        assert brightness_out.shape == (8, 32)
    
    def test_practical_linear_in_sequential(self):
        """Test PracticalMultiWeightLinear in sequential model."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            PracticalMultiWeightLinear(64, 32),
            nn.ReLU()
        )
        
        inputs = torch.randn(8, 128)
        output = model(inputs)
        assert output.shape == (8, 32)


if __name__ == "__main__":
    pytest.main([__file__])
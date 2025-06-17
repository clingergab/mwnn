"""Tests for Multi-Channel Model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch
from models.multi_channel.model import MultiChannelModel


class TestMultiChannelModel:
    """Test Multi-Channel Model implementation."""
    
    def test_model_creation(self):
        """Test model can be created with default parameters."""
        model = MultiChannelModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'
        )
        
        assert model is not None
        assert model.depth == 'shallow'
        assert model.num_classes == 10
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        batch_size = 4
        channels = 3
        height, width = 32, 32
        
        model = MultiChannelModel(
            input_channels=channels,
            num_classes=10,
            depth='shallow'
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 10)
    
    def test_pathway_outputs(self):
        """Test getting separate pathway outputs."""
        model = MultiChannelModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'
        )
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            color, brightness = model.get_pathway_outputs(x)
        
        # Check that outputs are different
        assert color.shape == brightness.shape
        assert not torch.allclose(color, brightness)
    
    def test_different_depths(self):
        """Test model creation with different depths."""
        for depth in ['shallow', 'medium', 'deep']:
            model = MultiChannelModel(
                input_channels=3,
                num_classes=10,
                depth=depth
            )
            model.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
            
            # Test forward pass
            x = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 10)
    
    def test_fusion_methods(self):
        """Test different fusion methods."""
        for fusion_method in ['concatenate', 'add', 'adaptive']:
            model = MultiChannelModel(
                input_channels=3,
                num_classes=10,
                depth='shallow',
                fusion_method=fusion_method
            )
            
            # Test forward pass
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 10)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = MultiChannelModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'
        )
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        target = torch.randint(0, 10, (2,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert torch.any(x.grad != 0)
        
        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.any(param.grad != 0)


if __name__ == "__main__":
    pytest.main([__file__])
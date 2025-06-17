"""Tests for shared building blocks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch
import torch.nn as nn
from models.components.blocks import (
    ConvBlock, ResidualBlock, AttentionBlock, 
    FusionBlock, GlobalContextBlock
)


class TestConvBlock:
    """Test ConvBlock implementation."""
    
    def test_conv_block_creation(self):
        """Test ConvBlock can be created."""
        block = ConvBlock(3, 64, kernel_size=3, stride=1, padding=1)
        assert isinstance(block.conv, nn.Conv2d)
        assert isinstance(block.bn, nn.BatchNorm2d)
        assert block.conv.in_channels == 3
        assert block.conv.out_channels == 64
    
    def test_conv_block_forward(self):
        """Test forward pass through ConvBlock."""
        block = ConvBlock(3, 32)
        x = torch.randn(2, 3, 32, 32)
        output = block(x)
        assert output.shape == (2, 32, 32, 32)
    
    def test_conv_block_activations(self):
        """Test different activation functions."""
        for activation in ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'gelu']:
            block = ConvBlock(3, 16, activation=activation)
            x = torch.randn(1, 3, 16, 16)
            output = block(x)
            assert output.shape == (1, 16, 16, 16)


class TestResidualBlock:
    """Test ResidualBlock implementation."""
    
    def test_residual_block_same_channels(self):
        """Test ResidualBlock with same input/output channels."""
        block = ResidualBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)
        assert output.shape == x.shape
    
    def test_residual_block_different_channels(self):
        """Test ResidualBlock with different input/output channels."""
        block = ResidualBlock(32, 64, stride=1)
        x = torch.randn(2, 32, 16, 16)
        output = block(x)
        assert output.shape == (2, 64, 16, 16)
    
    def test_residual_block_downsampling(self):
        """Test ResidualBlock with stride 2."""
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 16, 16)
        output = block(x)
        assert output.shape == (2, 64, 8, 8)
    
    def test_gradient_flow(self):
        """Test gradient flows through residual connection."""
        block = ResidualBlock(32, 32)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        output = block(x)
        loss = output.mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestAttentionBlock:
    """Test AttentionBlock implementation."""
    
    def test_attention_block_creation(self):
        """Test AttentionBlock creation."""
        block = AttentionBlock(64, reduction=16)
        assert hasattr(block, 'avg_pool')
        assert hasattr(block, 'fc')
    
    def test_attention_block_forward(self):
        """Test forward pass through AttentionBlock."""
        block = AttentionBlock(32, reduction=8)
        x = torch.randn(2, 32, 16, 16)
        output = block(x)
        assert output.shape == x.shape
        # Check that attention is applied (output should be different)
        assert not torch.allclose(output, x)
    
    def test_attention_weights_range(self):
        """Test that attention weights are in [0, 1] range."""
        block = AttentionBlock(16, reduction=4)
        x = torch.randn(2, 16, 8, 8)
        
        # Extract attention weights by checking the difference
        with torch.no_grad():
            output = block(x)
            # The attention weights should be between 0 and 1 (sigmoid output)
            attention_applied = output / (x + 1e-8)
            assert torch.all(attention_applied >= -0.1)  # Small tolerance for numerical errors


class TestFusionBlock:
    """Test FusionBlock implementation."""
    
    def test_fusion_block_concatenate(self):
        """Test concatenation fusion."""
        block = FusionBlock(32, fusion_type='concatenate')
        color = torch.randn(2, 32, 16, 16)
        brightness = torch.randn(2, 32, 16, 16)
        output = block(color, brightness)
        assert output.shape == (2, 64, 16, 16)
    
    def test_fusion_block_add(self):
        """Test addition fusion."""
        block = FusionBlock(32, fusion_type='add')
        color = torch.randn(2, 32, 16, 16)
        brightness = torch.randn(2, 32, 16, 16)
        output = block(color, brightness)
        assert output.shape == (2, 32, 16, 16)
        assert torch.allclose(output, color + brightness)
    
    def test_fusion_block_adaptive(self):
        """Test adaptive fusion."""
        block = FusionBlock(32, fusion_type='adaptive')
        color = torch.randn(2, 32, 16, 16)
        brightness = torch.randn(2, 32, 16, 16)
        output = block(color, brightness)
        assert output.shape == (2, 32, 16, 16)
        # Output should be different from simple addition
        assert not torch.allclose(output, color + brightness)
    
    def test_fusion_block_attention(self):
        """Test attention-based fusion."""
        block = FusionBlock(32, fusion_type='attention')
        color = torch.randn(2, 32, 16, 16)
        brightness = torch.randn(2, 32, 16, 16)
        output = block(color, brightness)
        assert output.shape == (2, 32, 16, 16)
    
    def test_invalid_fusion_type(self):
        """Test invalid fusion type raises error."""
        block = FusionBlock(32, fusion_type='invalid')
        color = torch.randn(2, 32, 16, 16)
        brightness = torch.randn(2, 32, 16, 16)
        with pytest.raises(ValueError):
            block(color, brightness)


class TestGlobalContextBlock:
    """Test GlobalContextBlock implementation."""
    
    def test_global_context_creation(self):
        """Test GlobalContextBlock creation."""
        block = GlobalContextBlock(64)
        assert hasattr(block, 'channel_add_conv')
    
    def test_global_context_forward(self):
        """Test forward pass through GlobalContextBlock."""
        block = GlobalContextBlock(32)
        x = torch.randn(2, 32, 16, 16)
        output = block(x)
        assert output.shape == x.shape
        # Output should include global context
        assert not torch.allclose(output, x)
    
    def test_global_context_different_sizes(self):
        """Test GlobalContextBlock with different input sizes."""
        block = GlobalContextBlock(16)
        
        # Test different spatial dimensions
        for h, w in [(8, 8), (16, 16), (32, 32)]:
            x = torch.randn(2, 16, h, w)
            output = block(x)
            assert output.shape == (2, 16, h, w)


if __name__ == "__main__":
    pytest.main([__file__])
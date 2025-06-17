"""Tests for Multi-Weight Neural Network neurons."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch


class TestMultiChannelNeuron:
    """Test MultiChannelNeuron implementation (Option 1A from DESIGN.md)."""
    
    def test_creation(self):
        """Test neuron creation."""
        # Import here to avoid hanging during collection
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        
        from models.components.neurons import MultiChannelNeuron
        
        neuron = MultiChannelNeuron(input_size=128)
        assert neuron.input_size == 128
        assert hasattr(neuron, 'color_weights')
        assert hasattr(neuron, 'brightness_weights')
        
    def test_forward_pass(self):
        """Test forward pass."""
        # Import here to avoid hanging during collection
        import sys
        import os
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        
        from models.components.neurons import MultiChannelNeuron
        
        neuron = MultiChannelNeuron(input_size=64)
        color_input = torch.randn(8, 64)
        brightness_input = torch.randn(8, 64)
        
        color_out, brightness_out = neuron(color_input, brightness_input)
        assert color_out.shape == (8,)  # Single neuron output
        assert brightness_out.shape == (8,)
        assert not torch.allclose(color_out, brightness_out)
        
    def test_design_compliance(self):
        """Test compliance with DESIGN.md Option 1A specification."""
        # Import here to avoid hanging during collection
        import sys
        import os
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        
        from models.components.neurons import MultiChannelNeuron
        
        # DESIGN.md Option 1A specifies:
        # y_color = f(Σ(wc_i * xc_i) + b_c)
        # y_brightness = f(Σ(wb_i * xb_i) + b_b)
        
        neuron = MultiChannelNeuron(input_size=3)
        color_input = torch.tensor([[1.0, 2.0, 3.0]])
        brightness_input = torch.tensor([[0.5, 1.0, 1.5]])
        
        color_out, brightness_out = neuron(color_input, brightness_input)
        
        # Verify mathematical formulation manually
        expected_color = torch.matmul(color_input, neuron.color_weights) + neuron.color_bias
        expected_color = torch.relu(expected_color)
        
        expected_brightness = torch.matmul(brightness_input, neuron.brightness_weights) + neuron.brightness_bias
        expected_brightness = torch.relu(expected_brightness)
        
        assert torch.allclose(color_out, expected_color, atol=1e-6)
        assert torch.allclose(brightness_out, expected_brightness, atol=1e-6)


class TestPracticalMultiWeightNeuron:
    """Test PracticalMultiWeightNeuron implementation (Option 2A from DESIGN.md)."""
    
    def test_creation(self):
        """Test neuron creation."""
        from models.components.neurons import PracticalMultiWeightNeuron
        
        neuron = PracticalMultiWeightNeuron(input_size=128)
        assert neuron.input_size == 128
        assert hasattr(neuron, 'color_weights')
        assert hasattr(neuron, 'brightness_weights')
    
    def test_forward_pass(self):
        """Test forward pass."""
        from models.components.neurons import PracticalMultiWeightNeuron
        
        neuron = PracticalMultiWeightNeuron(input_size=64)
        inputs = torch.randn(8, 64)
        
        output = neuron(inputs)
        assert output.shape == (8,)  # Single neuron output


class TestContinuousIntegrationNeuron:
    """Test ContinuousIntegrationNeuron implementation (Option 1B from DESIGN.md)."""
    
    def test_creation(self):
        """Test neuron creation."""
        from models.components.neurons import ContinuousIntegrationNeuron
        
        neuron = ContinuousIntegrationNeuron(input_size=128)
        assert neuron.input_size == 128
        assert hasattr(neuron, 'color_weights')
        assert hasattr(neuron, 'brightness_weights')
        assert hasattr(neuron, 'integration_weights')
    
    def test_forward_pass(self):
        """Test forward pass with integration."""
        from models.components.neurons import ContinuousIntegrationNeuron
        
        neuron = ContinuousIntegrationNeuron(input_size=64)
        inputs = torch.randn(8, 64)
        
        color_out, brightness_out, integrated_out = neuron(inputs)
        assert color_out.shape == (8,)
        assert brightness_out.shape == (8,)
        assert integrated_out.shape == (8,)
        assert not torch.allclose(color_out, brightness_out)


class TestCrossModalMultiWeightNeuron:
    """Test CrossModalMultiWeightNeuron implementation (Option 1C from DESIGN.md)."""
    
    def test_creation(self):
        """Test neuron creation."""
        from models.components.neurons import CrossModalMultiWeightNeuron
        
        neuron = CrossModalMultiWeightNeuron(input_size=128)
        assert neuron.input_size == 128
        assert hasattr(neuron, 'color_weights')
        assert hasattr(neuron, 'brightness_weights')
        assert hasattr(neuron, 'cross_weights_cb')
        assert hasattr(neuron, 'cross_weights_bc')
    
    def test_forward_pass(self):
        """Test forward pass with cross-modal influence."""
        from models.components.neurons import CrossModalMultiWeightNeuron
        
        neuron = CrossModalMultiWeightNeuron(input_size=64)
        color_input = torch.randn(8, 64)
        brightness_input = torch.randn(8, 64)
        
        color_out, brightness_out = neuron(color_input, brightness_input)
        assert color_out.shape == (8,)
        assert brightness_out.shape == (8,)


class TestAdaptiveMultiWeightNeuron:
    """Test AdaptiveMultiWeightNeuron implementation (Option 2B from DESIGN.md)."""
    
    def test_creation(self):
        """Test neuron creation."""
        from models.components.neurons import AdaptiveMultiWeightNeuron
        
        neuron = AdaptiveMultiWeightNeuron(input_size=128)
        assert neuron.input_size == 128
        assert hasattr(neuron, 'color_processing_weights')
        assert hasattr(neuron, 'brightness_processing_weights')
        assert hasattr(neuron, 'output_selector')
    
    def test_forward_pass(self):
        """Test forward pass with adaptive selection."""
        from models.components.neurons import AdaptiveMultiWeightNeuron
        
        neuron = AdaptiveMultiWeightNeuron(input_size=64)
        inputs = torch.randn(8, 64)
        
        output = neuron(inputs)
        assert output.shape == (8,)  # Single adaptive output


class TestAttentionMultiWeightNeuron:
    """Test AttentionMultiWeightNeuron implementation (Attention-based approach from DESIGN.md)."""
    
    def test_creation(self):
        """Test neuron creation."""
        from models.components.neurons import AttentionMultiWeightNeuron
        
        neuron = AttentionMultiWeightNeuron(input_size=128)
        assert neuron.input_size == 128
        assert hasattr(neuron, 'color_weights')
        assert hasattr(neuron, 'brightness_weights')
        assert hasattr(neuron, 'color_to_brightness_weights')
        assert hasattr(neuron, 'brightness_to_color_weights')
    
    def test_forward_pass(self):
        """Test forward pass with attention mechanism."""
        from models.components.neurons import AttentionMultiWeightNeuron
        
        neuron = AttentionMultiWeightNeuron(input_size=64)
        inputs = torch.randn(8, 64)
        
        color_out, brightness_out = neuron(inputs)
        assert color_out.shape == (8,)
        assert brightness_out.shape == (8,)
        assert not torch.allclose(color_out, brightness_out)


if __name__ == "__main__":
    pytest.main([__file__])
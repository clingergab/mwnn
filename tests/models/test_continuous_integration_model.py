"""Tests for Continuous Integration Model implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
import pytest
from models.continuous_integration.model import ContinuousIntegrationModel


class TestContinuousIntegrationModel:
    """Test Continuous Integration Model implementation (Option 1B from DESIGN.md)."""
    
    def test_creation(self):
        """Test model creation with different configurations."""
        # Basic creation
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'
        )
        assert model.input_channels == 3
        assert model.num_classes == 10
        assert hasattr(model, 'integration_modules')
        assert hasattr(model, 'final_integration')
    
    def test_forward_pass(self):
        """Test forward pass with different input sizes."""
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow',
            integration_points=['late']
        )
        model.eval()
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 10), f"Wrong output shape for batch {batch_size}"
    
    def test_feature_extraction(self):
        """Test feature extraction separates color and brightness."""
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'
        )
        
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            color, brightness = model.extract_features(x)
        
        # Check shapes according to HSV extraction
        assert color.shape == (2, 2, 32, 32), f"Wrong color shape: {color.shape}"
        assert brightness.shape == (2, 1, 32, 32), f"Wrong brightness shape: {brightness.shape}"
        
        # Check that features are different
        assert not torch.allclose(color[:, 0], color[:, 1]), "Color channels should be different"
    
    def test_integration_weights(self):
        """Test integration weights are learnable and accessible."""
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow',
            integration_points=['early', 'late']
        )
        
        # Get integration weights
        weights = model.get_integration_weights()
        assert isinstance(weights, dict), "Integration weights should be a dictionary"
        assert len(weights) > 0, "Should have integration weights"
        
        # Check that weights are properly formatted
        for stage, stage_weights in weights.items():
            assert isinstance(stage_weights, dict), f"Stage {stage} weights should be a dict"
            assert 'color' in stage_weights, f"Stage {stage} missing color weight"
            assert 'brightness' in stage_weights, f"Stage {stage} missing brightness weight"
    
    def test_different_depths(self):
        """Test model works with different depths."""
        depths = ['shallow', 'medium', 'deep']
        
        for depth in depths:
            model = ContinuousIntegrationModel(
                input_channels=3,
                num_classes=10,
                depth=depth,
                integration_points=['late']  # Use late only to avoid complexity issues
            )
            
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 10), f"Wrong output shape for depth {depth}"
    
    def test_different_integration_points(self):
        """Test different integration point configurations."""
        integration_configs = [
            ['early'],
            ['late'], 
            ['middle'],
            ['early', 'late']
        ]
        
        for config in integration_configs:
            model = ContinuousIntegrationModel(
                input_channels=3,
                num_classes=10,
                depth='shallow',
                integration_points=config
            )
            
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 10), f"Wrong output shape for config {config}"
    
    def test_gradient_flow(self):
        """Test that gradients flow through integration modules."""
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow',
            integration_points=['late']
        )
        model.train()
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that integration module parameters have gradients
        integration_has_gradients = False
        for module in model.integration_modules.values():
            if hasattr(module, 'color_weight') and module.color_weight.grad is not None:
                integration_has_gradients = True
                break
        
        # Check final integration has gradients
        final_has_gradients = (hasattr(model.final_integration, 'color_weight') and 
                              model.final_integration.color_weight.grad is not None)
        
        assert integration_has_gradients or final_has_gradients, "Integration modules should receive gradients"
    
    def test_design_md_compliance(self):
        """Test compliance with DESIGN.md Option 1B specifications."""
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow',
            integration_points=['early', 'late']
        )
        
        # Check that model has the required DESIGN.md components
        assert hasattr(model, 'extract_features'), "Should have feature extraction"
        assert hasattr(model, 'integration_modules'), "Should have integration modules"
        assert hasattr(model, 'final_integration'), "Should have final integration"
        
        # Check integration weights are learnable
        weights = model.get_integration_weights()
        for stage, stage_weights in weights.items():
            # Weights should sum to approximately 1 (softmax normalized)
            total_weight = sum(stage_weights.values())
            assert abs(total_weight - 1.0) < 0.1, f"Stage {stage} weights should sum to ~1.0, got {total_weight}"
    
    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = ContinuousIntegrationModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Should have more parameters than a basic CNN due to integration modules
        assert total_params > 100000, f"Model seems too small: {total_params} parameters"
        assert total_params < 10000000, f"Model seems too large: {total_params} parameters"
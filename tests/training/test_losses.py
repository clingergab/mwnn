"""Tests for loss functions."""

import torch
import torch.nn as nn
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training.losses import (
    MultiPathwayLoss,
    IntegrationRegularizationLoss,
    PathwayBalanceLoss,
    RobustnessLoss
)


class DummyModel(nn.Module):
    """Dummy model for testing loss functions."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # Flatten input for simplicity
        x = x.view(x.size(0), -1)
        if x.size(1) != 32:
            # Adapt to expected input size
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), 32).squeeze(1)
        return self.fc(x)


class TestMultiPathwayLoss:
    """Test multi-pathway loss function."""
    
    def test_initialization_default(self):
        """Test MultiPathwayLoss default initialization."""
        loss_fn = MultiPathwayLoss()
        
        assert isinstance(loss_fn.primary_loss, nn.CrossEntropyLoss)
        assert loss_fn.pathway_regularization == 0.1
        assert loss_fn.balance_pathways
    
    def test_initialization_custom(self):
        """Test MultiPathwayLoss custom initialization."""
        custom_loss = nn.MSELoss()
        loss_fn = MultiPathwayLoss(
            primary_loss=custom_loss,
            pathway_regularization=0.2,
            balance_pathways=False
        )
        
        assert loss_fn.primary_loss is custom_loss
        assert loss_fn.pathway_regularization == 0.2
        assert not loss_fn.balance_pathways
    
    def test_forward_without_pathway_outputs(self):
        """Test loss computation without pathway outputs."""
        loss_fn = MultiPathwayLoss()
        
        outputs = torch.randn(4, 10, requires_grad=True)  # Enable gradients
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(outputs, targets)
        
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        assert loss.item() > 0
    
    def test_forward_with_pathway_outputs(self):
        """Test loss computation with pathway outputs."""
        loss_fn = MultiPathwayLoss(pathway_regularization=0.1)
        
        outputs = torch.randn(4, 10, requires_grad=True)  # Enable gradients
        targets = torch.randint(0, 10, (4,))
        
        # Create dummy pathway outputs
        pathway_outputs = {
            'color': torch.randn(4, 8, requires_grad=True),
            'brightness': torch.randn(4, 8, requires_grad=True)
        }
        
        loss = loss_fn(outputs, targets, pathway_outputs)
        
        assert torch.is_tensor(loss)
        assert loss.requires_grad
        
        # Should be different from loss without pathway regularization
        loss_no_reg = loss_fn(outputs, targets)
        assert not torch.allclose(loss, loss_no_reg)
    
    def test_pathway_regularization_computation(self):
        """Test pathway regularization computation."""
        loss_fn = MultiPathwayLoss(pathway_regularization=0.5, balance_pathways=True)
        
        # Create pathway outputs with known properties
        color_out = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Orthogonal
        brightness_out = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # Orthogonal
        
        pathway_outputs = {'color': color_out, 'brightness': brightness_out}
        
        reg_loss = loss_fn._compute_pathway_regularization(pathway_outputs)
        
        assert torch.is_tensor(reg_loss)
        assert reg_loss.item() >= 0
    
    def test_pathway_regularization_missing_outputs(self):
        """Test regularization with missing pathway outputs."""
        loss_fn = MultiPathwayLoss()
        
        # Test with missing color pathway
        pathway_outputs = {'brightness': torch.randn(4, 8)}
        reg_loss = loss_fn._compute_pathway_regularization(pathway_outputs)
        assert reg_loss.item() == 0.0
        
        # Test with missing brightness pathway
        pathway_outputs = {'color': torch.randn(4, 8)}
        reg_loss = loss_fn._compute_pathway_regularization(pathway_outputs)
        assert reg_loss.item() == 0.0
        
        # Test with empty dict
        reg_loss = loss_fn._compute_pathway_regularization({})
        assert reg_loss.item() == 0.0


class TestIntegrationRegularizationLoss:
    """Test integration regularization loss."""
    
    def test_initialization(self):
        """Test IntegrationRegularizationLoss initialization."""
        loss_fn = IntegrationRegularizationLoss(sparsity_weight=0.02, temporal_smoothness=0.03)
        
        assert loss_fn.sparsity_weight == 0.02
        assert loss_fn.temporal_smoothness == 0.03
        assert loss_fn.previous_weights is None
    
    def test_forward_sparsity_only(self):
        """Test loss computation with sparsity regularization only."""
        loss_fn = IntegrationRegularizationLoss(sparsity_weight=0.1, temporal_smoothness=0.0)
        
        integration_weights = {
            'stage1': {'color': 0.3, 'brightness': 0.7, 'integrated': 0.0},
            'stage2': {'color': 0.8, 'brightness': 0.2, 'integrated': 0.0}
        }
        
        loss = loss_fn(integration_weights)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
    
    def test_forward_temporal_smoothness(self):
        """Test loss computation with temporal smoothness."""
        loss_fn = IntegrationRegularizationLoss(sparsity_weight=0.0, temporal_smoothness=0.1)
        
        integration_weights = {
            'stage1': {'color': 0.5, 'brightness': 0.5, 'integrated': 0.0}
        }
        
        # First call - no previous weights
        loss1 = loss_fn(integration_weights)
        assert loss1.item() == 0.0  # No previous weights to compare
        
        # Second call - should have temporal smoothness term
        integration_weights['stage1']['color'] = 0.8  # Change weights
        integration_weights['stage1']['brightness'] = 0.2
        
        loss2 = loss_fn(integration_weights)
        assert loss2.item() > 0  # Should have smoothness penalty
    
    def test_forward_combined(self):
        """Test loss computation with both sparsity and temporal smoothness."""
        loss_fn = IntegrationRegularizationLoss(sparsity_weight=0.1, temporal_smoothness=0.1)
        
        integration_weights = {
            'stage1': {'color': 0.5, 'brightness': 0.5, 'integrated': 0.0}
        }
        
        # Multiple calls to test temporal component
        loss1 = loss_fn(integration_weights)
        loss2 = loss_fn(integration_weights)  # Same weights
        
        # Should have sparsity penalty both times
        assert loss1.item() > 0
        assert loss2.item() > 0


class TestPathwayBalanceLoss:
    """Test pathway balance loss."""
    
    def test_initialization(self):
        """Test PathwayBalanceLoss initialization."""
        loss_fn = PathwayBalanceLoss(balance_weight=0.2)
        assert loss_fn.balance_weight == 0.2
    
    def test_forward_balanced(self):
        """Test loss with balanced pathway contributions."""
        loss_fn = PathwayBalanceLoss(balance_weight=1.0)
        
        # Balanced contributions
        color_contrib = torch.tensor([1.0, 2.0, 3.0])
        brightness_contrib = torch.tensor([1.0, 2.0, 3.0])
        
        loss = loss_fn(color_contrib, brightness_contrib)
        
        assert torch.is_tensor(loss)
        assert loss.item() == 0.0  # Should be zero for perfectly balanced
    
    def test_forward_imbalanced(self):
        """Test loss with imbalanced pathway contributions."""
        loss_fn = PathwayBalanceLoss(balance_weight=1.0)
        
        # Imbalanced contributions
        color_contrib = torch.tensor([3.0, 4.0, 5.0])  # Higher magnitude
        brightness_contrib = torch.tensor([1.0, 1.0, 1.0])  # Lower magnitude
        
        loss = loss_fn(color_contrib, brightness_contrib)
        
        assert loss.item() > 0  # Should penalize imbalance
    
    def test_forward_zero_contributions(self):
        """Test loss with zero contributions."""
        loss_fn = PathwayBalanceLoss()
        
        # One pathway has zero contribution
        color_contrib = torch.zeros(3)
        brightness_contrib = torch.tensor([1.0, 2.0, 3.0])
        
        loss = loss_fn(color_contrib, brightness_contrib)
        
        # Should handle division by zero gracefully
        assert torch.is_tensor(loss)
        assert torch.isfinite(loss)


class TestRobustnessLoss:
    """Test robustness loss."""
    
    def test_initialization(self):
        """Test RobustnessLoss initialization."""
        loss_fn = RobustnessLoss(robustness_weight=0.2, perturbation_type='color')
        
        assert loss_fn.robustness_weight == 0.2
        assert loss_fn.perturbation_type == 'color'
        assert isinstance(loss_fn.primary_loss, nn.CrossEntropyLoss)
    
    def test_initialization_custom_loss(self):
        """Test RobustnessLoss with custom primary loss."""
        custom_loss = nn.MSELoss()
        loss_fn = RobustnessLoss(primary_loss=custom_loss)
        
        assert loss_fn.primary_loss is custom_loss
    
    def test_forward_no_robustness(self):
        """Test loss computation with robustness weight zero."""
        model = DummyModel(num_classes=10)
        loss_fn = RobustnessLoss(robustness_weight=0.0)
        
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(model, inputs, targets)
        
        # Should be same as primary loss only
        expected_loss = loss_fn.primary_loss(model(inputs), targets)
        assert torch.allclose(loss, expected_loss)
    
    def test_forward_with_robustness(self):
        """Test loss computation with robustness term."""
        model = DummyModel(num_classes=10)
        loss_fn = RobustnessLoss(robustness_weight=0.1, perturbation_type='brightness')
        
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(model, inputs, targets)
        
        # Should be different from primary loss only
        primary_only = loss_fn.primary_loss(model(inputs), targets)
        assert not torch.allclose(loss, primary_only)
        assert loss.item() >= primary_only.item()  # Should be higher due to consistency term
    
    def test_create_perturbations_brightness(self):
        """Test brightness perturbations."""
        loss_fn = RobustnessLoss(perturbation_type='brightness')
        
        inputs = torch.ones(2, 3, 16, 16) * 0.5  # Gray images
        perturbed = loss_fn._create_perturbations(inputs)
        
        assert perturbed.shape == inputs.shape
        assert not torch.allclose(perturbed, inputs)
        assert torch.all(perturbed >= 0)  # Should be clamped to valid range
        assert torch.all(perturbed <= 1)
    
    def test_create_perturbations_color(self):
        """Test color perturbations."""
        loss_fn = RobustnessLoss(perturbation_type='color')
        
        inputs = torch.rand(2, 3, 16, 16)
        perturbed = loss_fn._create_perturbations(inputs)
        
        assert perturbed.shape == inputs.shape
        assert not torch.allclose(perturbed, inputs)
        assert torch.all(perturbed >= 0)  # Should be clamped to valid range
        assert torch.all(perturbed <= 1)
    
    def test_create_perturbations_both(self):
        """Test combined color and brightness perturbations."""
        loss_fn = RobustnessLoss(perturbation_type='both')
        
        inputs = torch.rand(2, 3, 16, 16)
        perturbed = loss_fn._create_perturbations(inputs)
        
        assert perturbed.shape == inputs.shape
        assert not torch.allclose(perturbed, inputs)
        assert torch.all(perturbed >= 0)
        assert torch.all(perturbed <= 1)


class TestLossIntegration:
    """Test integration between different loss components."""
    
    def test_loss_combination(self):
        """Test combining multiple loss functions."""
        primary_loss = MultiPathwayLoss(pathway_regularization=0.1)
        balance_loss = PathwayBalanceLoss(balance_weight=0.1)
        
        outputs = torch.randn(4, 10, requires_grad=True)  # Enable gradients
        targets = torch.randint(0, 10, (4,))
        
        pathway_outputs = {
            'color': torch.randn(4, 8, requires_grad=True),
            'brightness': torch.randn(4, 8, requires_grad=True)
        }
        
        # Compute individual losses
        primary = primary_loss(outputs, targets, pathway_outputs)
        balance = balance_loss(pathway_outputs['color'], pathway_outputs['brightness'])
        
        # Combined loss
        total_loss = primary + balance
        
        assert torch.is_tensor(total_loss)
        assert total_loss.requires_grad
        assert total_loss.item() > 0
    
    def test_gradient_flow(self):
        """Test that gradients flow through loss functions."""
        model = DummyModel(num_classes=5)
        loss_fn = MultiPathwayLoss()
        
        inputs = torch.randn(4, 3, 16, 16)
        targets = torch.randint(0, 5, (4,))
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))


@pytest.fixture
def sample_model():
    """Fixture providing a sample model for testing."""
    return DummyModel(num_classes=10)


@pytest.fixture
def sample_batch():
    """Fixture providing sample batch data."""
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.randint(0, 10, (8,))
    return inputs, targets


def test_loss_functions_device_compatibility(sample_model, sample_batch):
    """Test that loss functions work on different devices."""
    inputs, targets = sample_batch
    
    # Test on CPU
    loss_fn = MultiPathwayLoss()
    outputs = sample_model(inputs)
    loss_cpu = loss_fn(outputs, targets)
    
    assert loss_cpu.device.type == 'cpu'
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model_gpu = sample_model.cuda()
        inputs_gpu = inputs.cuda()
        targets_gpu = targets.cuda()
        loss_fn_gpu = loss_fn.cuda()
        
        outputs_gpu = model_gpu(inputs_gpu)
        loss_gpu = loss_fn_gpu(outputs_gpu, targets_gpu)
        
        assert loss_gpu.device.type == 'cuda'


def test_loss_functions_numerical_stability():
    """Test numerical stability of loss functions."""
    # Test with extreme values
    outputs = torch.tensor([[1e6, -1e6], [-1e6, 1e6]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    
    loss_fn = MultiPathwayLoss()
    loss = loss_fn(outputs, targets)
    
    # Should not be NaN or infinity
    assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__])
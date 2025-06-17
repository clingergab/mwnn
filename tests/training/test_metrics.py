"""Tests for training metrics."""

import torch
import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training.metrics import (
    calculate_accuracy,
    calculate_per_class_accuracy,
    calculate_pathway_statistics,
    evaluate_robustness,
    apply_perturbation,
    calculate_integration_entropy,
    pathway_contribution_analysis,
    MetricsTracker,
    calculate_feature_diversity,
    evaluate_color_brightness_specialization
)


class TestBasicMetrics:
    """Test basic metric calculations."""
    
    def test_calculate_accuracy_perfect(self):
        """Test accuracy calculation with perfect predictions."""
        outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        
        accuracy = calculate_accuracy(outputs, targets)
        
        assert accuracy == 100.0
    
    def test_calculate_accuracy_partial(self):
        """Test accuracy calculation with partial correct predictions."""
        outputs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])  # First is wrong, others correct
        
        accuracy = calculate_accuracy(outputs, targets)
        
        assert accuracy == pytest.approx(66.67, abs=0.01)
    
    def test_calculate_accuracy_zero(self):
        """Test accuracy calculation with all wrong predictions."""
        outputs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        targets = torch.tensor([1, 0, 1])  # All predictions are wrong
        
        accuracy = calculate_accuracy(outputs, targets)
        
        assert accuracy == 0.0
    
    def test_calculate_per_class_accuracy(self):
        """Test per-class accuracy calculation."""
        # Create outputs and targets for 3 classes
        outputs = torch.tensor([
            [0.9, 0.05, 0.05],  # Pred: 0, True: 0 ✓
            [0.1, 0.8, 0.1],    # Pred: 1, True: 1 ✓
            [0.1, 0.1, 0.8],    # Pred: 2, True: 2 ✓
            [0.8, 0.1, 0.1],    # Pred: 0, True: 1 ✗
            [0.1, 0.8, 0.1],    # Pred: 1, True: 1 ✓
        ])
        targets = torch.tensor([0, 1, 2, 1, 1])
        
        per_class_acc = calculate_per_class_accuracy(outputs, targets, num_classes=3)
        
        assert per_class_acc[0] == 100.0  # Class 0: 1/1 correct
        assert per_class_acc[1] == pytest.approx(66.67, abs=0.01)  # Class 1: 2/3 correct
        assert per_class_acc[2] == 100.0  # Class 2: 1/1 correct
    
    def test_calculate_per_class_accuracy_missing_class(self):
        """Test per-class accuracy with missing classes."""
        outputs = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
        targets = torch.tensor([0, 0])  # Only class 0
        
        per_class_acc = calculate_per_class_accuracy(outputs, targets, num_classes=3)
        
        assert per_class_acc[0] == 100.0  # Class 0: all correct
        assert per_class_acc[1] == 0.0    # Class 1: no samples
        assert per_class_acc[2] == 0.0    # Class 2: no samples


class TestPerturbationFunctions:
    """Test perturbation application functions."""
    
    def test_apply_perturbation_brightness(self):
        """Test brightness perturbation."""
        inputs = torch.ones(2, 3, 4, 4) * 0.5  # Gray images
        
        perturbed = apply_perturbation(inputs, 'brightness')
        
        assert perturbed.shape == inputs.shape
        assert not torch.allclose(perturbed, inputs)
        assert torch.all(perturbed >= 0)
        assert torch.all(perturbed <= 1)
    
    def test_apply_perturbation_color(self):
        """Test color perturbation."""
        inputs = torch.rand(2, 3, 4, 4)
        
        perturbed = apply_perturbation(inputs, 'color')
        
        assert perturbed.shape == inputs.shape
        assert not torch.allclose(perturbed, inputs)
        assert torch.all(perturbed >= 0)
        assert torch.all(perturbed <= 1)
    
    def test_apply_perturbation_noise(self):
        """Test noise perturbation."""
        inputs = torch.zeros(2, 3, 4, 4)
        
        perturbed = apply_perturbation(inputs, 'noise')
        
        assert perturbed.shape == inputs.shape
        assert not torch.allclose(perturbed, inputs)
        assert torch.all(perturbed >= 0)
        assert torch.all(perturbed <= 1)
    
    def test_apply_perturbation_unknown(self):
        """Test unknown perturbation type."""
        inputs = torch.rand(2, 3, 4, 4)
        
        perturbed = apply_perturbation(inputs, 'unknown')
        
        # Should return unchanged input
        assert torch.allclose(perturbed, inputs)


class TestAdvancedMetrics:
    """Test advanced metric calculations."""
    
    def test_calculate_integration_entropy(self):
        """Test integration entropy calculation."""
        # Uniform distribution (maximum entropy)
        weights_uniform = {'color': 0.33, 'brightness': 0.33, 'integrated': 0.34}
        entropy_uniform = calculate_integration_entropy(weights_uniform)
        
        # Single pathway (minimum entropy)
        weights_single = {'color': 1.0, 'brightness': 0.0, 'integrated': 0.0}
        entropy_single = calculate_integration_entropy(weights_single)
        
        assert entropy_uniform > entropy_single
        assert entropy_single >= 0
        assert entropy_uniform >= 0
    
    def test_calculate_feature_diversity(self):
        """Test feature diversity calculation."""
        # Identical features (minimum diversity)
        features_identical = torch.ones(4, 8)
        diversity_min = calculate_feature_diversity(features_identical)
        
        # Orthogonal features (higher diversity)
        features_orthogonal = torch.tensor([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.]
        ])
        diversity_high = calculate_feature_diversity(features_orthogonal)
        
        assert diversity_high > diversity_min
        assert diversity_min >= 0
        assert diversity_high <= 2.0  # Maximum distance between normalized vectors


class TestMetricsTracker:
    """Test metrics tracking functionality."""
    
    def test_initialization(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()
        
        assert tracker.metrics == {}
        assert tracker.history == {}
    
    def test_update_metric(self):
        """Test updating metrics."""
        tracker = MetricsTracker()
        
        tracker.update('accuracy', 85.5)
        tracker.update('accuracy', 87.2)
        tracker.update('loss', 0.45)
        
        assert len(tracker.metrics['accuracy']) == 2
        assert len(tracker.metrics['loss']) == 1
        assert tracker.metrics['accuracy'] == [85.5, 87.2]
        assert tracker.metrics['loss'] == [0.45]
    
    def test_update_metric_with_step(self):
        """Test updating metrics with step information."""
        tracker = MetricsTracker()
        
        tracker.update('accuracy', 85.5, step=100)
        tracker.update('accuracy', 87.2, step=200)
        
        assert tracker.history['accuracy'] == [(100, 85.5), (200, 87.2)]
    
    def test_get_average(self):
        """Test getting metric averages."""
        tracker = MetricsTracker()
        
        tracker.update('accuracy', 80.0)
        tracker.update('accuracy', 85.0)
        tracker.update('accuracy', 90.0)
        
        avg_all = tracker.get_average('accuracy')
        avg_last_2 = tracker.get_average('accuracy', last_n=2)
        
        assert avg_all == 85.0
        assert avg_last_2 == 87.5
    
    def test_get_average_nonexistent(self):
        """Test getting average of non-existent metric."""
        tracker = MetricsTracker()
        
        avg = tracker.get_average('nonexistent')
        
        assert avg == 0.0
    
    def test_reset_specific_metric(self):
        """Test resetting specific metric."""
        tracker = MetricsTracker()
        
        tracker.update('accuracy', 85.0)
        tracker.update('loss', 0.5)
        
        tracker.reset('accuracy')
        
        assert tracker.metrics['accuracy'] == []
        assert len(tracker.metrics['loss']) == 1
    
    def test_reset_all_metrics(self):
        """Test resetting all metrics."""
        tracker = MetricsTracker()
        
        tracker.update('accuracy', 85.0)
        tracker.update('loss', 0.5)
        
        tracker.reset()
        
        assert tracker.metrics['accuracy'] == []
        assert tracker.metrics['loss'] == []
    
    def test_get_all_averages(self):
        """Test getting all metric averages."""
        tracker = MetricsTracker()
        
        tracker.update('accuracy', 80.0)
        tracker.update('accuracy', 90.0)
        tracker.update('loss', 0.4)
        tracker.update('loss', 0.6)
        
        all_averages = tracker.get_all_averages()
        
        assert all_averages['accuracy'] == 85.0
        assert all_averages['loss'] == 0.5


class TestModelSpecificMetrics:
    """Test metrics that require model interaction."""
    
    def create_mock_model(self, has_pathway_outputs=False, has_pathway_contributions=False):
        """Create a mock model for testing."""
        class MockModel:
            def __init__(self):
                self.eval_called = False
                
            def eval(self):
                self.eval_called = True
                
        model = MockModel()
        
        if has_pathway_outputs:
            def get_pathway_outputs(inputs):
                return (
                    torch.randn(inputs.size(0), 4),  # color output
                    torch.randn(inputs.size(0), 4)   # brightness output
                )
            model.get_pathway_outputs = get_pathway_outputs
        
        if has_pathway_contributions:
            def get_pathway_contributions(inputs):
                return (
                    torch.randn(inputs.size(0), 8),  # color contributions
                    torch.randn(inputs.size(0), 8)   # brightness contributions
                )
            model.get_pathway_contributions = get_pathway_contributions
        
        return model
    
    def create_mock_dataloader(self, num_batches=2):
        """Create a mock dataloader for testing."""
        batches = []
        for i in range(num_batches):
            inputs = torch.randn(2, 3, 8, 8)
            targets = torch.randint(0, 3, (2,))
            batches.append((inputs, targets))
        
        dataloader = Mock()
        dataloader.__iter__ = Mock(return_value=iter(batches))
        return dataloader
    
    def test_calculate_pathway_statistics_with_pathways(self):
        """Test pathway statistics calculation with pathway outputs."""
        model = self.create_mock_model(has_pathway_outputs=True)
        dataloader = self.create_mock_dataloader(num_batches=2)
        device = torch.device('cpu')
        
        stats = calculate_pathway_statistics(model, dataloader, device)
        
        assert 'color' in stats
        assert 'brightness' in stats
        assert 'pathway_correlation' in stats
        
        # Check that required statistics are present
        for pathway in ['color', 'brightness']:
            assert 'mean' in stats[pathway]
            assert 'std' in stats[pathway]
            assert 'min' in stats[pathway]
            assert 'max' in stats[pathway]
            assert 'sparsity' in stats[pathway]
    
    def test_calculate_pathway_statistics_without_pathways(self):
        """Test pathway statistics with model that doesn't support pathways."""
        model = self.create_mock_model(has_pathway_outputs=False)
        dataloader = self.create_mock_dataloader()
        device = torch.device('cpu')
        
        stats = calculate_pathway_statistics(model, dataloader, device)
        
        assert stats == {}
    
    def test_evaluate_robustness(self):
        """Test robustness evaluation."""
        # Create a simple model that always predicts class 0
        class MockRobustnessModel:
            def eval(self):
                pass
            
            def __call__(self, x):
                return torch.zeros(x.size(0), 3)  # Always predict class 0
        
        model = MockRobustnessModel()
        
        # Create data where all targets are class 0 (so accuracy should be 100%)
        data = [(torch.randn(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))]
        
        class MockDataLoader:
            def __iter__(self):
                return iter(data)
        
        dataloader = MockDataLoader()
        device = torch.device('cpu')
        
        results = evaluate_robustness(model, dataloader, device, ['brightness', 'color'])
        
        assert 'brightness_robustness' in results
        assert 'color_robustness' in results
        assert all(isinstance(v, float) for v in results.values())
    
    def test_pathway_contribution_analysis_with_contributions(self):
        """Test pathway contribution analysis."""
        model = self.create_mock_model(has_pathway_contributions=True)
        dataloader = self.create_mock_dataloader()
        device = torch.device('cpu')
        
        analysis = pathway_contribution_analysis(model, dataloader, device)
        
        assert 'color_contributions' in analysis
        assert 'brightness_contributions' in analysis
        assert isinstance(analysis['color_contributions'], np.ndarray)
        assert isinstance(analysis['brightness_contributions'], np.ndarray)
    
    def test_pathway_contribution_analysis_without_contributions(self):
        """Test pathway contribution analysis with unsupported model."""
        model = self.create_mock_model(has_pathway_contributions=False)
        dataloader = self.create_mock_dataloader()
        device = torch.device('cpu')
        
        analysis = pathway_contribution_analysis(model, dataloader, device)
        
        assert analysis == {}
    
    def test_evaluate_color_brightness_specialization(self):
        """Test color-brightness specialization evaluation."""
        model = self.create_mock_model(has_pathway_outputs=True)
        dataloader = self.create_mock_dataloader(num_batches=1)
        device = torch.device('cpu')
        
        results = evaluate_color_brightness_specialization(model, dataloader, device)
        
        expected_keys = [
            'color_pathway_color_sensitivity',
            'color_pathway_brightness_sensitivity',
            'brightness_pathway_color_sensitivity',
            'brightness_pathway_brightness_sensitivity',
            'color_specialization_ratio',
            'brightness_specialization_ratio'
        ]
        
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], float)


class TestMetricsIntegration:
    """Test integration between different metric components."""
    
    def test_metrics_tracker_with_calculated_metrics(self):
        """Test using MetricsTracker with calculated metrics."""
        tracker = MetricsTracker()
        
        # Simulate training loop
        for epoch in range(3):
            # Simulate batch metrics
            outputs = torch.randn(10, 5)
            targets = torch.randint(0, 5, (10,))
            
            accuracy = calculate_accuracy(outputs, targets)
            tracker.update('accuracy', accuracy, step=epoch)
        
        # Check that metrics are tracked
        assert len(tracker.metrics['accuracy']) == 3
        avg_accuracy = tracker.get_average('accuracy')
        assert 0 <= avg_accuracy <= 100
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive model evaluation."""
        # Create synthetic evaluation data
        outputs = torch.randn(20, 10)
        targets = torch.randint(0, 10, (20,))
        
        # Calculate multiple metrics
        accuracy = calculate_accuracy(outputs, targets)
        per_class_acc = calculate_per_class_accuracy(outputs, targets, num_classes=10)
        
        # Test feature diversity
        features = torch.randn(20, 16)
        diversity = calculate_feature_diversity(features)
        
        # Integration entropy
        weights = {'color': 0.4, 'brightness': 0.6}
        entropy = calculate_integration_entropy(weights)
        
        # All metrics should be valid numbers
        assert 0 <= accuracy <= 100
        assert all(0 <= acc <= 100 for acc in per_class_acc.values())
        assert diversity >= 0
        assert entropy >= 0


@pytest.fixture
def sample_outputs_targets():
    """Fixture providing sample outputs and targets."""
    outputs = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    return outputs, targets


def test_accuracy_calculation_consistency(sample_outputs_targets):
    """Test that accuracy calculation is consistent."""
    outputs, targets = sample_outputs_targets
    
    # Calculate accuracy multiple times
    acc1 = calculate_accuracy(outputs, targets)
    acc2 = calculate_accuracy(outputs, targets)
    
    assert acc1 == acc2


def test_metrics_device_compatibility():
    """Test that metrics work on different devices."""
    outputs = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    
    # CPU
    acc_cpu = calculate_accuracy(outputs, targets)
    
    # GPU (if available)
    if torch.cuda.is_available():
        outputs_gpu = outputs.cuda()
        targets_gpu = targets.cuda()
        acc_gpu = calculate_accuracy(outputs_gpu, targets_gpu)
        
        assert acc_cpu == acc_gpu


def test_metrics_numerical_stability():
    """Test numerical stability of metrics."""
    # Test with extreme values
    outputs_extreme = torch.tensor([[1e6, -1e6, 0], [-1e6, 1e6, 0]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    
    accuracy = calculate_accuracy(outputs_extreme, targets)
    
    # Should not be NaN and should be valid percentage
    assert not np.isnan(accuracy)
    assert 0 <= accuracy <= 100


if __name__ == '__main__':
    pytest.main([__file__])
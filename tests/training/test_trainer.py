"""Tests for training utilities - simplified version."""

import torch
import torch.nn as nn
import pytest
import tempfile
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from training.trainer import Trainer
from models.base import BaseMultiWeightModel


class SimpleDummyModel(nn.Module):
    """Simple dummy model that doesn't inherit from BaseMultiWeightModel."""
    
    def __init__(self, input_size=32, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.input_size = input_size
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        if x.size(1) != self.input_size:
            # Adapt to expected input size
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), self.input_size).squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


class DummyMultiWeightModel(BaseMultiWeightModel):
    """Dummy model for testing trainer."""
    
    def __init__(self, input_size=32, num_classes=10):
        # Set attributes before calling super().__init__()
        self.input_size = input_size
        self.test_num_classes = num_classes
        super().__init__(input_channels=3, num_classes=num_classes)
    
    def _build_model(self):
        """Build the dummy model architecture."""
        self.fc = nn.Linear(self.input_size, self.test_num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        if x.size(1) != self.input_size:
            # Adapt to expected input size
            x = nn.functional.adaptive_avg_pool1d(x.unsqueeze(1), self.input_size).squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


class TestBasicTrainerFunctionality:
    """Test basic trainer functionality with simplified model."""
    
    def test_trainer_creation(self):
        """Test basic trainer creation."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model, 
                device,
                log_dir=os.path.join(temp_dir, 'logs'),
                checkpoint_dir=os.path.join(temp_dir, 'checkpoints')
            )
            
            assert trainer.device == device
            assert trainer.current_epoch == 0
            assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_train_epoch_basic(self):
        """Test basic training epoch."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model, 
                device,
                log_dir=os.path.join(temp_dir, 'logs'),
                checkpoint_dir=os.path.join(temp_dir, 'checkpoints')
            )
            
            # Create sample dataset
            dataset = TensorDataset(
                torch.randn(16, 3, 32, 32),
                torch.randint(0, 10, (16,))
            )
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            
            metrics = trainer.train_epoch(dataloader, criterion)
            
            assert 'loss' in metrics
            assert 'accuracy' in metrics
            assert isinstance(metrics['loss'], float)
            assert isinstance(metrics['accuracy'], float)
            assert metrics['loss'] > 0
            assert 0 <= metrics['accuracy'] <= 100
    
    def test_validate_basic(self):
        """Test basic validation."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model, 
                device,
                log_dir=os.path.join(temp_dir, 'logs'),
                checkpoint_dir=os.path.join(temp_dir, 'checkpoints')
            )
            
            # Create validation dataset
            val_dataset = TensorDataset(
                torch.randn(12, 3, 32, 32),
                torch.randint(0, 10, (12,))
            )
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            
            metrics = trainer.validate(val_loader, criterion)
            
            assert 'loss' in metrics
            assert 'accuracy' in metrics
            assert isinstance(metrics['loss'], float)
            assert isinstance(metrics['accuracy'], float)


class TestTrainerOptimizers:
    """Test trainer with different optimizers."""
    
    def test_adam_optimizer(self):
        """Test trainer with Adam optimizer."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        trainer = Trainer(model, device, optimizer_name='adam')
        
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_sgd_optimizer(self):
        """Test trainer with SGD optimizer."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        trainer = Trainer(model, device, optimizer_name='sgd')
        
        assert isinstance(trainer.optimizer, torch.optim.SGD)


class TestTrainerCheckpoints:
    """Test checkpoint functionality."""
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(model, device, checkpoint_dir=temp_dir)
            trainer.current_epoch = 5
            trainer.best_metric = 85.5
            
            checkpoint_path = trainer.save_checkpoint(
                epoch=5, 
                metrics={'accuracy': 85.5}
            )
            
            assert checkpoint_path.exists()
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint['epoch'] == 5
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        model = SimpleDummyModel()
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(model, device, checkpoint_dir=temp_dir)
            
            # Save a checkpoint first
            trainer.current_epoch = 3
            checkpoint_path = trainer.save_checkpoint(
                epoch=3, 
                metrics={'accuracy': 78.5}
            )
            
            # Create new trainer and load checkpoint
            new_trainer = Trainer(model, device, checkpoint_dir=temp_dir)
            metrics = new_trainer.load_checkpoint(checkpoint_path)
            
            # When loading checkpoint, current_epoch is set to checkpoint_epoch + 1 for training resumption
            assert new_trainer.current_epoch == 4  # 3 + 1 for next epoch
            assert 'accuracy' in metrics


class TestMultiWeightModelIntegration:
    """Test integration with MultiWeight models."""
    
    def test_multiweight_model_training(self):
        """Test trainer with MultiWeight model."""
        model = DummyMultiWeightModel()
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(
                model, 
                device,
                log_dir=os.path.join(temp_dir, 'logs'),
                checkpoint_dir=os.path.join(temp_dir, 'checkpoints')
            )
            
            # Create sample dataset
            dataset = TensorDataset(
                torch.randn(8, 3, 32, 32),
                torch.randint(0, 10, (8,))
            )
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            
            # Should work without errors
            metrics = trainer.train_epoch(dataloader, criterion)
            
            assert 'loss' in metrics
            assert 'accuracy' in metrics


def test_gradient_flow():
    """Test that gradients flow properly during training."""
    model = SimpleDummyModel()
    device = torch.device('cpu')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = Trainer(model, device, log_dir=temp_dir, checkpoint_dir=temp_dir)
        
        # Create sample data
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.randint(0, 10, (4,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        trainer.optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))


def test_device_compatibility():
    """Test that trainer works on CPU."""
    model = SimpleDummyModel()
    device = torch.device('cpu')
    trainer = Trainer(model, device)
    
    # Model should be on the correct device
    assert next(trainer.model.parameters()).device == device
    
    # Optimizer should work
    assert hasattr(trainer.optimizer, 'step')


def test_trainer_reproducibility():
    """Test training reproducibility with seed."""
    torch.manual_seed(42)
    
    model1 = SimpleDummyModel()
    trainer1 = Trainer(model1, torch.device('cpu'))
    
    torch.manual_seed(42)
    
    model2 = SimpleDummyModel()
    trainer2 = Trainer(model2, torch.device('cpu'))
    
    # Models should have identical initial parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__])
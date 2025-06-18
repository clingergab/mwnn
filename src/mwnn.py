"""
MWNN - Multi-Weight Neural Networks
API for ImageNet Classification

Simple usage:
    from mwnn import MWNN
    
    # Load data
    train_loader, val_loader = MWNN.load_imagenet_data(data_path, batch_size=64)
    
    # Create model
    model = MWNN(num_classes=1000, depth='deep', device='auto')
    
    # Train
    model.fit(train_loader, val_loader, epochs=30)
    
    # Evaluate
    accuracy = model.evaluate(val_loader)
    
    # Save/Load
    model.save('best_model.pth')
    model = MWNN.load('best_model.pth')
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Union
from datetime import datetime

try:
    from .models.continuous_integration import ContinuousIntegrationModel
    from .preprocessing.imagenet_dataset import create_imagenet_separate_pathway_dataloaders
    from .training.trainer import MWNNTrainer
    from .utils.device import get_optimal_device
except ImportError:
    # Fall back to absolute imports when run as script
    from models.continuous_integration import ContinuousIntegrationModel
    from preprocessing.imagenet_dataset import create_imagenet_separate_pathway_dataloaders
    from training.trainer import MWNNTrainer
    from utils.device import get_optimal_device


class MWNN:
    """
    Multi-Weight Neural Network - Clean API for ImageNet Classification
    
    A simplified interface for training and using MWNN models with dual pathways
    (RGB + brightness) for enhanced image classification.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        depth: str = 'deep',
        device: Union[str, torch.device] = 'auto',
        base_channels: int = 64
    ):
        """
        Initialize MWNN model.
        
        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet)
            depth: Model depth - 'shallow', 'medium', or 'deep'
            device: Device to use - 'auto', 'cpu', 'cuda', or torch.device
            base_channels: Base number of channels for the model
        """
        self.num_classes = num_classes
        self.depth = depth
        self.base_channels = base_channels
        
        # Set device
        if device == 'auto':
            self.device = get_optimal_device()
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self._create_model()
        
        # Training state
        self.trainer = None
        self.is_trained = False
        
    def _create_model(self):
        """Create the underlying MWNN model."""
        # Create a simple config for the model
        self.config = {
            'depth': self.depth,
            'base_channels': self.base_channels,
            'num_classes': self.num_classes
        }
        
        self.model = ContinuousIntegrationModel(
            num_classes=self.num_classes,
            depth=self.depth,
            base_channels=self.base_channels
        ).to(self.device)
        
    @staticmethod
    def load_imagenet_data(
        data_path: Union[str, Path],
        batch_size: int = 64,
        num_workers: int = 4,
        subset_size: Optional[int] = None
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Load ImageNet data with dual pathways (RGB + brightness).
        
        Args:
            data_path: Path to ImageNet data directory (should contain val_images/ and ILSVRC2013_devkit/)
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            subset_size: Optional subset size for testing
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        data_path = Path(data_path)
        
        # Look for standard ImageNet structure
        val_images_dir = data_path / 'val_images'
        devkit_dir = data_path / 'ILSVRC2013_devkit'
        
        # Determine the correct data_dir to pass to the dataloader function
        # The dataloader function will append "{split}_images" to data_dir
        if val_images_dir.exists() and devkit_dir.exists():
            # Standard structure: data_path contains val_images/ and ILSVRC2013_devkit/
            data_dir = str(data_path)
            devkit_path = str(devkit_dir)
        else:
            # Alternative structure: check if data_path directly contains images
            # Try to find devkit directory
            possible_devkit_paths = [
                data_path / 'ILSVRC2013_devkit',
                data_path.parent / 'ILSVRC2013_devkit',
                data_path
            ]
            
            devkit_path = None
            for potential_devkit in possible_devkit_paths:
                if potential_devkit.exists():
                    devkit_path = str(potential_devkit)
                    break
            
            if devkit_path is None:
                devkit_path = str(data_path)  # Fallback
            
            # For data_dir, use the parent if val_images exists, otherwise use data_path
            if val_images_dir.exists():
                data_dir = str(data_path)
            else:
                data_dir = str(data_path.parent) if (data_path.parent / 'val_images').exists() else str(data_path)
            
        return create_imagenet_separate_pathway_dataloaders(
            data_dir=data_dir,
            devkit_dir=devkit_path,
            batch_size=batch_size,
            num_workers=num_workers,
            load_subset=subset_size
        )
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 30,
        learning_rate: float = 0.002,
        save_checkpoints: bool = True,
        checkpoint_dir: Union[str, Path] = 'checkpoints'
    ) -> dict:
        """
        Train the MWNN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        # Initialize trainer
        self.trainer = MWNNTrainer(
            model=self.model,
            device=self.device,
            learning_rate=learning_rate,
            checkpoint_dir=Path(checkpoint_dir)
        )
        
        # Train the model
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            save_checkpoints=save_checkpoints
        )
        
        self.is_trained = True
        return history
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained and self.trainer is None:
            raise ValueError("Model must be trained or loaded before evaluation")
            
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                # Unpack dual inputs
                rgb_data, brightness_data = data
                rgb_data = rgb_data.to(self.device)
                brightness_data = brightness_data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(rgb_data, brightness_data)
                loss = criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def predict(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Make predictions on data.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Tensor of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                # Unpack dual inputs
                rgb_data, brightness_data = data
                rgb_data = rgb_data.to(self.device)
                brightness_data = brightness_data.to(self.device)
                
                # Forward pass
                output = self.model(rgb_data, brightness_data)
                pred = output.argmax(dim=1)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions)
    
    def save(self, path: Union[str, Path]):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_classes': self.num_classes,
            'depth': self.depth,
            'base_channels': self.base_channels,
            'is_trained': self.is_trained,
            'save_timestamp': datetime.now().isoformat()
        }
        
        if self.trainer and hasattr(self.trainer, 'optimizer'):
            save_dict['optimizer_state_dict'] = self.trainer.optimizer.state_dict()
            
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Union[str, torch.device] = 'auto'):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded MWNN instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create new instance
        instance = cls(
            num_classes=checkpoint['num_classes'],
            depth=checkpoint['depth'],
            device=device,
            base_channels=checkpoint['base_channels']
        )
        
        # Load model state
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.config = checkpoint['config']
        instance.is_trained = checkpoint.get('is_trained', True)
        
        print(f"Model loaded from {path}")
        return instance
    
    def summary(self):
        """Print model summary."""
        print("MWNN Model Summary:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Depth: {self.depth}")
        print(f"  Base Channels: {self.base_channels}")
        print(f"  Device: {self.device}")
        print(f"  Trained: {self.is_trained}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        
    def __repr__(self):
        return f"MWNN(num_classes={self.num_classes}, depth='{self.depth}', device='{self.device}')"


# Convenience functions for quick usage
def load_imagenet_data(data_path, batch_size=64, **kwargs):
    """Convenience function to load ImageNet data."""
    return MWNN.load_imagenet_data(data_path, batch_size, **kwargs)

def create_model(num_classes=1000, depth='deep', **kwargs):
    """Convenience function to create MWNN model."""
    return MWNN(num_classes=num_classes, depth=depth, **kwargs)

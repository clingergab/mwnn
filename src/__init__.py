"""
MWNN - Multi-Weight Neural Networks
Clean, Keras-like API for dual-pathway image classification

Main Components:
- MWNN: Main model class with simple fit/evaluate/predict interface
- ContinuousIntegrationModel: Core dual-pathway neural network
- MWNNTrainer: Clean trainer with progress bars
- Preprocessing utilities for ImageNet

Example usage:
    from mwnn import MWNN
    
    # Load data
    train_loader, val_loader = MWNN.load_imagenet_data('/path/to/imagenet')
    
    # Create and train model
    model = MWNN(num_classes=1000, depth='deep')
    history = model.fit(train_loader, val_loader, epochs=30)
    
    # Evaluate
    results = model.evaluate(val_loader)
    print(f"Accuracy: {results['accuracy']:.2f}%")
"""

from .mwnn import MWNN, load_imagenet_data, create_model
from .models.continuous_integration import ContinuousIntegrationModel
from .training.trainer import MWNNTrainer
from .utils.device import get_optimal_device

__version__ = "2.0.0"
__author__ = "MWNN Team"

__all__ = [
    'MWNN',
    'ContinuousIntegrationModel', 
    'MWNNTrainer',
    'load_imagenet_data',
    'create_model',
    'get_optimal_device'
]
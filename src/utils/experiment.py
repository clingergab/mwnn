"""Experiment running utilities for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import json
import time

# Local imports will be done inside functions to avoid circular imports
from .config import load_config, save_config


logger = logging.getLogger(__name__)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Create a model instance from configuration."""
    # Local imports to avoid circular dependencies
    from src.models import (
        MultiChannelModel, ContinuousIntegrationModel, 
        CrossModalModel, AttentionBasedModel, SingleOutputModel
    )
    
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'MultiChannelModel')
    model_params = model_config.get('params', {})
    
    model_classes = {
        'MultiChannelModel': MultiChannelModel,
        'ContinuousIntegrationModel': ContinuousIntegrationModel,
        'CrossModalModel': CrossModalModel,
        'AttentionBasedModel': AttentionBasedModel,
        'SingleOutputModel': SingleOutputModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = model_classes[model_type]
    return model_class(**model_params)


def create_data_loaders_from_config(config: Dict[str, Any]) -> tuple:
    """Create data loaders from configuration."""
    # Local imports to avoid circular dependencies
    from src.preprocessing import get_data_loader
    
    data_config = config.get('data', {})
    dataset_name = data_config.get('dataset', 'CIFAR10')
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    # This is a simplified version - in practice you'd want to support
    # different datasets and their specific configurations
    train_loader = get_data_loader(
        dataset_name=dataset_name,
        split='train',
        batch_size=batch_size,
        feature_extraction_method=data_config.get('feature_extraction', 'hsv'),
        augment=data_config.get('augmentation', True)
    )
    
    val_loader = get_data_loader(
        dataset_name=dataset_name,
        split='val',
        batch_size=batch_size,
        feature_extraction_method=data_config.get('feature_extraction', 'hsv'),
        augment=False
    )
    
    test_loader = get_data_loader(
        dataset_name=dataset_name,
        split='test',
        batch_size=batch_size,
        feature_extraction_method=data_config.get('feature_extraction', 'hsv'),
        augment=False
    )
    
    return train_loader, val_loader, test_loader


def run_experiment(config: Dict[str, Any], 
                   experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """Run a complete experiment from configuration.
    
    Args:
        config: Experiment configuration dictionary
        experiment_name: Optional name for the experiment
        
    Returns:
        Dictionary containing experiment results
    """
    # Local imports to avoid circular dependencies
    from src.training.trainer import Trainer
    
    if experiment_name is None:
        experiment_name = f"experiment_{int(time.time())}"
    
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Create output directory
    output_dir = Path(f"experiments/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, output_dir / "config.yaml")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model_from_config(config)
    logger.info(f"Created model: {type(model).__name__}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders_from_config(config)
    
    # Create trainer
    training_config = config.get('training', {})
    trainer = Trainer(
        model=model,
        device=device,
        optimizer_name=training_config.get('optimizer', 'adamw'),
        learning_rate=training_config.get('learning_rate', 1e-3),
        scheduler_name=training_config.get('scheduler', 'cosine'),
        mixed_precision=training_config.get('mixed_precision', False),
        log_dir=str(output_dir / "logs"),
        checkpoint_dir=str(output_dir / "checkpoints")
    )
    
    # Train model
    num_epochs = training_config.get('num_epochs', 100)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        criterion=criterion
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_loader, criterion)
    
    # Compile results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'training_history': history,
        'test_results': test_results,
        'training_time': training_time,
        'model_info': {
            'type': type(model).__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Experiment completed: {experiment_name}")
    logger.info(f"Test accuracy: {test_results.get('accuracy', 'N/A'):.2f}%")
    
    return results


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    return load_config(config_path)

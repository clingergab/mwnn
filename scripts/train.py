#!/usr/bin/env python3
"""Unified training script for all Multi-Weight Neural Network models."""

import argparse
import logging
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.multi_channel.model import MultiChannelModel
from src.models.continuous_integration.model import ContinuousIntegrationModel
from src.models.cross_modal.model import CrossModalModel
from src.models.single_output.model import SingleOutputModel
from src.models.attention_based.model import AttentionBasedModel
from src.training.trainer import Trainer, MultiStageTrainer
from src.utils.config import load_config, get_model_config


def setup_logging(log_dir: Path, model_name: str):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_type: str, config: dict):
    """Create model based on type and configuration."""
    model_classes = {
        'multi_channel': MultiChannelModel,
        'continuous_integration': ContinuousIntegrationModel,
        'cross_modal': CrossModalModel,
        'single_output': SingleOutputModel,
        'attention_based': AttentionBasedModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = model_classes[model_type]
    return model_class(**config['model']['params'])


def create_data_loaders(config: dict):
    """Create data loaders based on configuration."""
    data_config = config['data']
    
    # Define transforms
    transform_list = []
    
    if data_config.get('augmentation', {}).get('random_crop', True):
        transform_list.append(transforms.RandomCrop(32, padding=4))
    
    if data_config.get('augmentation', {}).get('random_horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.append(transforms.ToTensor())
    
    if data_config.get('augmentation', {}).get('normalize', True):
        # CIFAR-10 normalization
        transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        )
    
    transform_train = transforms.Compose(transform_list)
    
    # Validation transform (no augmentation)
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    dataset_name = data_config.get('dataset', 'CIFAR10')
    data_dir = data_config.get('data_dir', './data')
    
    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        val_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_val
        )
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        val_dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_val
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 32),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation'].get('test_batch_size', 64),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, num_classes


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Multi-Weight Neural Network')
    parser.add_argument('model', type=str, 
                       choices=['multi_channel', 'continuous_integration', 
                               'cross_modal', 'single_output', 'attention_based'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (uses model default if not specified)')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                       choices=['CIFAR10', 'CIFAR100'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--multi-stage', action='store_true',
                       help='Use multi-stage training')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Load model-specific default config
        config = get_model_config(args.model)
    
    # Override config with command line arguments
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup
    set_seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    log_dir = Path(args.log_dir) / args.model
    logger = setup_logging(log_dir, args.model)
    logger.info(f"Training {args.model} model on {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, num_classes = create_data_loaders(config)
    
    # Update model config with correct number of classes
    config['model']['params']['num_classes'] = num_classes
    
    # Create model
    logger.info("Creating model...")
    model = create_model(args.model, config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    checkpoint_dir = Path(args.checkpoint_dir) / args.model
    
    trainer_class = MultiStageTrainer if args.multi_stage else Trainer
    trainer = trainer_class(
        model=model,
        device=device,
        optimizer_name=config['training'].get('optimizer', 'adamw'),
        learning_rate=config['training'].get('learning_rate', 0.001),
        weight_decay=config['training'].get('weight_decay', 0.0001),
        scheduler_name=config['training'].get('scheduler', 'cosine'),
        scheduler_params=config['training'].get('scheduler_params', {'T_max': 100}),
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        mixed_precision=config['training'].get('mixed_precision', False),
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training
    logger.info("Starting training...")
    
    if args.multi_stage and 'multi_stage' in config:
        # Multi-stage training
        stages = config['multi_stage']['stages']
        
        for stage in stages:
            logger.info(f"Starting stage: {stage['name']}")
            
            freeze_config = {
                'freeze_color': stage.get('freeze_color', False),
                'freeze_brightness': stage.get('freeze_brightness', False)
            }
            
            # Update learning rate for stage
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = stage.get('learning_rate', 
                                             config['training']['learning_rate'])
            
            trainer.train_stage(
                stage_name=stage['name'],
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=stage['epochs'],
                criterion=criterion,
                freeze_config=freeze_config
            )
    else:
        # Regular training
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training'].get('num_epochs', 100),
            criterion=criterion,
            save_best=True,
            early_stopping_patience=config['training'].get('early_stopping_patience', 20)
        )
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
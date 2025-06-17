#!/usr/bin/env python3
"""
ImageNet Training Script for Multi-Weight Neural Networks
Supports all MWNN architectures with RGB+Luminance preprocessing
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel
from src.models.multi_channel.model import MultiChannelModel
from src.models.cross_modal.model import CrossModalModel
from src.models.single_output.model import SingleOutputModel
from src.models.attention_based.model import AttentionBasedModel
from src.training.mwnn_trainer import MWNNTrainer
from src.preprocessing.imagenet_dataset import create_imagenet_separate_pathway_dataloaders
from src.preprocessing.imagenet_config import get_preset_config


def setup_logging(log_dir: Path, model_name: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"imagenet_{model_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_model_class(model_name: str):
    """Get model class by name."""
    models = {
        'continuous_integration': ContinuousIntegrationModel,
        'multi_channel': MultiChannelModel,
        'cross_modal': CrossModalModel,
        'single_output': SingleOutputModel,
        'attention_based': AttentionBasedModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]


def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create model instance with ImageNet configuration and GPU optimizations."""
    model_class = get_model_class(model_name)
    
    # ImageNet-specific parameters
    imagenet_params = {
        'num_classes': 1000,  # ImageNet-1K classes
    }
    
    # Model-specific defaults with GPU optimizations
    if model_name == 'continuous_integration':
        imagenet_params.update({
            'base_channels': 32,  # Memory optimized for MPS
            'depth': 'shallow',   # Use shallow for better memory efficiency
            'dropout_rate': 0.2,
            'integration_points': ['early', 'late'],  # Fewer integration points
            'enable_mixed_precision': True,
            'memory_efficient': True
        })
    elif model_name == 'multi_channel':
        imagenet_params.update({
            'hidden_channels': [64, 128, 256, 512],
            'dropout_rate': 0.2
        })
    
    # Override with user parameters
    imagenet_params.update(kwargs)
    
    return model_class(**imagenet_params)


def train_imagenet_model(
    model_name: str = 'continuous_integration',
    data_dir: str = 'data/ImageNet-1K',
    devkit_dir: str = 'data/ImageNet-1K/ILSVRC2013_devkit',
    config_preset: str = 'training',
    epochs: int = 100,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    load_subset: Optional[int] = None,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    enable_gpu_optimizations: bool = True,
    use_mixed_precision: bool = True,
    **model_kwargs
) -> Dict:
    """
    Train a Multi-Weight Neural Network on ImageNet.
    
    Args:
        model_name: Name of the model architecture
        data_dir: Path to ImageNet data directory
        devkit_dir: Path to ImageNet devkit directory
        config_preset: Preprocessing configuration preset
        epochs: Number of training epochs
        batch_size: Training batch size (uses config default if None)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        load_subset: Number of samples to load (None for full dataset)
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        device: Device to use (auto-detects best available GPU if None)
        num_workers: Number of data loader workers
        enable_gpu_optimizations: Enable GPU-specific optimizations
        use_mixed_precision: Use automatic mixed precision training
        **model_kwargs: Additional model parameters
        
    Returns:
        Dictionary with training results
    """
    
    # Setup directories
    checkpoint_path = Path(checkpoint_dir)
    log_path = Path(log_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_path, model_name)
    logger.info(f"🚀 Starting ImageNet training with {model_name} model")
    logger.info(f"📁 Data: {data_dir}")
    logger.info(f"📁 Devkit: {devkit_dir}")
    
    # Setup device with automatic detection (no user intervention needed)
    if device is None:
        # Auto-detect optimal device including Mac M-series GPUs
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
            logger.info("🍎 Auto-detected Apple Silicon GPU (MPS) - using for optimal performance")
        elif torch.cuda.is_available():
            device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"🚀 Auto-detected NVIDIA GPU: {device_name} - using for optimal performance")
        else:
            device = 'cpu'
            logger.info("💻 No GPU detected - using CPU")
    else:
        # User specified device - validate it's available
        if device == 'mps' and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            logger.warning("⚠️  MPS requested but not available - falling back to auto-detection")
            device = None
            return train_imagenet_model(**locals())  # Recursive call with device=None
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA requested but not available - falling back to auto-detection")
            device = None
            return train_imagenet_model(**locals())  # Recursive call with device=None
        else:
            logger.info(f"🎯 Using user-specified device: {device}")
    
    device = torch.device(device)
    logger.info(f"🔧 Final device: {device}")
    
    # Apply GPU-specific optimizations
    if enable_gpu_optimizations:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("✅ Applied CUDA optimizations")
        elif device.type == 'mps':
                try:
                    # Use very conservative memory allocation for MPS to avoid OOM
                    torch.mps.set_per_process_memory_fraction(0.5)
                    logger.info("✅ Applied Apple Silicon MPS optimizations with very conservative memory usage")
                except AttributeError:
                    logger.warning("⚠️  MPS memory optimization not available in this PyTorch version")
    
    # Check mixed precision support
    mixed_precision_supported = False
    if use_mixed_precision:
        if device.type == 'cuda':
            try:
                major, minor = torch.cuda.get_device_capability(device)
                mixed_precision_supported = major >= 7 or (major == 6 and minor >= 1)
            except Exception:
                mixed_precision_supported = False
        elif device.type == 'mps':
            mixed_precision_supported = hasattr(torch.amp, 'autocast')
        
        if mixed_precision_supported:
            logger.info("✅ Mixed precision training enabled")
        else:
            logger.info("⚠️  Mixed precision training not supported on this device")
            use_mixed_precision = False
    
    # Get preprocessing config
    config = get_preset_config(config_preset, data_dir, devkit_dir)
    if batch_size is not None:
        config.batch_size = batch_size
    if num_workers is not None:
        config.num_workers = num_workers
    if load_subset is not None:
        config.load_subset = load_subset
    
    logger.info(f"⚙️ Config: {config_preset}")
    logger.info(f"📊 Batch size: {config.batch_size}")
    logger.info(f"🔄 Workers: {config.num_workers}")
    logger.info(f"📈 Feature method: {config.feature_method}")
    
    # Create data loaders
    logger.info("📚 Creating data loaders...")
    try:
        train_loader, val_loader = create_imagenet_separate_pathway_dataloaders(
            data_dir=data_dir,
            devkit_dir=devkit_dir,
            batch_size=config.batch_size,
            input_size=config.input_size,
            num_workers=config.num_workers,
            val_split=config.val_split,
            load_subset=config.load_subset
        )
        
        logger.info(f"✅ Train batches: {len(train_loader)}")
        logger.info(f"✅ Val batches: {len(val_loader)}")
        
    except Exception as e:
        logger.error(f"❌ Error creating data loaders: {e}")
        raise
    
    # Create model with GPU optimizations
    logger.info(f"🏗️ Creating {model_name} model...")
    model = create_model(model_name, **model_kwargs)
    
    # Move to device and apply optimizations
    if hasattr(model, 'to_device'):
        # For continuous_integration model with built-in optimizations
        model = model.to_device(device)
    else:
        model = model.to(device)
    
    # Apply additional GPU optimizations if available
    # Note: torch.compile disabled due to MPS stability issues
    # if enable_gpu_optimizations and hasattr(torch, 'compile') and device.type in ['cuda', 'mps']:
    #     try:
    #         model = torch.compile(model, mode='max-autotune')
    #         logger.info("✅ Model compiled for better performance")
    #     except Exception as e:
    #         logger.warning(f"⚠️  Model compilation failed: {e}")
    
    logger.info("ℹ️  Using uncompiled model for maximum stability")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Enable memory-efficient features if available (disabled due to gradient issues)
    # if hasattr(model, 'enable_gradient_checkpointing'):
    #     model.enable_gradient_checkpointing(True)
    #     logger.info("✅ Gradient checkpointing enabled for memory efficiency")
    
    logger.info("ℹ️  Gradient checkpointing disabled for stability")
    
    # Setup trainer with GPU optimizations
    logger.info("🎯 Setting up trainer...")
    
    # Create optimizer with GPU-optimized settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=device.type == 'cuda'  # Use fused AdamW for CUDA
    )
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )
    
    # Setup mixed precision scaler if supported
    scaler = None
    if mixed_precision_supported and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("✅ Mixed precision scaler initialized")
    
    trainer = MWNNTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )
    
    # Start training
    logger.info(f"🚀 Starting training for {epochs} epochs...")
    start_time = time.time()
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f}s")
        
        # Save final results
        results = {
            'model_name': model_name,
            'epochs': epochs,
            'training_time': training_time,
            'best_val_acc': max(history['val_accuracy']),
            'final_train_acc': history['train_accuracy'][-1],
            'final_val_acc': history['val_accuracy'][-1],
            'history': history
        }
        
        results_file = checkpoint_path / f"imagenet_{model_name}_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Results saved to: {results_file}")
        logger.info(f"🎯 Best validation accuracy: {results['best_val_acc']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def main():
    """Main training function with command line arguments."""
    parser = argparse.ArgumentParser(description='Train MWNN on ImageNet')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='continuous_integration',
                        choices=['continuous_integration', 'multi_channel', 'cross_modal', 
                                'single_output', 'attention_based'],
                        help='Model architecture to train')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/ImageNet-1K',
                        help='Path to ImageNet data directory')
    parser.add_argument('--devkit-dir', type=str, default='data/ImageNet-1K/ILSVRC2013_devkit',
                        help='Path to ImageNet devkit directory')
    parser.add_argument('--config-preset', type=str, default='training',
                        choices=['development', 'training', 'evaluation', 'research'],
                        help='Preprocessing configuration preset')
    parser.add_argument('--load-subset', type=int, default=None,
                        help='Number of samples to load (for testing)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization weight')
    
    # System arguments
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu', None],
                        help='Device to use (auto-detects best GPU if not specified)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loader workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs')
    
    # GPU optimization arguments
    parser.add_argument('--disable-gpu-optimizations', action='store_true',
                        help='Disable GPU-specific optimizations')
    parser.add_argument('--disable-mixed-precision', action='store_true',
                        help='Disable automatic mixed precision training')
    parser.add_argument('--enable-gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing for memory efficiency')
    
    args = parser.parse_args()
    
    # Run training with GPU optimizations
    results = train_imagenet_model(
        model_name=args.model,
        data_dir=args.data_dir,
        devkit_dir=args.devkit_dir,
        config_preset=args.config_preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_subset=args.load_subset,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
        num_workers=args.num_workers,
        enable_gpu_optimizations=not args.disable_gpu_optimizations,
        use_mixed_precision=not args.disable_mixed_precision
    )
    
    print("\n🎉 Training completed!")
    print(f"🎯 Best validation accuracy: {results['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()

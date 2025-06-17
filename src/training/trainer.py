"""Training utilities for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time
try:
    from ..models.base import BaseMultiWeightModel
except ImportError:
    from models.base import BaseMultiWeightModel


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for Multi-Weight Neural Networks."""
    
    def __init__(self,
                 model: BaseMultiWeightModel,
                 device: torch.device,
                 optimizer_name: str = 'adam',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 scheduler_name: Optional[str] = 'cosine',
                 scheduler_params: Optional[Dict] = None,
                 gradient_clip: Optional[float] = None,
                 mixed_precision: bool = False,
                 log_dir: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None):
        
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)
        
        # Setup scheduler
        self.scheduler = self._create_scheduler(scheduler_name, scheduler_params)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def _create_optimizer(self, optimizer_name: str, lr: float, 
                         weight_decay: float) -> optim.Optimizer:
        """Create optimizer based on name."""
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        optimizer_class = optimizers.get(optimizer_name.lower(), optim.Adam)
        
        # Special handling for SGD
        if optimizer_name.lower() == 'sgd':
            return optimizer_class(self.model.parameters(), lr=lr,
                                 weight_decay=weight_decay, momentum=0.9)
        else:
            return optimizer_class(self.model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    
    def _create_scheduler(self, scheduler_name: Optional[str],
                         params: Optional[Dict]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if scheduler_name is None:
            return None
        
        params = params or {}
        
        # Set default parameters for schedulers that require them
        if scheduler_name.lower() == 'cosine' and 'T_max' not in params:
            params['T_max'] = 100  # Default to 100 epochs
        elif scheduler_name.lower() == 'step' and 'step_size' not in params:
            params['step_size'] = 30  # Default step size
        elif scheduler_name.lower() == 'multistep' and 'milestones' not in params:
            params['milestones'] = [50, 75]  # Default milestones
        elif scheduler_name.lower() == 'exponential' and 'gamma' not in params:
            params['gamma'] = 0.9  # Default decay rate
        
        schedulers = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'step': optim.lr_scheduler.StepLR,
            'multistep': optim.lr_scheduler.MultiStepLR,
            'exponential': optim.lr_scheduler.ExponentialLR,
            'plateau': optim.lr_scheduler.ReduceLROnPlateau
        }
        
        scheduler_class = schedulers.get(scheduler_name.lower())
        if scheduler_class is None:
            logger.warning(f"Unknown scheduler: {scheduler_name}")
            return None
        
        return scheduler_class(self.optimizer, **params)
    
    def train_epoch(self, train_loader: DataLoader, 
                   criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                  self.gradient_clip)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('train/acc_step', 100. * correct / total, 
                                     self.global_step)
            
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader, 
                criterion: nn.Module) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        # Calculate validation metrics
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              criterion: nn.Module,
              save_best: bool = True,
              early_stopping_patience: Optional[int] = None) -> Dict[str, List[float]]:
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, criterion)
            
            # Validate
            val_metrics = self.validate(val_loader, criterion)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, current_lr)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['accuracy']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if save_best and is_best:
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            elif epoch % 10 == 0:  # Save periodic checkpoint
                self._save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['learning_rates'].append(current_lr)
        
        # Save final model
        self._save_checkpoint(self.current_epoch, val_metrics, is_best=False, 
                            final=True)
        
        # Save training history
        self._save_training_history()
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_metric:.2f}%")
        
        return self.training_history
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], lr: float):
        """Log metrics for the epoch."""
        # Console logging
        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%, "
            f"LR: {lr:.6f}"
        )
        
        # Tensorboard logging
        self.writer.add_scalars('loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, self.current_epoch)
        
        self.writer.add_scalars('accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, self.current_epoch)
        
        self.writer.add_scalar('learning_rate', lr, self.current_epoch)
        
        # Log model-specific metrics
        if hasattr(self.model, 'get_integration_weights'):
            weights = self.model.get_integration_weights()
            for name, weight_dict in weights.items():
                self.writer.add_scalars(f'integration_weights/{name}', 
                                      weight_dict, self.current_epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                        is_best: bool = False, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'model_name'):
            checkpoint['model_config'] = {
                'model_name': self.model.model_name,
                'num_parameters': getattr(self.model, 'get_num_parameters', 
                                         lambda: sum(p.numel() for p in self.model.parameters()))()
            }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        elif final:
            path = self.checkpoint_dir / 'final_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False, final: bool = False) -> Path:
        """Public API for saving model checkpoint."""
        self._save_checkpoint(epoch, metrics, is_best, final)
        
        # Return the path where the checkpoint was saved
        if is_best:
            return self.checkpoint_dir / 'best_model.pth'
        elif final:
            return self.checkpoint_dir / 'final_model.pth'
        else:
            return self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        logger.info(f"Saved training history to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, float]:
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"Best metric so far: {self.best_metric:.2f}")
        
        return checkpoint.get('metrics', {})


class MultiStageTrainer(Trainer):
    """Trainer with multi-stage training capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_history = []
    
    def train_stage(self,
                   stage_name: str,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int,
                   criterion: nn.Module,
                   freeze_config: Optional[Dict[str, bool]] = None) -> Dict[str, List[float]]:
        """Train a specific stage with optional weight freezing."""
        logger.info(f"Starting training stage: {stage_name}")
        
        # Apply freeze configuration
        if freeze_config:
            if freeze_config.get('freeze_color', False):
                self.model.freeze_color_weights()
            if freeze_config.get('freeze_brightness', False):
                self.model.freeze_brightness_weights()
        
        # Train the stage
        stage_history = self.train(train_loader, val_loader, num_epochs, 
                                  criterion, save_best=True)
        
        # Unfreeze all weights for next stage
        self.model.unfreeze_all_weights()
        
        # Save stage history
        self.stage_history.append({
            'stage_name': stage_name,
            'history': stage_history,
            'freeze_config': freeze_config
        })
        
        return stage_history
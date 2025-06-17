"""Custom trainer for Multi-Weight Neural Networks with separate pathway inputs."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple
import logging


class MWNNTrainer:
    """Trainer for MWNN models that take separate RGB and brightness inputs."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 scheduler=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc='Training')
        
        for batch_idx, ((rgb_data, brightness_data), targets) in enumerate(progress_bar):
            # Move data to device
            rgb_data = rgb_data.to(self.device)
            brightness_data = brightness_data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(rgb_data, brightness_data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validation')
            
            for batch_idx, ((rgb_data, brightness_data), targets) in enumerate(progress_bar):
                # Move data to device
                rgb_data = rgb_data.to(self.device)
                brightness_data = brightness_data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(rgb_data, brightness_data)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                accuracy = 100. * correct / total
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.2f}%'
                })
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int) -> Dict:
        """Train the model for multiple epochs."""
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Train Acc: {train_metrics['accuracy']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Track best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        
        return history

"""
MWNN Training Module
Clean trainer class for MWNN models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
from tqdm.auto import tqdm


class MWNNTrainer:
    """
    Clean trainer class for MWNN models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.002,
        checkpoint_dir: Path = Path('checkpoints')
    ):
        """
        Initialize trainer.
        
        Args:
            model: MWNN model to train
            device: Device to train on
            learning_rate: Learning rate for optimization
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.7)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc='🚀 Training',
            leave=False,
            unit='batch',
            miniters=1
        )
        
        for batch_idx, (data, target) in progress_bar:
            # Unpack dual inputs
            rgb_data, brightness_data = data
            rgb_data = rgb_data.to(self.device)
            brightness_data = brightness_data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(rgb_data, brightness_data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
                current_acc = 100. * correct / total
                current_loss = running_loss / (batch_idx + 1)
                lr = self.optimizer.param_groups[0]["lr"]
                
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.1f}%',
                    'LR': f'{lr:.2e}'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        progress_bar.close()
        print(f'✅ Epoch {epoch} Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            val_loader,
            desc='🔍 Validation',
            leave=False,
            unit='batch',
            miniters=1
        )
        
        with torch.no_grad():
            for data, target in progress_bar:
                # Unpack dual inputs
                rgb_data, brightness_data = data
                rgb_data = rgb_data.to(self.device)
                brightness_data = brightness_data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(rgb_data, brightness_data)
                val_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Update progress bar
                current_acc = 100. * correct / total
                batches_processed = total // target.size(0)
                current_loss = val_loss / batches_processed if batches_processed > 0 else 0
                
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        progress_bar.close()
        print(f'✅ Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        return val_loss, val_acc
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        save_checkpoints: bool = True
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_checkpoints: Whether to save checkpoints
            
        Returns:
            Training history dictionary
        """
        print(f"🎯 Starting training for {epochs} epochs...")
        start_time = time.time()
        
        # Overall progress bar
        epoch_progress = tqdm(
            range(1, epochs + 1),
            desc='🏋️ Training Progress',
            leave=True,
            unit='epoch'
        )
        
        for epoch in epoch_progress:
            epoch_progress.set_description(f'🏋️ Epoch {epoch}/{epochs}')
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Update epoch progress
            best_acc = max(self.history['val_acc'])
            epoch_progress.set_postfix({
                'T_Loss': f'{train_loss:.3f}',
                'T_Acc': f'{train_acc:.1f}%',
                'V_Loss': f'{val_loss:.3f}',
                'V_Acc': f'{val_acc:.1f}%',
                'Best': f'{best_acc:.1f}%',
                'LR': f'{current_lr:.1e}'
            })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if save_checkpoints:
                    self.save_checkpoint(epoch, val_acc, val_loss, is_best=True)
        
        epoch_progress.close()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Total training time: {total_time/60:.1f} minutes")
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_mwnn_model.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Best model saved: {checkpoint_path}")
        
        # Also save latest
        latest_path = self.checkpoint_dir / 'latest_mwnn_model.pth'
        torch.save(checkpoint, latest_path)
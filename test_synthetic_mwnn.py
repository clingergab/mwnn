#!/usr/bin/env python3
"""
Test Multi-Weight Neural Network on Synthetic Data
This will help validate the architecture on a simple synthetic task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import json
import os
from datetime import datetime

# Import your model components
from src.models.continuous_integration.model import ContinuousIntegrationModel
from src.models.continuous_integration.gpu_optimizer import GPUOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing the multi-weight architecture"""
    
    def __init__(self, num_samples=10000, image_size=32, num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # Create pattern-based images
            label = i % num_classes
            
            # Generate RGB image with patterns based on class
            rgb_image = self._generate_pattern_image(label)
            
            self.data.append(rgb_image)
            self.labels.append(label)
    
    def _generate_pattern_image(self, label):
        """Generate a pattern-based RGB image for the given label"""
        img = torch.zeros(3, self.image_size, self.image_size)
        
        # Create different patterns for different classes
        if label == 0:  # Horizontal stripes
            for i in range(0, self.image_size, 4):
                img[:, i:i+2, :] = 0.8
        elif label == 1:  # Vertical stripes
            for i in range(0, self.image_size, 4):
                img[:, :, i:i+2] = 0.8
        elif label == 2:  # Diagonal pattern
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i + j) % 4 < 2:
                        img[:, i, j] = 0.8
        elif label == 3:  # Checkerboard
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i // 4 + j // 4) % 2 == 0:
                        img[:, i, j] = 0.8
        elif label == 4:  # Circle pattern
            center = self.image_size // 2
            radius = self.image_size // 4
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i - center) ** 2 + (j - center) ** 2 < radius ** 2:
                        img[:, i, j] = 0.8
        elif label == 5:  # Cross pattern
            center = self.image_size // 2
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if abs(i - center) < 3 or abs(j - center) < 3:
                        img[:, i, j] = 0.8
        elif label == 6:  # Corner triangles
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if i + j < self.image_size // 2 or i + j > 3 * self.image_size // 2:
                        img[:, i, j] = 0.8
        elif label == 7:  # Random noise pattern (fixed seed for consistency)
            torch.manual_seed(label * 1000)
            img = torch.rand(3, self.image_size, self.image_size) > 0.5
            img = img.float() * 0.8
        elif label == 8:  # Gradient pattern
            for i in range(self.image_size):
                for j in range(self.image_size):
                    img[:, i, j] = (i + j) / (2 * self.image_size)
        else:  # label == 9: Border pattern
            img[:, :3, :] = 0.8  # Top border
            img[:, -3:, :] = 0.8  # Bottom border
            img[:, :, :3] = 0.8  # Left border
            img[:, :, -3:] = 0.8  # Right border
        
        # Add some noise for realism
        noise = torch.randn_like(img) * 0.1
        img = torch.clamp(img + noise, 0, 1)
        
        # Normalize to ImageNet stats for consistency
        img = (img - 0.5) / 0.5  # Scale to [-1, 1]
        
        return img
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_brightness_channel(rgb_tensor):
    """Create brightness channel from RGB tensor using luminance formula"""
    # Standard luminance weights: 0.299*R + 0.587*G + 0.114*B
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device).view(1, 3, 1, 1)
    brightness = torch.sum(rgb_tensor * weights, dim=1, keepdim=True)
    return brightness


def create_model(device):
    """Create and optimize the continuous integration model for synthetic data"""
    logger.info("🏗️ Creating synthetic data-adapted continuous_integration model...")
    
    # Create the model with appropriate parameters
    model = ContinuousIntegrationModel(
        num_classes=10,              # 10 pattern classes
        base_channels=32,            # Smaller base channels for synthetic data
        depth='shallow',             # Use shallow architecture
        dropout_rate=0.1,
        integration_points=['early', 'middle'],  # Fewer integration points
        enable_mixed_precision=True,
        memory_efficient=True
    )
    
    # Configure backends for optimal performance
    GPUOptimizer.configure_backends(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Total parameters: {total_params:,}")
    logger.info(f"📊 Trainable parameters: {trainable_params:,}")
    
    return model, total_params


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Create brightness channel
        brightness_data = create_brightness_channel(data)
        
        optimizer.zero_grad()
        
        # Forward pass with both RGB and brightness
        output = model(data, brightness_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating', leave=False):
            data, target = data.to(device), target.to(device)
            
            # Create brightness channel
            brightness_data = create_brightness_channel(data)
            
            # Forward pass
            output = model(data, brightness_data)
            loss = criterion(output, target)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    # Setup
    logger.info("🚀 Starting Synthetic Data training with Multi-Weight Neural Network")
    
    # Device selection using GPUOptimizer
    device = GPUOptimizer.detect_optimal_device()
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    
    logger.info(f"📊 Batch size: {batch_size}")
    logger.info(f"🔄 Epochs: {num_epochs}")
    logger.info(f"📈 Learning rate: {learning_rate}")
    
    # Create synthetic datasets
    logger.info("📚 Creating synthetic datasets...")
    train_dataset = SyntheticDataset(num_samples=8000, image_size=32, num_classes=10)
    test_dataset = SyntheticDataset(num_samples=2000, image_size=32, num_classes=10)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    logger.info(f"✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Test samples: {len(test_dataset)}")
    
    # Create model
    model, total_params = create_model(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_acc = 0.0
    start_time = datetime.now()
    
    # Training loop
    logger.info("🚀 Starting training...")
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_synthetic_mwnn.pth')
            logger.info(f"💾 New best model saved! Accuracy: {best_val_acc:.2f}%")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Final results
    logger.info("\n🎉 Training completed!")
    logger.info(f"⏱️  Total training time: {training_time:.2f} seconds")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"📊 Model parameters: {total_params:,}")
    
    # Save results
    results = {
        'model_name': 'continuous_integration_synthetic',
        'dataset': 'synthetic_patterns',
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'total_parameters': total_params,
        'training_time': training_time,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_accuracy'][-1],
        'final_val_acc': history['val_accuracy'][-1],
        'history': history
    }
    
    # Save to file
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/synthetic_mwnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("📁 Results saved to checkpoints/synthetic_mwnn_results.json")
    
    return results


if __name__ == '__main__':
    main()

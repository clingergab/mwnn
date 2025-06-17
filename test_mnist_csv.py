#!/usr/bin/env python3
"""
Test Multi-Weight Neural Network on MNIST CSV Dataset
This will help validate the architecture on a simple dataset using CSV format
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
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


class MNISTCSVDataset(Dataset):
    """MNIST dataset loader for CSV format"""
    
    def __init__(self, csv_file, transform=None, max_samples=None):
        """
        Args:
            csv_file (str): Path to the CSV file
            transform: Optional transform to be applied on samples
            max_samples (int): Maximum number of samples to load (for testing)
        """
        logger.info(f"Loading MNIST data from {csv_file}...")
        
        # Load data with pandas in chunks to manage memory
        self.data = []
        self.labels = []
        
        chunk_size = 10000
        loaded_samples = 0
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            if max_samples and loaded_samples >= max_samples:
                break
                
            # First column is label, rest are pixel values
            labels = chunk.iloc[:, 0].values
            pixels = chunk.iloc[:, 1:].values
            
            for i in range(len(labels)):
                if max_samples and loaded_samples >= max_samples:
                    break
                    
                # Reshape pixels to 28x28 and normalize
                pixel_array = pixels[i].reshape(28, 28).astype(np.float32) / 255.0
                
                # Convert to RGB (3 channels) by replicating grayscale
                rgb_image = np.stack([pixel_array, pixel_array, pixel_array], axis=0)
                
                self.data.append(torch.from_numpy(rgb_image))
                self.labels.append(int(labels[i]))
                loaded_samples += 1
        
        self.transform = transform
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_mnist_transforms():
    """Create transforms for MNIST"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model expectations
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return transform


def create_brightness_channel(rgb_tensor):
    """Create brightness channel from RGB tensor using luminance formula"""
    # Standard luminance weights: 0.299*R + 0.587*G + 0.114*B
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device).view(1, 3, 1, 1)
    brightness = torch.sum(rgb_tensor * weights, dim=1, keepdim=True)
    return brightness


def create_model(device):
    """Create and optimize the continuous integration model for MNIST"""
    logger.info("🏗️ Creating MNIST-adapted continuous_integration model...")
    
    # Create the model with appropriate parameters
    model = ContinuousIntegrationModel(
        num_classes=10,              # MNIST has 10 classes
        base_channels=32,            # Smaller base channels for MNIST
        depth='shallow',             # Use shallow architecture for faster training
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
    logger.info("🚀 Starting MNIST CSV training with Multi-Weight Neural Network")
    
    # Device selection using GPUOptimizer
    device = GPUOptimizer.detect_optimal_device()
    
    # Hyperparameters
    batch_size = 64  # Smaller batch size for MPS
    num_epochs = 10
    learning_rate = 0.001
    max_train_samples = 10000  # Limit samples for faster testing
    max_test_samples = 2000
    
    logger.info(f"📊 Batch size: {batch_size}")
    logger.info(f"🔄 Epochs: {num_epochs}")
    logger.info(f"📈 Learning rate: {learning_rate}")
    logger.info(f"🎯 Max train samples: {max_train_samples}")
    logger.info(f"🎯 Max test samples: {max_test_samples}")
    
    # Create data loaders
    logger.info("📚 Loading MNIST CSV dataset...")
    transform = create_mnist_transforms()
    
    train_dataset = MNISTCSVDataset(
        'data/MNIST/mnist_train.csv', 
        transform=transform,
        max_samples=max_train_samples
    )
    test_dataset = MNISTCSVDataset(
        'data/MNIST/mnist_test.csv', 
        transform=transform,
        max_samples=max_test_samples
    )
    
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
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
            torch.save(model.state_dict(), 'checkpoints/best_mnist_mwnn.pth')
            logger.info(f"💾 New best model saved! Accuracy: {best_val_acc:.2f}%")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Final results
    logger.info("\n🎉 Training completed!")
    logger.info(f"⏱️  Total training time: {training_time:.2f} seconds")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"📊 Model parameters: {total_params:,}")
    
    # Compare with typical MNIST performance
    logger.info("\n📈 Performance Analysis:")
    if best_val_acc >= 95.0:
        logger.info("🎯 Excellent! Your model performs very well on MNIST")
    elif best_val_acc >= 90.0:
        logger.info("✅ Good! Your model shows solid performance on MNIST")
    elif best_val_acc >= 80.0:
        logger.info("⚠️  Moderate performance - may need architecture improvements")
    else:
        logger.info("❌ Low performance - architecture or training issues need investigation")
    
    # Save results
    results = {
        'model_name': 'continuous_integration_mnist_csv',
        'dataset': 'MNIST_CSV',
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'total_parameters': total_params,
        'max_train_samples': max_train_samples,
        'max_test_samples': max_test_samples,
        'training_time': training_time,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_accuracy'][-1],
        'final_val_acc': history['val_accuracy'][-1],
        'history': history
    }
    
    # Save to file
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/mnist_csv_mwnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("📁 Results saved to checkpoints/mnist_csv_mwnn_results.json")
    
    return results


if __name__ == '__main__':
    main()

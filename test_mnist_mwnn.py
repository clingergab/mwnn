#!/usr/bin/env python3
"""
Test Multi-Weight Neural Network on MNIST Dataset
This will help validate the architecture on a simpler task before ImageNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
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

def create_mnist_transforms():
    """Create transforms for MNIST - convert grayscale to RGB for consistency"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model expectations
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel RGB
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # MNIST stats replicated for 3 channels
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
    
    # Use a smaller, more appropriate config for MNIST
    model_config = {
        'num_classes': 10,  # MNIST has 10 classes
        'stages': 2,        # Keep it simple
        'channels': [16, 32],  # Smaller channels for MNIST
        'dropout_rate': 0.1,
        'use_batch_norm': True,
        'activation': 'relu'
    }
    
    model = ContinuousIntegrationModel(
        num_classes=model_config['num_classes'],
        stages=model_config['stages'],
        channels=model_config['channels'],
        dropout_rate=model_config['dropout_rate'],
        use_batch_norm=model_config['use_batch_norm'],
        activation=model_config['activation']
    )
    
    # Apply basic device transfer and optimizations
    model = model.to(device)
    
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
    logger.info("🚀 Starting MNIST training with Multi-Weight Neural Network")
    
    # Device selection using GPUOptimizer
    device = GPUOptimizer.detect_optimal_device()
    
    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001
    
    logger.info(f"📊 Batch size: {batch_size}")
    logger.info(f"🔄 Epochs: {num_epochs}")
    logger.info(f"📈 Learning rate: {learning_rate}")
    
    # Create data loaders
    logger.info("📚 Loading MNIST dataset...")
    transform = create_mnist_transforms()
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
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
            torch.save(model.state_dict(), 'checkpoints/best_mnist_mwnn.pth')
            logger.info(f"💾 New best model saved! Accuracy: {best_val_acc:.2f}%")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Final results
    logger.info("\n🎉 Training completed!")
    logger.info(f"⏱️  Total training time: {training_time:.2f} seconds")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"📊 Model parameters: {total_params:,}")
    
    # Save results
    results = {
        'model_name': 'continuous_integration_mnist',
        'dataset': 'MNIST',
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
    with open('checkpoints/mnist_mwnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("📁 Results saved to checkpoints/mnist_mwnn_results.json")
    
    return results

if __name__ == '__main__':
    main()

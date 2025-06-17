#!/usr/bin/env python3
"""
Deep MWNN Training for Google Colab - ImageNet-1K Only
Optimized for T4/A100 GPU acceleration with deep model architectures
Requires ImageNet-1K dataset in Google Drive
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel
from src.preprocessing.imagenet_dataset import create_imagenet_mwnn_dataset, get_imagenet_transforms
from src.preprocessing.imagenet_config import get_preset_config
from setup_colab import get_gpu_info, get_optimal_settings


def detect_optimal_batch_size(model, device, input_shape=(3, 224, 224), max_batch=512):
    """
    Detect optimal batch size for training
    """
    print("🔍 Detecting optimal batch size...")
    
    # Start with T4/A100 optimized sizes
    gpu_name = get_gpu_info().get('name', '').upper()
    if 'A100' in gpu_name:
        test_batch = 128
    elif 'T4' in gpu_name:
        test_batch = 64
    else:
        test_batch = 32
    
    # Test the starting batch size
    try:
        model.train()
        dummy_input = torch.randn(test_batch, *input_shape).to(device)
        dummy_target = torch.randint(0, 1000, (test_batch,)).to(device)
        
        # Test forward pass
        output = model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_target)
        
        # Test backward pass
        loss.backward()
        
        # Clear memory
        del dummy_input, dummy_target, output, loss
        torch.cuda.empty_cache()
        
        print(f"✅ Optimal batch size: {test_batch}")
        return test_batch
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ Batch size {test_batch} too large, trying smaller...")
            # Try smaller batch sizes
            for smaller_batch in [test_batch // 2, test_batch // 4, 16, 8]:
                try:
                    torch.cuda.empty_cache()
                    dummy_input = torch.randn(smaller_batch, *input_shape).to(device)
                    dummy_target = torch.randint(0, 1000, (smaller_batch,)).to(device)
                    
                    output = model(dummy_input)
                    loss = nn.CrossEntropyLoss()(output, dummy_target)
                    loss.backward()
                    
                    del dummy_input, dummy_target, output, loss
                    torch.cuda.empty_cache()
                    
                    print(f"✅ Optimal batch size: {smaller_batch}")
                    return smaller_batch
                    
                except RuntimeError:
                    continue
            
            print("❌ Could not find suitable batch size")
            return 8
        else:
            raise e


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    print(f'Epoch {epoch} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_time:.1f}s')
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(data, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    val_loss /= len(dataloader)
    val_acc = 100. * correct / total
    
    print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    
    return val_loss, val_acc


def run_deep_training(
    dataset_name='ImageNet',
    data_dir='/content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K',
    devkit_dir='/content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K/ILSVRC2013_devkit',
    complexity='deep',  # Default to deep models
    epochs=30,  # More epochs for deep training
    use_auto_batch_size=True,
    learning_rate=None,
    weight_decay=0.01,
    save_checkpoints=True,
    save_interval=5
):
    """
    Run deep MWNN training on ImageNet-1K
    Optimized for Google Colab environment
    """
    
    print("🚀 Starting Deep MWNN Training on ImageNet-1K")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Devkit directory: {devkit_dir}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        print(f"🎮 GPU: {gpu_info.get('name', 'Unknown')}")
        print(f"💾 GPU Memory: {gpu_info.get('memory_gb', 'Unknown')} GB")
    
    # Get preset configuration for deep models
    config = get_preset_config('deep_imagenet')
    
    # Create model
    print(f"🏗️  Creating deep MWNN model...")
    model = ContinuousIntegrationModel(
        complexity=complexity,
        dataset_name=dataset_name,
        learning_config=config
    )
    model = model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup data transforms
    transforms_dict = get_imagenet_transforms()
    
    # Create datasets and dataloaders
    print("📚 Creating ImageNet datasets...")
    train_dataset, val_dataset = create_imagenet_mwnn_dataset(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        transforms=transforms_dict
    )
    
    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(val_dataset):,}")
    
    # Determine batch size
    if use_auto_batch_size:
        batch_size = detect_optimal_batch_size(model, device)
    else:
        # Default optimized batch sizes
        gpu_name = get_gpu_info().get('name', '').upper()
        if 'A100' in gpu_name:
            batch_size = 128
        elif 'T4' in gpu_name:
            batch_size = 64
        else:
            batch_size = 32
    
    print(f"📦 Batch size: {batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    if learning_rate is None:
        # Scale learning rate with batch size
        base_lr = 0.001
        learning_rate = base_lr * (batch_size / 64)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"🎯 Learning rate: {learning_rate}")
    print(f"⚖️  Weight decay: {weight_decay}")
    print(f"🔄 Epochs: {epochs}")
    
    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    # Create checkpoints directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\n🎯 Starting training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\n📈 Epoch {epoch}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"📚 Learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_checkpoints:
                best_model_path = checkpoint_dir / 'best_imagenet_deep_mwnn.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': config
                }, best_model_path)
                print(f"💾 New best model saved: {best_model_path}")
        
        # Save periodic checkpoints
        if save_checkpoints and epoch % save_interval == 0:
            checkpoint_path = checkpoint_dir / f'imagenet_deep_mwnn_epoch_{epoch}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {total_time/3600:.2f} hours")
    
    # Final results
    results = {
        'dataset': dataset_name,
        'complexity': complexity,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'best_val_acc': best_val_acc,
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'training_time_hours': total_time / 3600,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'gpu_info': get_gpu_info(),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n🎉 Training Complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📊 Final training accuracy: {train_accs[-1]:.2f}%")
    print(f"📊 Final validation accuracy: {val_accs[-1]:.2f}%")
    
    return results


def main():
    """Main training function for Colab"""
    try:
        print("🔧 Setting up training environment...")
        
        # Run ImageNet training with deep models
        results = run_deep_training(
            dataset_name='ImageNet',
            complexity='deep',
            epochs=30,
            use_auto_batch_size=True
        )
        
        print("\n📊 Training Results:")
        print(f"🏆 Best validation accuracy: {results['best_val_acc']:.2f}%")
        print(f"⏱️  Training time: {results['training_time_hours']:.2f} hours")
        print(f"📦 Batch size used: {results['batch_size']}")
        print(f"🎯 Learning rate: {results['learning_rate']}")
        
        # Save summary
        summary_file = Path('checkpoints') / 'imagenet_deep_training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Summary saved: {summary_file}")
        print("🎉 ImageNet Deep MWNN training complete!")
        
    except Exception as e:
        print(f"❌ ImageNet training failed: {e}")
        print("💡 Make sure ImageNet data is available in Google Drive")
        print("📁 Expected path: /content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K")


if __name__ == "__main__":
    print("🚀 Deep MWNN Training on ImageNet")
    print("Optimized for Google Colab with T4/A100 GPU support")
    print("📋 Training deep models on ImageNet-1K from Google Drive")
    print()
    
    # Quick setup instructions
    print("📋 Quick Setup:")
    print("1. Mount Google Drive")
    print("2. Navigate to /content/drive/MyDrive/mwnn/multi-weight-neural-networks")
    print("3. Ensure ImageNet-1K dataset is in data/ImageNet-1K/")
    print("4. Run this script")
    print()
    
    # Verify GPU
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected! Please enable GPU in Colab:")
        print("   Runtime > Change runtime type > GPU > T4 or A100")
        sys.exit(1)
    
    # Check ImageNet data
    imagenet_path = Path('/content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K')
    if not imagenet_path.exists():
        print("⚠️  ImageNet data not found!")
        print("📁 Expected path: /content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K")
        print("💡 Please upload ImageNet-1K dataset to your Google Drive")
        
        # Try to provide guidance
        mydrive_path = Path('/content/drive/MyDrive')
        if mydrive_path.exists():
            print("\n📁 Available directories in MyDrive:")
            try:
                for item in sorted(mydrive_path.iterdir()):
                    if item.is_dir():
                        print(f"   📂 {item.name}/")
            except:
                pass
        
        # Don't exit, let the training function handle the error gracefully
    
    main()

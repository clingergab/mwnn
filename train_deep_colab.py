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
import argparse
from pathlib import Path
from datetime import datetime
import gc
from tqdm.auto import tqdm

sys.path.append('.')

from src.models.continuous_integration import ContinuousIntegrationModel
from src.preprocessing.imagenet_dataset import create_imagenet_separate_pathway_dataloaders
from src.preprocessing.imagenet_config import get_preset_config
from setup_colab import get_gpu_info


# GPU Memory Management and Cleanup
def clear_gpu_memory():
    """Clear GPU memory and kill memory-hogging processes"""
    print("🧹 Clearing GPU memory...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Get GPU memory info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB allocated")
            print(f"GPU {i}: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB reserved")

def kill_gpu_processes():
    """Kill processes using excessive GPU memory"""
    print("🔫 Checking for GPU memory hogs...")
    
    try:
        # Use nvidia-smi to find processes
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    pid, memory = line.split(',')
                    memory_gb = float(memory.strip()) / 1024
                    if memory_gb > 30:  # Kill processes using more than 30GB
                        print(f"🔫 Killing process {pid} using {memory_gb:.1f} GB")
                        try:
                            subprocess.run(['kill', '-9', pid.strip()], check=True)
                        except Exception:
                            print(f"Failed to kill process {pid}")
    except Exception as e:
        print(f"Could not check GPU processes: {e}")

def setup_memory_optimization():
    """Set up PyTorch memory optimization"""
    import os
    
    # Set PyTorch memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Set additional optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    print("🚀 Memory optimization configured")


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
        # Create test inputs for RGB (3 channels) and brightness (1 channel)
        test_rgb = torch.randn(test_batch, 3, 224, 224).to(device)
        test_brightness = torch.randn(test_batch, 1, 224, 224).to(device)
        test_target = torch.randint(0, 1000, (test_batch,)).to(device)
        
        # Test forward pass
        output = model(test_rgb, test_brightness)
        loss = nn.CrossEntropyLoss()(output, test_target)
        
        # Test backward pass
        loss.backward()
        
        # Clear memory
        del test_rgb, test_brightness, test_target, output, loss
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
                    test_rgb = torch.randn(smaller_batch, 3, 224, 224).to(device)
                    test_brightness = torch.randn(smaller_batch, 1, 224, 224).to(device)
                    test_target = torch.randint(0, 1000, (smaller_batch,)).to(device)
                    
                    output = model(test_rgb, test_brightness)
                    loss = nn.CrossEntropyLoss()(output, test_target)
                    loss.backward()
                    
                    del test_rgb, test_brightness, test_target, output, loss
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
    """Train for one epoch with visual progress bar"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    # Create progress bar for batches
    progress_bar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader),
        desc=f'🚀 Epoch {epoch}',
        ncols=100,
        leave=True
    )
    
    for batch_idx, (data, target) in progress_bar:
        # Unpack dual inputs: data is (rgb_data, brightness_data)
        rgb_data, brightness_data = data
        rgb_data = rgb_data.to(device)
        brightness_data = brightness_data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(rgb_data, brightness_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar with current metrics
        current_acc = 100. * correct / total
        current_loss = running_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Final epoch summary
    tqdm.write(f'✅ Epoch {epoch} Complete - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_time:.1f}s')
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate the model with visual progress bar"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar for validation
    progress_bar = tqdm(
        dataloader, 
        desc='🔍 Validation',
        ncols=100,
        leave=False
    )
    
    with torch.no_grad():
        for data, target in progress_bar:
            # Unpack dual inputs: data is (rgb_data, brightness_data)
            rgb_data, brightness_data = data
            rgb_data = rgb_data.to(device)
            brightness_data = brightness_data.to(device)
            target = target.to(device)
            output = model(rgb_data, brightness_data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_acc = 100. * correct / total
            current_loss = val_loss / (len(progress_bar.iterable) - len(progress_bar.iterable) + progress_bar.n + 1)
            
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    val_loss /= len(dataloader)
    val_acc = 100. * correct / total
    
    tqdm.write(f'✅ Validation Complete - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    
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
    
    # Clear GPU memory and setup optimization FIRST
    kill_gpu_processes()
    clear_gpu_memory()
    setup_memory_optimization()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        print(f"🎮 GPU: {gpu_info.get('name', 'Unknown')}")
        print(f"💾 GPU Memory: {gpu_info.get('memory_gb', 'Unknown')} GB")
    
    # Get preset configuration for deep models
    config = get_preset_config('training', data_dir, devkit_dir)
    
    # Create model
    print("🏗️  Creating deep MWNN model...")
    
    # Map complexity to depth parameter
    depth_mapping = {
        'shallow': 'shallow',
        'medium': 'medium', 
        'deep': 'deep'
    }
    model_depth = depth_mapping.get(complexity, 'deep')
    
    # Reduce model size for A100 memory constraints
    gpu_name = get_gpu_info().get('name', '').upper() if torch.cuda.is_available() else ''
    if 'A100' in gpu_name:
        base_channels = 24  # Further reduced for A100 with memory constraints
        print("🔧 Using reduced model size for A100 memory optimization")
    else:
        base_channels = 32
    
    model = ContinuousIntegrationModel(
        num_classes=1000,  # ImageNet-1K
        depth=model_depth,
        base_channels=base_channels,
        dropout_rate=0.2,
        integration_points=['early', 'middle', 'late'],
        enable_mixed_precision=True,
        memory_efficient=True
    )
    model = model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
     # Determine batch size with memory-conscious approach
    if use_auto_batch_size:
        # For A100, use optimized batch size based on available memory
        gpu_name = get_gpu_info().get('name', '').upper() if torch.cuda.is_available() else ''
        if 'A100' in gpu_name:
            batch_size = 128  # Optimized for A100 - much better utilization
            print(f"🚀 Using optimized A100 batch size: {batch_size}")
        else:
            batch_size = detect_optimal_batch_size(model, device)
    else:
        # Default optimized batch sizes
        gpu_name = get_gpu_info().get('name', '').upper() if torch.cuda.is_available() else ''
        if 'A100' in gpu_name:
            batch_size = 128  # Optimized for A100 (was 48)
        elif 'T4' in gpu_name:
            batch_size = 64   # Increased from 32
        else:
            batch_size = 32   # Increased from 16

    # Create datasets and dataloaders
    print("📚 Creating ImageNet dataloaders...")
    train_loader, val_loader = create_imagenet_separate_pathway_dataloaders(
        data_dir=data_dir,
        devkit_dir=devkit_dir,
        batch_size=batch_size,
        input_size=config.input_size,
        num_workers=config.num_workers,
        load_subset=config.load_subset,
        val_split=0.1
    )
    
    print(f"📊 Training batches: {len(train_loader):,}")
    print(f"📊 Validation batches: {len(val_loader):,}")
    
    print(f"📦 Batch size: {batch_size}")
    
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
    
    # Create overall training progress bar
    epoch_progress = tqdm(
        range(1, epochs + 1), 
        desc='🏋️ Overall Training',
        ncols=120,
        position=0,
        leave=True
    )
    
    for epoch in epoch_progress:
        # Update epoch progress description
        epoch_progress.set_description(f'🏋️ Training Epoch {epoch}/{epochs}')
        
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
        
        # Update epoch progress bar with metrics
        epoch_progress.set_postfix({
            'Train_Acc': f'{train_acc:.2f}%',
            'Val_Acc': f'{val_acc:.2f}%',
            'Best_Val': f'{max(val_accs):.2f}%',
            'LR': f'{current_lr:.2e}'
        })
        
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
    # Add command line argument support
    parser = argparse.ArgumentParser(description='Deep MWNN Training on ImageNet')
    parser.add_argument('--data_dir', type=str, 
                       default='/content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K',
                       help='Path to ImageNet data directory')
    parser.add_argument('--devkit_dir', type=str,
                       default='/content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K/ILSVRC2013_devkit', 
                       help='Path to ImageNet devkit directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    
    args = parser.parse_args()
    
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
    
    # Check ImageNet data with fallback paths
    imagenet_paths_to_try = [
        args.data_dir,
        '/content/drive/MyDrive/mwnn/multi-weight-neural-networks/data/ImageNet-1K',
        '/content/drive/MyDrive/projects/mwnn/multi-weight-neural-networks/data/ImageNet-1K',
        'data/ImageNet-1K'
    ]
    
    found_data_path = None
    for path in imagenet_paths_to_try:
        if Path(path).exists():
            found_data_path = path
            print(f"✅ Found ImageNet data at: {path}")
            break
    
    if not found_data_path:
        print("⚠️  ImageNet data not found in any expected location!")
        print("📁 Checked paths:")
        for path in imagenet_paths_to_try:
            print(f"   ❌ {path}")
        print("💡 Please upload ImageNet-1K dataset to your Google Drive")
        
        # Try to provide guidance
        mydrive_path = Path('/content/drive/MyDrive')
        if mydrive_path.exists():
            print("\n📁 Available directories in MyDrive:")
            try:
                for item in sorted(mydrive_path.iterdir()):
                    if item.is_dir():
                        print(f"   📂 {item.name}/")
            except Exception:
                pass
        sys.exit(1)
    
    # Update devkit path based on found data path
    if found_data_path != args.data_dir:
        devkit_path = f"{found_data_path}/ILSVRC2013_devkit"
    else:
        devkit_path = args.devkit_dir
    
    # Run training with discovered paths
    try:
        results, _ = run_deep_training(
            data_dir=found_data_path,
            devkit_dir=devkit_path,
            epochs=args.epochs
        )
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Best validation accuracy: {results.get('best_val_acc', 'N/A')}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

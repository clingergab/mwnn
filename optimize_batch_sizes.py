#!/usr/bin/env python3
"""
Batch Size Optimizer for A100 GPU
Find the optimal batch size that maximizes GPU utilization without OOM
"""

import torch
import torch.nn as nn
import sys
import gc
import time

sys.path.append('/content/drive/MyDrive/mwnn/multi-weight-neural-networks')

from src.models.continuous_integration.model import ContinuousIntegrationModel
from setup_colab import get_gpu_info

def test_batch_size(model, device, batch_size, input_shape=(3, 224, 224)):
    """Test if a batch size works without OOM"""
    try:
        # Clear cache first
        torch.cuda.empty_cache()
        
        # Create test data
        rgb_data = torch.randn(batch_size, 3, 224, 224).to(device)
        brightness_data = torch.randn(batch_size, 1, 224, 224).to(device)
        target = torch.randint(0, 1000, (batch_size,)).to(device)
        
        # Forward pass
        model.train()
        output = model(rgb_data, brightness_data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        
        # Clear gradients and cache
        model.zero_grad()
        del rgb_data, brightness_data, target, output, loss
        torch.cuda.empty_cache()
        
        return True, allocated, reserved
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return False, 0, 0
        else:
            raise e

def find_optimal_batch_size():
    """Find the optimal batch size for the current GPU"""
    print("🔍 Finding Optimal Batch Size for A100")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return 32
    
    gpu_info = get_gpu_info()
    print(f"🎮 GPU: {gpu_info.get('name', 'Unknown')}")
    print(f"💾 Total Memory: {gpu_info.get('memory_gb', 'Unknown')} GB")
    
    # Create model
    model = ContinuousIntegrationModel(
        num_classes=1000,
        depth='deep',
        base_channels=24,  # Current optimized size
        dropout_rate=0.2,
        integration_points=['early', 'middle', 'late'],
        enable_mixed_precision=True,
        memory_efficient=True
    )
    model = model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test batch sizes
    batch_sizes_to_test = [32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 256]
    results = []
    
    print(f"\n🧪 Testing batch sizes...")
    
    for batch_size in batch_sizes_to_test:
        print(f"  Testing batch_size={batch_size}...", end=" ")
        
        success, allocated, reserved = test_batch_size(model, device, batch_size)
        
        if success:
            results.append({
                'batch_size': batch_size,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': allocated + reserved,
                'utilization': (allocated + reserved) / gpu_info.get('memory_gb', 40) * 100
            })
            print(f"✅ {allocated:.1f}GB allocated, {reserved:.1f}GB reserved ({allocated+reserved:.1f}GB total)")
        else:
            print(f"❌ OOM")
            break
    
    if not results:
        print("❌ No batch sizes worked!")
        return 32
    
    # Find optimal batch size (highest that uses <85% of memory)
    print(f"\n📊 Batch Size Analysis:")
    print(f"{'Batch':<6} {'Memory':<8} {'Util%':<6} {'Recommendation'}")
    print("-" * 40)
    
    optimal_batch_size = 32
    for result in results:
        recommendation = ""
        if result['utilization'] < 70:
            recommendation = "👍 Good"
            optimal_batch_size = result['batch_size']
        elif result['utilization'] < 85:
            recommendation = "🎯 Optimal"
            optimal_batch_size = result['batch_size']
        else:
            recommendation = "⚠️  High"
    
        print(f"{result['batch_size']:<6} {result['total_gb']:<8.1f} {result['utilization']:<6.1f} {recommendation}")
    
    print(f"\n🎯 Recommended batch size: {optimal_batch_size}")
    print(f"💡 This will use ~{results[-1]['total_gb']:.1f}GB of {gpu_info.get('memory_gb', 40):.1f}GB available")
    
    return optimal_batch_size

def benchmark_training_speed():
    """Benchmark training speed with different batch sizes"""
    print("\n⚡ Training Speed Benchmark")
    print("=" * 30)
    
    device = torch.device('cuda')
    model = ContinuousIntegrationModel(
        num_classes=1000, depth='deep', base_channels=24,
        dropout_rate=0.2, integration_points=['early', 'middle', 'late'],
        enable_mixed_precision=True, memory_efficient=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    batch_sizes = [48, 64, 96, 128]
    
    for batch_size in batch_sizes:
        try:
            # Test 5 iterations
            times = []
            for _ in range(5):
                torch.cuda.empty_cache()
                
                rgb_data = torch.randn(batch_size, 3, 224, 224).to(device)
                brightness_data = torch.randn(batch_size, 1, 224, 224).to(device)
                target = torch.randint(0, 1000, (batch_size,)).to(device)
                
                start_time = time.time()
                
                optimizer.zero_grad()
                output = model(rgb_data, brightness_data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                del rgb_data, brightness_data, target, output, loss
            
            avg_time = sum(times) / len(times)
            samples_per_sec = batch_size / avg_time
            
            print(f"Batch {batch_size}: {avg_time:.3f}s/batch, {samples_per_sec:.1f} samples/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size}: OOM")
                torch.cuda.empty_cache()
            else:
                raise e

if __name__ == "__main__":
    optimal_batch_size = find_optimal_batch_size()
    benchmark_training_speed()
    
    print(f"\n🚀 Final Recommendation:")
    print(f"   Use batch_size={optimal_batch_size} for optimal GPU utilization")
    print(f"   Update your training script with this value!")

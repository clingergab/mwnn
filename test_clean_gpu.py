#!/usr/bin/env python3
"""
Clean GPU optimization test without torch.compile
"""

import sys
import time
import torch

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def clean_gpu_test():
    """Test GPU optimizations without compilation."""
    print("🧹 Clean GPU Optimization Test")
    print("=" * 40)
    
    # Detect device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print("📱 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("📱 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("📱 Using CPU")
    
    # Create model without compilation
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=32,
        depth='shallow',
        enable_mixed_precision=False,  # Disable for simplicity
        memory_efficient=False         # Disable for debugging
    )
    
    # Move to device manually (avoid automatic optimization)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model created and moved to {device}")
    
    # Create test data
    batch_size = 8
    rgb_data = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.float32)
    brightness_data = torch.randn(batch_size, 1, 224, 224, device=device, dtype=torch.float32)
    
    print(f"📊 Test data: RGB {rgb_data.shape}, Brightness {brightness_data.shape}")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(rgb_data, brightness_data)
    
    # Synchronize
    if device.type == 'mps':
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time multiple runs
    times = []
    num_runs = 10
    
    print("\n⏱️  Running performance test...")
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(rgb_data, brightness_data)
            
            # Synchronize
            if device.type == 'mps':
                try:
                    torch.mps.synchronize()
                except AttributeError:
                    pass
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000  # ms
            times.append(elapsed)
            
            if i == 0:
                print(f"   First run: {elapsed:.2f} ms, Output shape: {output.shape}")
    
    # Results
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    throughput = 1000.0 / mean_time  # FPS
    
    print("\n📊 Performance Results:")
    print(f"   Mean time: {mean_time:.2f} ms")
    print(f"   Min time: {min_time:.2f} ms")
    print(f"   Max time: {max_time:.2f} ms")
    print(f"   Throughput: {throughput:.2f} FPS")
    print(f"   Batch throughput: {throughput * batch_size:.1f} samples/sec")
    
    # Memory usage
    if device.type == 'mps':
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"   Memory allocated: {allocated:.3f} GB")
        except AttributeError:
            print("   Memory tracking not available")
    elif device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"   Memory allocated: {allocated:.3f} GB")
        print(f"   Memory reserved: {reserved:.3f} GB")
    
    # Test model features
    print("\n🔗 Integration weights:")
    weights = model.get_integration_weights()
    for stage, stage_weights in weights.items():
        print(f"   {stage}: {stage_weights}")
    
    print("\n✅ Clean GPU test completed successfully!")


if __name__ == "__main__":
    clean_gpu_test()

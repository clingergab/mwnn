#!/usr/bin/env python3
"""
Profile the continuous_integration model for GPU bottlenecks and optimization opportunities.
"""

import sys
import time
import torch
import torch.profiler

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def profile_model():
    """Profile the model to identify bottlenecks."""
    print("🔍 Profiling Continuous Integration Model for bottlenecks...")
    
    # Create model
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=64,
        depth='medium',
        enable_mixed_precision=True,
        memory_efficient=True
    )
    
    # The model auto-detects device - no manual intervention needed
    print(f"✅ Model created and moved to device: {model.device}")
    
    # Test data
    batch_size = 32
    rgb_data = torch.randn(batch_size, 3, 224, 224, device=model.device)
    brightness_data = torch.randn(batch_size, 1, 224, 224, device=model.device)
    
    # Warmup
    print("🔥 Warming up...")
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(rgb_data, brightness_data)
    
    # Memory usage before profiling
    if model.device.type in ['cuda', 'mps']:
        memory_info = model.get_memory_usage()
        print(f"📊 Memory usage: {memory_info}")
    
    # Time forward pass
    print("⏱️  Timing forward passes...")
    times = []
    with torch.no_grad():
        for i in range(100):
            start = time.time()
            output = model(rgb_data, brightness_data)
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            elif model.device.type == 'mps':
                torch.mps.synchronize()
            end = time.time()
            times.append(end - start)
            
            if i % 20 == 0:
                print(f"  Iteration {i}: {times[-1]*1000:.2f}ms")
    
    avg_time = sum(times[10:]) / len(times[10:])  # Skip first 10 for warmup
    print(f"📈 Average forward pass time: {avg_time*1000:.2f}ms")
    print(f"📈 Throughput: {batch_size/avg_time:.2f} samples/second")
    
    # Profile with PyTorch profiler if CUDA available
    if model.device.type == 'cuda':
        print("🔬 Running CUDA profiler...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(10):
                    output = model(rgb_data, brightness_data)
        
        # Print top operations
        print("\n🔥 Top GPU time consumers:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        print("\n💾 Top memory consumers:")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    
    # Test gradient computation (training mode)
    print("\n🎓 Testing training mode performance...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    targets = torch.randint(0, 1000, (batch_size,), device=model.device)
    
    training_times = []
    for i in range(50):
        start = time.time()
        
        optimizer.zero_grad()
        output = model(rgb_data, brightness_data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        elif model.device.type == 'mps':
            torch.mps.synchronize()
        
        end = time.time()
        training_times.append(end - start)
        
        if i % 10 == 0:
            print(f"  Training iteration {i}: {training_times[-1]*1000:.2f}ms, loss: {loss.item():.4f}")
    
    avg_training_time = sum(training_times[10:]) / len(training_times[10:])
    print(f"📈 Average training step time: {avg_training_time*1000:.2f}ms")
    print(f"📈 Training throughput: {batch_size/avg_training_time:.2f} samples/second")
    
    # Memory efficiency test
    print("\n🧠 Testing memory efficiency...")
    max_batch_size = batch_size
    try:
        while True:
            test_batch = max_batch_size * 2
            test_rgb = torch.randn(test_batch, 3, 224, 224, device=model.device)
            test_brightness = torch.randn(test_batch, 1, 224, 224, device=model.device)
            
            with torch.no_grad():
                _ = model(test_rgb, test_brightness)
            
            max_batch_size = test_batch
            print(f"✅ Successfully processed batch size: {max_batch_size}")
            
            # Cleanup
            del test_rgb, test_brightness
            
            if max_batch_size >= 512:  # Reasonable limit
                break
                
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"💥 OOM at batch size: {max_batch_size * 2}")
            print(f"✅ Maximum stable batch size: {max_batch_size}")
        else:
            print(f"❌ Error at batch size {max_batch_size * 2}: {e}")
    
    # Final memory stats
    if model.device.type in ['cuda', 'mps']:
        final_memory = model.get_memory_usage()
        print(f"\n📊 Final memory usage: {final_memory}")
    
    # Integration weight analysis
    print("\n🔗 Integration weights analysis:")
    weights = model.get_integration_weights()
    for stage, stage_weights in weights.items():
        print(f"  {stage}: {stage_weights}")
    
    print("\n✅ Profiling complete!")


if __name__ == "__main__":
    profile_model()

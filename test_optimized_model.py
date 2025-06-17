#!/usr/bin/env python3
"""
Test the optimized continuous_integration model with smaller batch sizes to verify optimizations.
"""

import sys
import time
import torch

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def test_optimized_model():
    """Test the optimized model with conservative settings."""
    print("🔍 Testing Optimized Continuous Integration Model...")
    
    # Create model with smaller channels
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=32,  # Reduced from 64
        depth='medium',
        enable_mixed_precision=True,
        memory_efficient=True
    )
    
    print(f"✅ Model created and moved to device: {model.device}")
    
    # Test with smaller batch size
    batch_size = 8  # Much smaller than 32
    rgb_data = torch.randn(batch_size, 3, 224, 224, device=model.device)
    brightness_data = torch.randn(batch_size, 1, 224, 224, device=model.device)
    
    # Warmup
    print("🔥 Warming up...")
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(rgb_data, brightness_data)
    
    # Memory usage before timing
    if model.device.type in ['cuda', 'mps']:
        memory_info = model.get_memory_usage()
        print(f"📊 Memory usage: {memory_info}")
    
    # Time forward pass
    print("⏱️  Timing forward passes...")
    times = []
    with torch.no_grad():
        for i in range(50):
            start = time.time()
            output = model(rgb_data, brightness_data)
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            elif model.device.type == 'mps':
                torch.mps.synchronize()
            end = time.time()
            times.append(end - start)
            
            if i % 10 == 0:
                print(f"  Iteration {i}: {times[-1]*1000:.2f}ms")
    
    avg_time = sum(times[5:]) / len(times[5:])  # Skip first 5 for warmup
    print(f"📈 Average forward pass time: {avg_time*1000:.2f}ms")
    print(f"📈 Throughput: {batch_size/avg_time:.2f} samples/second")
    
    # Test training mode with small batch
    print("\n🎓 Testing training mode performance...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    targets = torch.randint(0, 1000, (batch_size,), device=model.device)
    
    training_times = []
    for i in range(20):
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
        
        if i % 5 == 0:
            print(f"  Training iteration {i}: {training_times[-1]*1000:.2f}ms, loss: {loss.item():.4f}")
    
    avg_training_time = sum(training_times[5:]) / len(training_times[5:])
    print(f"📈 Average training step time: {avg_training_time*1000:.2f}ms")
    print(f"📈 Training throughput: {batch_size/avg_training_time:.2f} samples/second")
    
    # Test memory scaling
    print("\n🧠 Testing memory scaling...")
    model.eval()
    batch_sizes = [8, 16, 24, 32]
    
    for test_batch in batch_sizes:
        try:
            test_rgb = torch.randn(test_batch, 3, 224, 224, device=model.device)
            test_brightness = torch.randn(test_batch, 1, 224, 224, device=model.device)
            
            with torch.no_grad():
                start = time.time()
                _ = model(test_rgb, test_brightness)
                if model.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif model.device.type == 'mps':
                    torch.mps.synchronize()
                end = time.time()
            
            print(f"✅ Batch size {test_batch}: {(end-start)*1000:.2f}ms")
            
            # Cleanup
            del test_rgb, test_brightness
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"💥 OOM at batch size: {test_batch}")
                break
            else:
                print(f"❌ Error at batch size {test_batch}: {e}")
                break
    
    # Final memory stats
    if model.device.type in ['cuda', 'mps']:
        final_memory = model.get_memory_usage()
        print(f"\n📊 Final memory usage: {final_memory}")
    
    # Integration weights analysis
    print("\n🔗 Integration weights analysis:")
    weights = model.get_integration_weights()
    for stage, stage_weights in weights.items():
        print(f"  {stage}: {stage_weights}")
    
    print("\n✅ Optimization test complete!")


if __name__ == "__main__":
    test_optimized_model()

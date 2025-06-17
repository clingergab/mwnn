#!/usr/bin/env python3
"""
Memory-optimized test for the continuous_integration model to verify GPU efficiency.
"""

import sys
import time
import torch

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


def test_memory_efficiency():
    """Test model with different batch sizes to find optimal memory usage."""
    print("🔍 Testing Memory-Optimized Continuous Integration Model...")
    
    # Create model with memory optimizations
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=64,  # Restored original channels for performance
        depth='medium',
        enable_mixed_precision=True,
        memory_efficient=True
    )
    
    print(f"✅ Model created and moved to device: {model.device}")
    print(f"📊 Memory-efficient mode: {model.memory_efficient}")
    print(f"📊 Gradient checkpointing: {hasattr(model, 'enable_gradient_checkpointing')}")
    
    # Test with different batch sizes to find optimal
    batch_sizes = [8, 16, 24, 32, 48, 64]  # Start with more realistic batch sizes
    successful_batch_sizes = []
    
    for batch_size in batch_sizes:
        try:
            print(f"\n🔬 Testing batch size: {batch_size}")
            
            # Create test data with requires_grad for training
            rgb_data = torch.randn(batch_size, 3, 224, 224, device=model.device, requires_grad=True)
            brightness_data = torch.randn(batch_size, 1, 224, 224, device=model.device, requires_grad=True)
            
            # Test inference
            model.eval()
            with torch.no_grad():
                start = time.time()
                output = model(rgb_data, brightness_data)
                
                if model.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif model.device.type == 'mps':
                    torch.mps.synchronize()
                
                end = time.time()
                inference_time = (end - start) * 1000
                
                print(f"  ✅ Inference successful: {inference_time:.2f}ms")
                print(f"  ✅ Output shape: {output.shape}")
                print(f"  ✅ Throughput: {batch_size/(end-start):.2f} samples/sec")
                
                successful_batch_sizes.append(batch_size)
                
                # Test training step
                model.train()
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                targets = torch.randint(0, 1000, (batch_size,), device=model.device)
                
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
                training_time = (end - start) * 1000
                
                print(f"  ✅ Training step successful: {training_time:.2f}ms")
                print(f"  ✅ Loss: {loss.item():.4f}")
                print(f"  ✅ Training throughput: {batch_size/(end-start):.2f} samples/sec")
                
                # Memory stats
                if model.device.type in ['cuda', 'mps']:
                    memory_info = model.get_memory_usage()
                    print(f"  📊 Memory usage: {memory_info}")
                
            # Clean up
            del rgb_data, brightness_data, output, loss
            if model.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif model.device.type == 'mps':
                torch.mps.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ OOM at batch size {batch_size}")
                break
            else:
                print(f"  ❌ Error: {e}")
                break
    
    print("\n📈 Results Summary:")
    print(f"✅ Successfully tested batch sizes: {successful_batch_sizes}")
    print(f"🎯 Recommended batch size: {max(successful_batch_sizes) if successful_batch_sizes else 'None'}")
    
    # Test integration weights
    print("\n🔗 Integration Analysis:")
    weights = model.get_integration_weights()
    for stage, stage_weights in weights.items():
        print(f"  {stage}: {stage_weights}")
    
    # Performance comparison
    if successful_batch_sizes:
        optimal_batch = max(successful_batch_sizes)
        print(f"\n⚡ Performance with optimal batch size ({optimal_batch}):")
        
        rgb_data = torch.randn(optimal_batch, 3, 224, 224, device=model.device)
        brightness_data = torch.randn(optimal_batch, 1, 224, 224, device=model.device)
        
        model.eval()
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                _ = model(rgb_data, brightness_data)  # Use underscore to ignore output
                if model.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif model.device.type == 'mps':
                    torch.mps.synchronize()
                end = time.time()
                times.append(end - start)
        
        avg_time = sum(times[10:]) / len(times[10:])  # Skip warmup
        print(f"📊 Average inference time: {avg_time*1000:.2f}ms")
        print(f"📊 Average throughput: {optimal_batch/avg_time:.2f} samples/sec")
        print(f"📊 Per-sample latency: {avg_time*1000/optimal_batch:.2f}ms")
    
    print("\n✅ Memory efficiency test complete!")


if __name__ == "__main__":
    test_memory_efficiency()

#!/usr/bin/env python3
"""
GPU Optimization Demo for Continuous Integration Model
Demonstrates performance improvements and Mac M-series GPU support
"""

import sys
import time
import torch
import torch.nn as nn

# Add project root to path
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel
from src.models.continuous_integration.gpu_optimizer import GPUOptimizer, ModelProfiler


def benchmark_model_performance():
    """Benchmark the optimized continuous integration model."""
    print("🚀 GPU Optimization Benchmark for Continuous Integration Model")
    print("=" * 70)
    
    # Detect optimal device
    device = GPUOptimizer.detect_optimal_device()
    print(f"📱 Detected Device: {device}")
    
    # Configure backends for optimal performance
    GPUOptimizer.configure_backends(device)
    
    # Create model with optimizations enabled
    print("\n🏗️  Creating optimized model...")
    model = ContinuousIntegrationModel(
        num_classes=1000,
        base_channels=32,  # Reduced from 64 for stability
        depth='shallow',   # Use shallow for testing
        enable_mixed_precision=True,
        memory_efficient=False  # Disable for debugging
    )
    
    # Move to device and apply optimizations
    model = model.to_device(device)
    
    # Compile for better performance if available
    model = GPUOptimizer.compile_model(model, device, mode='max_performance')
    
    print(f"✅ Model created and optimized for {device}")
    
    # Create sample data (ImageNet-like, but smaller for stability)
    batch_size = 16  # Reduced batch size
    rgb_data = torch.randn(batch_size, 3, 224, 224, device=device)
    brightness_data = torch.randn(batch_size, 1, 224, 224, device=device)
    
    # Ensure tensors are contiguous
    rgb_data = rgb_data.contiguous()
    brightness_data = brightness_data.contiguous()
    
    print(f"\n📊 Input tensors: RGB {rgb_data.shape}, Brightness {brightness_data.shape}")
    print(f"📊 RGB tensor contiguous: {rgb_data.is_contiguous()}")
    print(f"📊 Brightness tensor contiguous: {brightness_data.is_contiguous()}")
    
    # Memory info before inference
    memory_before = GPUOptimizer.get_memory_info(device)
    print("\n💾 Memory usage before inference:")
    for key, value in memory_before.items():
        if 'gb' in key:
            print(f"   {key}: {value:.2f} GB")
        else:
            print(f"   {key}: {value}")
    
    # Profile forward pass
    print("\n⏱️  Profiling forward pass performance...")
    profiler = ModelProfiler(model, device)
    profile_results = profiler.profile_forward_pass(rgb_data, brightness_data, num_iterations=20)
    
    print(f"   Mean inference time: {profile_results['mean_time_ms']:.2f} ms")
    print(f"   Min inference time: {profile_results['min_time_ms']:.2f} ms")
    print(f"   Max inference time: {profile_results['max_time_ms']:.2f} ms")
    print(f"   Throughput: {profile_results['throughput_fps']:.2f} FPS")
    
    # Memory info after inference
    memory_after = profile_results['memory_info']
    print("\n💾 Memory usage after inference:")
    for key, value in memory_after.items():
        if 'gb' in key:
            print(f"   {key}: {value:.2f} GB")
        else:
            print(f"   {key}: {value}")
    
    # Test mixed precision support
    mixed_precision_supported = GPUOptimizer.enable_mixed_precision_support(device)
    print(f"\n🎯 Mixed precision support: {'✅ Yes' if mixed_precision_supported else '❌ No'}")
    
    if mixed_precision_supported:
        print("   -> Automatic mixed precision can be used for faster training")
    
    # Test gradient checkpointing
    print("\n🔄 Testing gradient checkpointing (memory efficiency)...")
    model.enable_gradient_checkpointing(True)
    model.train()
    
    # Simulate training step with gradient checkpointing
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    target = torch.randint(0, 1000, (batch_size,), device=device)
    
    start_time = time.time()
    optimizer.zero_grad()
    
    if mixed_precision_supported and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            output = model(rgb_data, brightness_data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(rgb_data, brightness_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    training_time = (time.time() - start_time) * 1000
    print(f"   Training step time: {training_time:.2f} ms")
    print(f"   Loss: {loss.item():.4f}")
    
    # Integration weights analysis
    print("\n🔗 Integration weights analysis:")
    weights = model.get_integration_weights()
    for stage, stage_weights in weights.items():
        print(f"   {stage}:")
        for pathway, weight in stage_weights.items():
            print(f"      {pathway}: {weight:.3f}")
    
    # Clear cache and show final memory
    GPUOptimizer.clear_cache(device)
    final_memory = GPUOptimizer.get_memory_info(device)
    print("\n🧹 Memory usage after cleanup:")
    for key, value in final_memory.items():
        if 'gb' in key:
            print(f"   {key}: {value:.2f} GB")
        else:
            print(f"   {key}: {value}")
    
    print("\n✨ Benchmark completed successfully!")
    return profile_results


def compare_device_performance():
    """Compare performance across different devices if available."""
    print("\n🏁 Device Performance Comparison")
    print("=" * 40)
    
    available_devices = []
    
    # Check available devices
    if torch.cuda.is_available():
        available_devices.append(torch.device('cuda'))
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        available_devices.append(torch.device('mps'))
    
    available_devices.append(torch.device('cpu'))
    
    if len(available_devices) <= 1:
        print("Only one device available, skipping comparison.")
        return
    
    batch_size = 16  # Smaller batch for comparison
    rgb_data = torch.randn(batch_size, 3, 224, 224)
    brightness_data = torch.randn(batch_size, 1, 224, 224)
    
    results = {}
    
    for device in available_devices:
        print(f"\n🔍 Testing {device}...")
        
        # Create model for this device
        model = ContinuousIntegrationModel(
            num_classes=1000,
            base_channels=32,  # Smaller for comparison
            depth='shallow',
            enable_mixed_precision=True,
            memory_efficient=True
        )
        
        model = model.to_device(device)
        
        # Move data to device
        rgb_device = rgb_data.to(device)
        brightness_device = brightness_data.to(device)
        
        # Profile
        profiler = ModelProfiler(model, device)
        profile_results = profiler.profile_forward_pass(
            rgb_device, brightness_device, num_iterations=10
        )
        
        results[str(device)] = profile_results
        print(f"   Mean time: {profile_results['mean_time_ms']:.2f} ms")
        print(f"   Throughput: {profile_results['throughput_fps']:.2f} FPS")
    
    # Find best device
    best_device = min(results.keys(), key=lambda d: results[d]['mean_time_ms'])
    print(f"\n🏆 Best performing device: {best_device}")
    
    return results


if __name__ == "__main__":
    print("🍎 Multi-Weight Neural Networks - GPU Optimization Demo")
    print("Optimized for Mac M-series GPUs and NVIDIA GPUs")
    print()
    
    try:
        # Main benchmark
        benchmark_results = benchmark_model_performance()
        
        # Device comparison
        comparison_results = compare_device_performance()
        
        print("\n🎉 All benchmarks completed successfully!")
        print(f"Best performance: {benchmark_results['throughput_fps']:.2f} FPS")
        
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        print("Please ensure you have the required dependencies installed.")
        sys.exit(1)

#!/usr/bin/env python3
"""
Batch Size Optimization Utility for MWNN on Colab
Automatically finds the optimal batch size for different GPU configurations
"""

import torch
import torch.nn as nn
import json
import time
import sys
from pathlib import Path

sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel
from setup_colab import get_gpu_info


def benchmark_batch_size(model, batch_size, device, input_shape=(3, 224, 224), iterations=10):
    """Benchmark a specific batch size"""
    try:
        model.train()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Warmup
        for _ in range(3):
            color_input = torch.randn(batch_size, *input_shape, device=device)
            brightness_input = torch.randn(batch_size, 1, input_shape[1], input_shape[2], device=device)
            test_target = torch.randint(0, model.num_classes, (batch_size,), device=device)
            
            output = model(color_input, brightness_input)
            loss = nn.CrossEntropyLoss()(output, test_target)
            loss.backward()
            model.zero_grad()
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(iterations):
            color_input = torch.randn(batch_size, *input_shape, device=device)
            brightness_input = torch.randn(batch_size, 1, input_shape[1], input_shape[2], device=device)
            test_target = torch.randint(0, model.num_classes, (batch_size,), device=device)
            
            output = model(color_input, brightness_input)
            loss = nn.CrossEntropyLoss()(output, test_target)
            loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        throughput = batch_size / avg_time
        
        # Memory usage
        if device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
            memory_cached = torch.cuda.max_memory_reserved() / 1e9  # GB
        else:
            memory_used = 0
            memory_cached = 0
        
        return {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'throughput': throughput,
            'memory_used_gb': memory_used,
            'memory_cached_gb': memory_cached,
            'success': True
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {
                'batch_size': batch_size,
                'error': 'OOM',
                'success': False
            }
        else:
            return {
                'batch_size': batch_size,
                'error': str(e),
                'success': False
            }
    except Exception as e:
        return {
            'batch_size': batch_size,
            'error': str(e),
            'success': False
        }


def find_optimal_batch_sizes():
    """Find optimal batch sizes for different model complexities"""
    print("🔍 Finding Optimal Batch Sizes for MWNN")
    
    # Force CPU for macOS to avoid MPS issues, Colab will use CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    gpu_info = None
    if device.type == 'cuda':
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"   GPU: {gpu_info['name']}")
            print(f"   Memory: {gpu_info['memory_gb']:.1f} GB")
    
    # Model configurations to test
    model_configs = {
        'shallow_10': {'num_classes': 10, 'depth': 'shallow', 'base_channels': 32},
        'shallow_100': {'num_classes': 100, 'depth': 'shallow', 'base_channels': 32},
        'shallow_1000': {'num_classes': 1000, 'depth': 'shallow', 'base_channels': 32},
        'medium_100': {'num_classes': 100, 'depth': 'medium', 'base_channels': 64},
        'medium_1000': {'num_classes': 1000, 'depth': 'medium', 'base_channels': 64},
        'deep_100': {'num_classes': 100, 'depth': 'deep', 'base_channels': 64},
        'deep_1000': {'num_classes': 1000, 'depth': 'deep', 'base_channels': 64},
    }
    
    # Batch sizes to test
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    
    all_results = {}
    
    for config_name, config in model_configs.items():
        print(f"\n🧪 Testing {config_name}")
        print(f"   Classes: {config['num_classes']}, Depth: {config['depth']}")
        
        try:
            # Create model with forced device override
            model = ContinuousIntegrationModel(**config)
            # Override auto device detection and force specified device
            model.device = device
            model = model.to(device)
            model._device = device  # Also set private device attribute if it exists
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {total_params:,}")
            
            config_results = []
            max_successful_batch = 0
            
            for batch_size in batch_sizes:
                print(f"      Testing batch size {batch_size}...", end=" ")
                
                result = benchmark_batch_size(model, batch_size, device)
                config_results.append(result)
                
                if result['success']:
                    max_successful_batch = batch_size
                    print(f"✅ {result['throughput']:.1f} samples/sec, "
                          f"{result['memory_used_gb']:.1f}GB")
                else:
                    print(f"❌ {result['error']}")
                    if result['error'] == 'OOM':
                        break  # No point testing larger batches
            
            all_results[config_name] = {
                'model_config': config,
                'total_parameters': total_params,
                'max_successful_batch': max_successful_batch,
                'results': config_results
            }
            
            print(f"   🎯 Max successful batch size: {max_successful_batch}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
        except Exception as e:
            print(f"   ❌ Failed to test {config_name}: {e}")
            continue
    
    # Analyze results and create recommendations
    recommendations = analyze_results(all_results, gpu_info)
    
    # Save results
    output_data = {
        'gpu_info': gpu_info,
        'device': str(device),
        'batch_size_results': all_results,
        'recommendations': recommendations,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = Path('checkpoints') / 'batch_size_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return output_data


def analyze_results(all_results, gpu_info):
    """Analyze batch size results and create recommendations"""
    recommendations = {}
    
    for config_name, data in all_results.items():
        max_batch = data['max_successful_batch']
        
        # Find optimal batch size (best throughput without being too close to limit)
        successful_results = [r for r in data['results'] if r['success']]
        
        if successful_results:
            # Choose batch size that's 75% of maximum for safety
            safe_batch = int(max_batch * 0.75)
            
            # Find closest tested batch size
            optimal_batch = min(successful_results, 
                              key=lambda x: abs(x['batch_size'] - safe_batch))['batch_size']
            
            # Get throughput info
            optimal_result = next(r for r in successful_results if r['batch_size'] == optimal_batch)
            
            recommendations[config_name] = {
                'optimal_batch_size': optimal_batch,
                'max_batch_size': max_batch,
                'expected_throughput': optimal_result['throughput'],
                'memory_usage_gb': optimal_result['memory_used_gb'],
                'safety_margin': f"{((max_batch - optimal_batch) / max_batch * 100):.0f}%"
            }
    
    return recommendations


def print_recommendations(results_file='checkpoints/batch_size_optimization_results.json'):
    """Print batch size recommendations"""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        print("🎯 Batch Size Recommendations")
        print("=" * 60)
        
        gpu_info = data.get('gpu_info')
        if gpu_info:
            print(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
        print()
        
        recommendations = data['recommendations']
        
        # Group by use case
        use_cases = {
            'Development/Testing (10 classes)': ['shallow_10'],
            'Medium Scale (100 classes)': ['shallow_100', 'medium_100', 'deep_100'],
            'Production Scale (1000 classes)': ['shallow_1000', 'medium_1000', 'deep_1000']
        }
        
        for use_case, configs in use_cases.items():
            print(f"📊 {use_case}")
            for config in configs:
                if config in recommendations:
                    rec = recommendations[config]
                    print(f"   {config:15s}: Batch {rec['optimal_batch_size']:3d} "
                          f"({rec['expected_throughput']:6.1f} samples/sec, "
                          f"{rec['memory_usage_gb']:4.1f}GB)")
            print()
        
        # Quick reference
        print("🚀 Quick Reference for Colab:")
        print("   • CIFAR-10 experiments: Use batch size 128-256")
        print("   • CIFAR-100 experiments: Use batch size 64-128")  
        print("   • ImageNet-scale: Use batch size 32-64")
        print("   • Deep models: Reduce batch size by 25-50%")
        
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Run the optimization first with: python optimize_batch_sizes.py")
    except Exception as e:
        print(f"❌ Error reading results: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize batch sizes for MWNN on Colab')
    parser.add_argument('--action', choices=['optimize', 'show'], default='optimize',
                       help='Action to perform (default: optimize)')
    
    args = parser.parse_args()
    
    if args.action == 'optimize':
        find_optimal_batch_sizes()
        print("\n" + "="*60)
        print_recommendations()
    elif args.action == 'show':
        print_recommendations()


if __name__ == "__main__":
    main()

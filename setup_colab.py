#!/usr/bin/env python3
"""
Auto GPU Configuration for MWNN on Google Colab
Detects and optimizes settings based on available GPU
"""

import torch
import subprocess
import sys
from pathlib import Path


def detect_colab_environment():
    """Detect if running in Google Colab"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_gpu_info():
    """Get detailed GPU information"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        'name': torch.cuda.get_device_name(0),
        'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'compute_capability': torch.cuda.get_device_capability(0),
        'device_count': torch.cuda.device_count()
    }
    
    return gpu_info


def get_optimal_batch_size(gpu_info):
    """Get optimal batch size based on GPU memory"""
    if not gpu_info:
        return 32  # CPU fallback
    
    memory_gb = gpu_info['memory_gb']
    
    if memory_gb >= 40:  # A100
        return 128
    elif memory_gb >= 24:  # RTX 3090/4090
        return 96
    elif memory_gb >= 16:  # V100
        return 64
    elif memory_gb >= 12:  # T4
        return 48
    else:  # Smaller GPUs
        return 32


def get_optimal_settings(gpu_info):
    """Get optimal training settings based on GPU"""
    if not gpu_info:
        return {
            'batch_size': 32,
            'num_workers': 2,
            'mixed_precision': False,
            'gradient_accumulation': 1
        }
    
    memory_gb = gpu_info['memory_gb']
    
    settings = {
        'batch_size': get_optimal_batch_size(gpu_info),
        'num_workers': 4 if memory_gb >= 16 else 2,
        'mixed_precision': memory_gb >= 12,  # Enable for T4 and above
        'gradient_accumulation': 1 if memory_gb >= 16 else 2
    }
    
    return settings


def install_colab_packages():
    """Install required packages for Colab"""
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "matplotlib seaborn pandas numpy scipy",
        "tensorboard"
    ]
    
    for package in packages:
        print(f"Installing: {package}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + package.split(), 
                      check=True, capture_output=True)


def setup_colab_directories():
    """Create necessary directories"""
    dirs = ['checkpoints', 'logs', 'data/MNIST', 'data/ImageNet-1K']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def print_setup_summary(gpu_info, settings):
    """Print setup summary"""
    print("🚀 MWNN Colab Setup Complete!")
    print("=" * 50)
    
    if gpu_info:
        print(f"🖥️  GPU: {gpu_info['name']}")
        print(f"💾 Memory: {gpu_info['memory_gb']:.1f} GB")
        print(f"🔧 Compute: {gpu_info['compute_capability']}")
    else:
        print("💻 Running on CPU")
    
    print("\n⚙️  Optimal Settings:")
    print(f"   Batch Size: {settings['batch_size']}")
    print(f"   Num Workers: {settings['num_workers']}")
    print(f"   Mixed Precision: {settings['mixed_precision']}")
    print(f"   Gradient Accumulation: {settings['gradient_accumulation']}")
    
    print("\n🎯 Recommended Next Steps:")
    print("   1. Upload your MNIST data to data/MNIST/")
    print("   2. Run: python test_mnist_csv.py")
    print("   3. Run: python test_ablation_study.py")
    print("   4. Run: python debug_imagenet_pipeline.py")


def main():
    """Main setup function"""
    print("🔧 Setting up MWNN for Google Colab...")
    
    # Check if in Colab
    is_colab = detect_colab_environment()
    if is_colab:
        print("✅ Google Colab environment detected")
        install_colab_packages()
    else:
        print("ℹ️  Not in Colab - skipping package installation")
    
    # Setup directories
    setup_colab_directories()
    print("✅ Directories created")
    
    # Get GPU info and optimal settings
    gpu_info = get_gpu_info()
    settings = get_optimal_settings(gpu_info)
    
    # Save settings for other scripts
    import json
    with open('colab_settings.json', 'w') as f:
        json.dump({
            'gpu_info': gpu_info,
            'optimal_settings': settings,
            'is_colab': is_colab
        }, f, indent=2)
    
    print_setup_summary(gpu_info, settings)
    
    return gpu_info, settings


if __name__ == "__main__":
    main()

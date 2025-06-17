#!/usr/bin/env python3
"""
Colab Setup Script for ImageNet MWNN Training
Prepares Google Colab environment for ImageNet-1K training with optimal batch sizes
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_colab_environment():
    """Setup the Colab environment for ImageNet training"""
    
    print("🔧 Setting up Google Colab for ImageNet MWNN Training")
    print("="*60)
    
    # Check if we're in Colab
    try:
        import google.colab
        in_colab = True
        print("✅ Google Colab environment detected")
    except ImportError:
        in_colab = False
        print("⚠️  Not in Colab - some features may not work")
    
    # Install required packages
    print("\n📦 Installing required packages...")
    
    packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "Pillow",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "tensorboard"
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package} already installed")
        except ImportError:
            print(f"   📥 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Setup directories
    print("\n📁 Setting up directories...")
    
    directories = [
        "/content/data",
        "/content/data/ImageNet-1K", 
        "/content/checkpoints",
        "/content/logs",
        "/content/results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {directory}")
    
    # Extract MWNN package if it exists
    print("\n📦 Extracting MWNN package...")
    
    package_path = "/content/mwnn_colab_package.tar.gz"
    if os.path.exists(package_path):
        subprocess.run(["tar", "-xzf", package_path, "-C", "/content/"])
        print("   ✅ MWNN package extracted")
        
        # Add to Python path
        sys.path.insert(0, "/content/multi-weight-neural-networks")
        print("   ✅ Added to Python path")
    else:
        print("   ⚠️  MWNN package not found - please upload mwnn_colab_package.tar.gz")
    
    return in_colab


def download_imagenet_instructions():
    """Provide instructions for downloading ImageNet"""
    
    print("\n🗂️  ImageNet-1K Dataset Setup")
    print("="*40)
    
    print("📋 To download ImageNet-1K dataset:")
    print()
    print("Option 1 - Kaggle API (Recommended):")
    print("   1. Install Kaggle API: !pip install kaggle")
    print("   2. Upload your kaggle.json credentials")
    print("   3. Run: !kaggle competitions download -c imagenet-object-localization-challenge")
    print("   4. Extract to /content/data/ImageNet-1K/")
    print()
    
    print("Option 2 - Direct Download:")
    print("   1. Register at https://image-net.org/")
    print("   2. Download ILSVRC2012 training and validation sets")
    print("   3. Upload and extract to /content/data/ImageNet-1K/")
    print()
    
    print("Expected directory structure:")
    print("/content/data/ImageNet-1K/")
    print("├── train/")
    print("│   ├── n01440764/")
    print("│   ├── n01443537/")
    print("│   └── ... (1000 class folders)")
    print("├── val/")
    print("│   ├── ILSVRC2012_val_00000001.JPEG")
    print("│   └── ... (50,000 validation images)")
    print("└── ILSVRC2013_devkit/")
    print("    └── data/")
    print("        ├── ILSVRC2013_clsloc_validation_ground_truth.txt")
    print("        └── meta_clsloc.mat")
    print()


def test_gpu_setup():
    """Test GPU setup and detect optimal batch size"""
    
    print("\n🖥️  GPU Setup and Optimization")
    print("="*35)
    
    import torch
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        
        # Import and test batch size optimization
        try:
            sys.path.append('/content/multi-weight-neural-networks')
            from setup_colab import get_gpu_info, get_optimal_settings
            
            gpu_info = get_gpu_info()
            optimal_settings = get_optimal_settings(gpu_info)
            
            print(f"\n🎯 Optimal Settings Detected:")
            print(f"   Batch Size: {optimal_settings['batch_size']}")
            print(f"   Mixed Precision: {optimal_settings['mixed_precision']}")
            print(f"   Num Workers: {optimal_settings['num_workers']}")
            
        except ImportError:
            print("⚠️  MWNN package not found - using default settings")
            optimal_settings = {
                'batch_size': 32,
                'mixed_precision': True,
                'num_workers': 4
            }
    else:
        print("❌ No GPU available - ImageNet training requires CUDA")
        optimal_settings = None
    
    return optimal_settings


def create_colab_training_script():
    """Create a simplified training script for Colab"""
    
    script_content = '''
# Quick ImageNet Training on Colab
import sys
sys.path.append('/content/multi-weight-neural-networks')

from train_deep_colab import run_deep_training

# Run ImageNet training with optimal settings
results = run_deep_training(
    dataset_name='ImageNet',
    data_dir='/content/data/ImageNet-1K',
    devkit_dir='/content/data/ImageNet-1K/ILSVRC2013_devkit',
    complexity='deep',
    epochs=30,
    use_auto_batch_size=True,
    save_checkpoints=True
)

print("\\n🎯 Training Complete!")
print(f"Best Accuracy: {results['best_val_acc']:.2f}%")
'''
    
    script_path = "/content/quick_train.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\n📝 Created quick training script: {script_path}")
    print("   Run with: !python /content/quick_train.py")


def main():
    """Main setup function"""
    
    # Setup environment
    in_colab = setup_colab_environment()
    
    # Provide ImageNet download instructions
    download_imagenet_instructions()
    
    # Test GPU setup
    optimal_settings = test_gpu_setup()
    
    if in_colab and optimal_settings:
        # Create training script
        create_colab_training_script()
        
        print("\n🚀 Colab Setup Complete!")
        print("="*25)
        print("✅ Environment configured")
        print("✅ GPU detected and optimized")
        print("✅ Quick training script created")
        print()
        print("📋 Next Steps:")
        print("1. Download ImageNet-1K dataset")
        print("2. Run: !python /content/quick_train.py")
        print("3. Monitor training in /content/logs/")
        
    else:
        print("\n⚠️  Setup incomplete - please check GPU availability and dataset")


if __name__ == "__main__":
    main()

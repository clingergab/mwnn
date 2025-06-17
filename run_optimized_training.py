#!/usr/bin/env python3
"""
Run optimized ImageNet training with recommended hyperparameters.
"""

import subprocess
import sys

def run_optimized_training():
    """Run training with optimized hyperparameters for the continuous_integration model."""
    
    print("🚀 Starting Optimized ImageNet Training")
    print("=" * 50)
    
    # Recommended hyperparameters based on our GPU optimization tests
    cmd = [
        sys.executable, "train_imagenet_mwnn.py",
        "--model", "continuous_integration",
        "--epochs", "10",
        "--batch_size", "16",      # Conservative for MPS, can be 32-64 for CUDA
        "--learning_rate", "3e-4", # Lower for better convergence with smaller batch
        "--weight_decay", "1e-4",  # Good default
        "--config_preset", "training",
        "--enable_gpu_optimizations",
        "--use_mixed_precision",
        "--load_subset", "1000",   # Small subset for testing (remove for full training)
        "--num_workers", "4"       # Optimal for most systems
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\n📊 Hyperparameter Summary:")
    print("- Model: continuous_integration (GPU-optimized)")
    print("- Epochs: 10")
    print("- Batch Size: 16 (conservative for MPS stability)")
    print("- Learning Rate: 3e-4 (scaled for batch size)")
    print("- Weight Decay: 1e-4")
    print("- Scheduler: CosineAnnealingLR (automatic)")
    print("- Mixed Precision: Enabled")
    print("- Subset: 1000 samples (for testing)")
    print("\n🔧 To train on full dataset, remove --load_subset argument")
    print("🔧 For CUDA GPUs, you can use --batch_size 32 or 64")
    print("\n" + "=" * 50)
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    
    print("✅ Training completed successfully!")
    return True

if __name__ == "__main__":
    run_optimized_training()

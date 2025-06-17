# 🚀 MWNN ImageNet Training - Colab Quick Start

Copy and paste these cells into Google Colab for instant setup:

## Cell 1: Check GPU and Upload Package
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU - go to Runtime > Change runtime type > GPU")

print("\n📁 Upload mwnn_colab_package.tar.gz using the file browser on the left")
```

## Cell 2: Extract Package
```python
# Extract the MWNN package
!tar -xzf mwnn_colab_package.tar.gz
%cd multi-weight-neural-networks
!ls -la

print("✅ Package extracted successfully!")
```

## Cell 3: Setup Environment
```python
# Setup Colab environment for ImageNet training
!python setup_imagenet_colab.py

# Verify installation
!python -c "from train_deep_colab import run_deep_training; print('✅ MWNN ready!')"
```

## Cell 4: Setup ImageNet Dataset
```python
# Install Kaggle and setup directories
!pip install kaggle
!mkdir -p /content/data/ImageNet-1K

print("📋 Next steps:")
print("1. Upload your kaggle.json file")
print("2. Run the next cell to download ImageNet")
```

## Cell 5: Download ImageNet (after uploading kaggle.json)
```python
# Download ImageNet via Kaggle
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'

!kaggle competitions download -c imagenet-object-localization-challenge -p /content/data/
!cd /content/data && unzip -q imagenet-object-localization-challenge.zip -d ImageNet-1K/

# Verify download
!ls /content/data/ImageNet-1K/

print("✅ ImageNet dataset ready!")
```

## Cell 6: Start Training
```python
# Import and start training
from train_deep_colab import run_deep_training

# Configuration for Colab
config = {
    'dataset_name': 'ImageNet',
    'data_dir': '/content/data/ImageNet-1K',
    'devkit_dir': '/content/data/ImageNet-1K/ILSVRC2013_devkit',
    'complexity': 'deep',        # 45M parameter model
    'epochs': 30,
    'use_auto_batch_size': True, # Automatically optimize for your GPU
    'save_checkpoints': True
}

print("🚀 Starting ImageNet MWNN Training...")
print("This will take 20-40 hours depending on your GPU")
print("="*60)

# Start training
results = run_deep_training(**config)

# Show results
if results:
    print(f"\n🎯 Training Complete!")
    print(f"Best Accuracy: {results['best_val_acc']:.2f}%")
    print(f"Training Time: {results['total_training_time']/3600:.1f} hours")
```

## Cell 7: Monitor Progress (Optional - run in separate cell while training)
```python
# Monitor training progress
!tail -f /content/logs/training.log  # If logging to file
# Or check GPU usage
!nvidia-smi
```

## Cell 8: Download Results
```python
# Download trained model and results
from google.colab import files

# Download the best model
files.download('/content/checkpoints/best_imagenet_model.pth')

# Download training results
files.download('/content/results/imagenet_training_results.json')

print("✅ Results downloaded to your local machine!")
```

---

## 🎯 Pro Tips for Colab:

1. **Use Colab Pro**: Get better GPUs and longer runtime
2. **Enable GPU**: Runtime > Change runtime type > Hardware accelerator > GPU  
3. **Monitor Runtime**: Keep the tab open to prevent disconnection
4. **Save Frequently**: Checkpoints are saved automatically every 5 epochs
5. **Check Memory**: Run `!nvidia-smi` to monitor GPU usage

## 📱 Mobile Monitoring:
You can monitor training progress on your phone by:
- Keeping the Colab tab open
- Checking the output periodically
- Using TensorBoard (if enabled) for live plots

## ⚡ Quick Test (5 minutes):
To test everything works before full training:
```python
# Quick test with tiny subset
results = run_deep_training(
    dataset_name='ImageNet',
    complexity='shallow',  # Smaller model
    epochs=1,
    load_subset=100,      # Only 100 images
    use_auto_batch_size=True
)
```

**Total setup time: ~10-15 minutes**  
**Training time: 20-40 hours**  
**Expected accuracy: 75-80% on ImageNet**

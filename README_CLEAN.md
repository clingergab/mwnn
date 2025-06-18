# MWNN - Multi-Weight Neural Networks

A clean, Keras-like framework for dual-pathway image classification using Multi-Weight Neural Networks.

## 🚀 Quick Start

### Installation
```bash
git clone <repository>
cd multi-weight-neural-networks
pip install -r requirements.txt
```

### Simple Training
```python
from src.mwnn import MWNN

# Load ImageNet data
train_loader, val_loader = MWNN.load_imagenet_data('/path/to/imagenet', batch_size=64)

# Create model
model = MWNN(num_classes=1000, depth='deep', device='auto')

# Train
history = model.fit(train_loader, val_loader, epochs=30)

# Evaluate
results = model.evaluate(val_loader)
print(f"Accuracy: {results['accuracy']:.2f}%")

# Save
model.save('best_model.pth')
```

### Command Line Training
```bash
python train.py --data_path /path/to/imagenet --epochs 30 --batch_size 64
```

## 🧠 Model Architecture

MWNN uses a dual-pathway architecture:
- **RGB Pathway**: Processes standard RGB images
- **Brightness Pathway**: Processes brightness/luminance information
- **Integration Module**: Combines both pathways for enhanced classification

## 📊 Features

- **Clean API**: Keras-like interface for easy usage
- **Dual Pathways**: RGB + brightness for improved accuracy
- **Auto Device Detection**: Automatically uses GPU when available
- **Progress Bars**: Clean, single-line progress tracking
- **Checkpointing**: Automatic model saving during training
- **ImageNet Ready**: Optimized for ImageNet-1K classification

## 🛠️ Project Structure

```
multi-weight-neural-networks/
├── src/
│   ├── mwnn.py                 # Main MWNN class (clean API)
│   ├── models/                 # Neural network architectures
│   ├── preprocessing/          # Data loading and preprocessing
│   ├── training/               # Training utilities
│   └── utils/                  # Helper functions
├── tests/                      # Test suite
├── configs/                    # Configuration files
├── train.py                    # Simple training script
├── demo.py                     # API demonstration
└── README.md                   # This file
```

## 🎯 Examples

### Load and Inspect Model
```python
from src.mwnn import MWNN

model = MWNN(num_classes=1000, depth='deep')
model.summary()
```

### Load Saved Model
```python
model = MWNN.load('best_model.pth')
results = model.evaluate(test_loader)
```

### Make Predictions
```python
predictions = model.predict(test_loader)
```

## ⚙️ Configuration

### Model Depths
- `shallow`: Lightweight model for testing
- `medium`: Balanced model for moderate datasets
- `deep`: Full model for ImageNet-1K

### Training Parameters
- `learning_rate`: Default 0.002
- `batch_size`: Recommended 64 (T4) or 128 (A100)
- `epochs`: Typically 30 for ImageNet

## 🔧 Advanced Usage

### Custom Training Loop
```python
from src.training.trainer import MWNNTrainer

trainer = MWNNTrainer(model, device, learning_rate=0.001)
history = trainer.train(train_loader, val_loader, epochs=50)
```

### GPU Optimization
```python
from src.utils.device import get_optimal_device, get_gpu_info

device = get_optimal_device()
gpu_info = get_gpu_info()
print(f"Using device: {device}")
```

## 📈 Performance

- **ImageNet-1K**: Competitive accuracy with dual-pathway architecture
- **Memory Efficient**: Optimized for T4/A100 GPUs
- **Training Speed**: Clean progress bars with ETA estimates

## 🤝 Contributing

1. Keep the clean API design
2. Add tests for new features
3. Follow the existing code style
4. Update documentation

## 📄 License

[Your License Here]

---

**Note**: This is a refactored version focusing on clean, maintainable code with a simple API. The complex training scripts have been replaced with this Keras-like interface for better usability.

# ImageNet-1K Dataset Preprocessing for Multi-Weight Neural Networks

This document provides comprehensive guidance for using the ImageNet-1K dataset preprocessing utilities with Multi-Weight Neural Networks (MWNNs).

## Overview

The ImageNet preprocessing system provides:
- **Optimized data loading** for the ImageNet-1K validation dataset
- **RGB+Luminance feature extraction** preserving all original data while adding brightness information
- **Multiple legacy feature extraction methods** (HSV, LAB, YUV) for MWNN architectures  
- **Configurable preprocessing pipelines** with preset configurations
- **Performance benchmarking** and analysis tools
- **Integration** with the existing MWNN training framework

### New RGB+Luminance Approach

The system now implements the updated MWNN design that:
- **Preserves all RGB data**: No information loss from color space conversions
- **Adds explicit luminance**: ITU-R BT.709 standard luminance as 4th channel
- **Enables clean separation**: Color and brightness pathways from input layer
- **Minimizes overhead**: Only 33% memory increase (3→4 channels)

## Quick Start

### 1. Basic Dataset Loading (New RGB+Luminance Approach)

```python
from src.preprocessing.imagenet_dataset import create_imagenet_rgb_luminance_dataset

# Create dataset with 4-channel RGB+Luminance format
dataset = create_imagenet_rgb_luminance_dataset(
    data_dir="data/ImageNet-1K",
    devkit_dir="data/ImageNet-1K/ILSVRC2013_devkit",
    input_size=224,
    augment=True,
    load_subset=1000  # For testing, remove for full dataset
)

print(f"Dataset loaded with {len(dataset)} samples")

# Each sample returns 4-channel tensor: RGB + Luminance
image, label = dataset[0]
print(f"Image shape: {image.shape}")  # Should be (4, 224, 224)

# Extract color and brightness pathways
color_channels, brightness_channel = dataset.get_color_brightness_channels(image)
print(f"Color (RGB): {color_channels.shape}")      # (3, 224, 224)
print(f"Brightness: {brightness_channel.shape}")   # (1, 224, 224)
```

### 1b. Legacy Feature Extraction (HSV/LAB/YUV)

```python
from src.preprocessing.imagenet_dataset import create_imagenet_mwnn_dataset

# Create dataset with legacy color space conversion
dataset = create_imagenet_mwnn_dataset(
    data_dir="data/ImageNet-1K",
    devkit_dir="data/ImageNet-1K/ILSVRC2013_devkit",
    input_size=224,
    feature_method='hsv',  # or 'lab', 'yuv'
    augment=True,
    load_subset=1000
)
```

### 2. Using Configuration Presets (RGB+Luminance DataLoaders)

```python
from src.preprocessing.imagenet_config import get_preset_config
from src.preprocessing.imagenet_dataset import create_imagenet_rgb_luminance_dataloaders

# Load development preset
config = get_preset_config(
    preset='development',
    data_dir='data/ImageNet-1K',
    devkit_dir='data/ImageNet-1K/ILSVRC2013_devkit'
)

# Create dataloaders with 4-channel RGB+Luminance format
train_loader, val_loader = create_imagenet_rgb_luminance_dataloaders(
    data_dir=config.data_dir,
    devkit_dir=config.devkit_dir,
    batch_size=config.batch_size,
    input_size=config.input_size,
    num_workers=config.num_workers,
    load_subset=config.load_subset
)

# Each batch contains 4-channel tensors
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")  # (batch_size, 4, 224, 224)
    
    # Extract pathways for MWNN processing
    color_pathway = images[:, :3, :, :]    # RGB channels
    brightness_pathway = images[:, 3:4, :, :] # Luminance channel
    break
```

### 3. Configuration Files

```python
from src.preprocessing.imagenet_config import ImageNetPreprocessingConfig

# Load from YAML configuration
config = ImageNetPreprocessingConfig.from_yaml('configs/preprocessing/imagenet_training.yaml')

# Update paths for your setup
config.data_dir = '/your/path/to/ImageNet-1K'
config.devkit_dir = '/your/path/to/ImageNet-1K/ILSVRC2013_devkit'
```

## Dataset Structure

### Expected Directory Layout

```
data/ImageNet-1K/
├── val_images/                     # 50,000 validation images
│   ├── ILSVRC2012_val_00000001_n01440764.JPEG
│   ├── ILSVRC2012_val_00000002_n01443537.JPEG
│   └── ...
└── ILSVRC2013_devkit/
    └── data/
        ├── ILSVRC2013_clsloc_validation_ground_truth.txt
        ├── meta_clsloc.mat
        └── ...
```

### Label Mapping

- **Ground truth file**: Contains class indices (1-1000) for each validation image
- **Metadata file**: Maps synset IDs to class names (optional, fallback available)
- **Class names**: Human-readable class descriptions

## Feature Extraction

### Supported Methods

1. **HSV (Hue, Saturation, Value)**
   - Good for color-based discrimination
   - Robust to lighting changes
   - Default method for MWNN

2. **LAB (L*a*b* color space)**
   - Perceptually uniform color space
   - Better for color similarity metrics
   - Good for fine-grained classification

3. **YUV (Luminance and Chrominance)**
   - Separates brightness from color information
   - Good for texture analysis
   - Efficient for compression

### Example Usage

```python
# Different feature extraction methods
methods = ['hsv', 'lab', 'yuv']

for method in methods:
    dataset = create_imagenet_mwnn_dataset(
        data_dir="data/ImageNet-1K",
        devkit_dir="data/ImageNet-1K/ILSVRC2013_devkit",
        feature_method=method,
        load_subset=100
    )
    
    # Extract features for first sample
    image, label = dataset[0]
    print(f"{method.upper()}: Shape {image.shape}, Label {label}")
```

## Configuration Presets

### Available Presets

| Preset | Use Case | Batch Size | Augmentation | Subset | Workers |
|--------|----------|------------|--------------|--------|---------|
| `development` | Quick testing | 16 | No | 100 | 2 |
| `training` | Model training | 32 | Yes | None | 4 |
| `evaluation` | Model evaluation | 64 | No | None | 4 |
| `research` | High-quality experiments | 16 | Enhanced | None | 6 |

### Customizing Configurations

```python
# Start with a preset and customize
config = get_preset_config(
    preset='training',
    data_dir='data/ImageNet-1K',
    devkit_dir='data/ImageNet-1K/ILSVRC2013_devkit',
    # Custom overrides
    batch_size=64,
    input_size=256,
    feature_method='lab'
)
```

## Data Augmentation

### Standard Augmentations

- **Random Resized Crop**: Scale (0.8, 1.0)
- **Random Horizontal Flip**: 50% probability
- **Color Jitter**: Brightness, contrast, saturation, hue variations
- **Normalization**: ImageNet mean and std

### Custom Augmentation

```python
from torchvision import transforms

# Custom transform pipeline
custom_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Performance Optimization

### Recommended Settings

```python
# For development/debugging
config = {
    'batch_size': 16,
    'num_workers': 2,
    'load_subset': 100,
    'pin_memory': True
}

# For training
config = {
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True
}

# For evaluation
config = {
    'batch_size': 64,
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': False
}
```

### Memory Considerations

- **Subset loading**: Use `load_subset` parameter for testing
- **Batch size**: Balance between memory usage and throughput
- **Workers**: More workers = faster loading but more memory
- **Pin memory**: Enable for GPU training

## Integration with MWNN Training

### Basic Integration

```python
from src.preprocessing.imagenet_dataset import create_imagenet_dataloaders
from src.models import MultiWeightNeuralNetwork
from src.training import MWNNTrainer

# Create dataloaders
train_loader, val_loader = create_imagenet_dataloaders(
    data_dir="data/ImageNet-1K",
    devkit_dir="data/ImageNet-1K/ILSVRC2013_devkit",
    batch_size=32,
    feature_method='hsv'
)

# Create model
model = MultiWeightNeuralNetwork(
    num_classes=1000,
    feature_dim=3,  # HSV channels
    # ... other parameters
)

# Train model
trainer = MWNNTrainer(model, train_loader, val_loader)
trainer.train()
```

### Advanced Integration

```python
# Load configuration
config = ImageNetPreprocessingConfig.from_yaml('configs/preprocessing/imagenet_training.yaml')

# Create model with configuration
model = MultiWeightNeuralNetwork(
    num_classes=1000,
    input_size=config.input_size,
    feature_method=config.feature_method,
    # ... other parameters from config
)

# Create optimized dataloaders
train_loader, val_loader = create_imagenet_dataloaders(
    data_dir=str(config.data_dir),
    devkit_dir=str(config.devkit_dir),
    batch_size=config.batch_size,
    input_size=config.input_size,
    feature_method=config.feature_method,
    num_workers=config.num_workers,
    load_subset=config.load_subset,
    val_split=config.val_split
)
```

## Command Line Tools

### Dataset Analysis

```bash
# Analyze dataset structure and statistics
python src/preprocessing/preprocessing_utils.py analyze --sample-size 1000

# Test loading performance
python src/preprocessing/preprocessing_utils.py test-loading \
    --batch-size 32 --num-workers 4

# Benchmark different configurations
python src/preprocessing/preprocessing_utils.py benchmark \
    --benchmark-batches 100
```

### Configuration Management

```bash
# Create default configuration files
python src/preprocessing/preprocessing_utils.py create-configs

# Validate configuration file
python src/preprocessing/preprocessing_utils.py validate-config \
    configs/preprocessing/imagenet_training.yaml --test-dataset
```

## Troubleshooting

### Common Issues

1. **Ground truth file not found**
   ```
   ERROR: Ground truth file not found
   ```
   - Check that `ILSVRC2013_clsloc_validation_ground_truth.txt` exists
   - Verify devkit directory path

2. **Memory errors with large batch sizes**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce batch size
   - Reduce number of workers
   - Use subset loading for testing

3. **Slow loading performance**
   - Increase number of workers
   - Enable pin_memory for GPU training
   - Use SSD storage for data

4. **Image loading errors**
   ```
   Error loading image: cannot identify image file
   ```
   - Check image file integrity
   - Verify file permissions
   - Use dataset validation tools

### Performance Tips

1. **Use appropriate worker count**: Generally 2-4x number of CPU cores
2. **Enable persistent workers**: Reduces worker restart overhead
3. **Use appropriate batch size**: Balance memory and throughput
4. **Cache frequently used data**: Enable label and feature caching
5. **Use fast storage**: SSD recommended for image data

## Examples

### Complete Training Example

```python
import torch
from src.preprocessing.imagenet_config import ImageNetPreprocessingConfig
from src.preprocessing.imagenet_dataset import create_imagenet_dataloaders
from src.models import MultiWeightNeuralNetwork
from src.training import MWNNTrainer

# Load configuration
config = ImageNetPreprocessingConfig.from_yaml('configs/preprocessing/imagenet_training.yaml')

# Update paths
config.data_dir = 'data/ImageNet-1K'
config.devkit_dir = 'data/ImageNet-1K/ILSVRC2013_devkit'

# Create dataloaders
train_loader, val_loader = create_imagenet_dataloaders(
    data_dir=str(config.data_dir),
    devkit_dir=str(config.devkit_dir),
    batch_size=config.batch_size,
    input_size=config.input_size,
    feature_method=config.feature_method,
    num_workers=config.num_workers,
    val_split=config.val_split
)

# Create model
model = MultiWeightNeuralNetwork(
    num_classes=1000,
    input_size=config.input_size,
    feature_method=config.feature_method
)

# Create trainer
trainer = MWNNTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Train
trainer.train(epochs=10)
```

### Evaluation Example

```python
# Load evaluation configuration
config = ImageNetPreprocessingConfig.from_yaml('configs/preprocessing/imagenet_evaluation.yaml')

# Create evaluation dataset (no augmentation)
eval_dataset = create_imagenet_mwnn_dataset(
    data_dir=str(config.data_dir),
    devkit_dir=str(config.devkit_dir),
    input_size=config.input_size,
    feature_method=config.feature_method,
    augment=False  # Important for evaluation
)

# Create evaluation dataloader
eval_loader = DataLoader(
    eval_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory
)

# Load trained model and evaluate
model = MultiWeightNeuralNetwork.load_from_checkpoint('checkpoints/best_model.pth')
evaluator = MWNNEvaluator(model, eval_loader)
results = evaluator.evaluate()

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
```

## API Reference

See the individual module documentation:
- `src.preprocessing.imagenet_dataset`: Dataset classes and functions
- `src.preprocessing.imagenet_config`: Configuration management
- `src.preprocessing.preprocessing_utils`: Command-line utilities

## Contributing

When adding new features to the preprocessing system:
1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Consider performance implications
5. Maintain backward compatibility

# ImageNet Preprocessing Configurations

This directory contains preprocessing configurations for ImageNet-1K dataset with Multi-Weight Neural Networks.

## Available Configurations

1. **imagenet_development.yaml**: Fast loading for development/debugging
   - Small subset (100 samples)
   - No augmentation
   - Minimal resources

2. **imagenet_training.yaml**: Optimized for training
   - Full augmentation
   - Feature caching
   - Balanced performance

3. **imagenet_evaluation.yaml**: For model evaluation
   - No augmentation
   - Larger batch sizes
   - Full dataset

4. **imagenet_research.yaml**: High-quality preprocessing
   - Enhanced augmentation
   - Higher resolution
   - Research-grade quality

5. **imagenet_template.yaml**: Template for custom configurations

## Usage

```python
from src.preprocessing.imagenet_config import ImageNetPreprocessingConfig
from src.preprocessing import create_imagenet_dataloaders

# Load configuration
config = ImageNetPreprocessingConfig.from_yaml('configs/preprocessing/imagenet_training.yaml')

# Update paths to your data
config.data_dir = '/your/path/to/ImageNet-1K'
config.devkit_dir = '/your/path/to/ImageNet-1K/ILSVRC2013_devkit'

# Create dataloaders
train_loader, val_loader = create_imagenet_dataloaders(
    data_dir=config.data_dir,
    devkit_dir=config.devkit_dir,
    batch_size=config.batch_size,
    input_size=config.input_size,
    feature_method=config.feature_method,
    num_workers=config.num_workers,
    load_subset=config.load_subset,
    val_split=config.val_split
)
```

## Customization

1. Copy `imagenet_template.yaml` to a new file
2. Modify the paths and parameters as needed
3. Load using `ImageNetPreprocessingConfig.from_yaml()`

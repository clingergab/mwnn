# Multi-Channel Model

The Multi-Channel Model maintains separate processing pathways for color and brightness throughout the network, inspired by the parallel pathways in biological visual systems.

## Architecture Overview

```
Input Image → Feature Extraction → Color/Brightness Separation
                                          ↓
                    ┌────────────────────────────────────┐
                    │                                    │
                Color Pathway                   Brightness Pathway
                    │                                    │
                Conv Blocks                        Conv Blocks
                    │                                    │
                Residual Blocks                   Residual Blocks
                    │                                    │
                Global Context                    Global Context
                    │                                    │
                    └────────────┬───────────────────────┘
                                 │
                         Feature Fusion
                                 │
                         Classification Head
                                 │
                              Output
```

## Key Features

1. **Separate Pathways**: Color (Hue + Saturation) and Brightness (Value) are processed independently
2. **Flexible Fusion**: Multiple fusion strategies (concatenate, add, adaptive, attention)
3. **Global Context**: Each pathway incorporates global context information
4. **Scalable Depth**: Three depth configurations (shallow, medium, deep)

## Usage

### Basic Usage

```python
from mwnn.models.multi_channel import MultiChannelModel

# Create model
model = MultiChannelModel(
    num_classes=10,
    depth='medium',
    fusion_method='adaptive'
)

# Forward pass
outputs = model(images)
```

### Advanced Configuration

```python
# Create a deep model with attention fusion
model = MultiChannelModel(
    input_channels=3,
    num_classes=1000,
    feature_extraction_method='lab',  # Use LAB color space
    base_channels=128,
    depth='deep',
    fusion_method='attention',
    dropout_rate=0.3
)

# Get pathway-specific outputs
color_features, brightness_features = model.get_pathway_outputs(images)
```

## Training Tips

1. **Multi-Stage Training**: 
   - Stage 1: Train pathways separately by freezing one at a time
   - Stage 2: Fine-tune the entire network

2. **Pathway Balancing**:
   - Monitor pathway correlation to ensure specialization
   - Use pathway balance loss if one dominates

3. **Augmentation Strategy**:
   - Apply color augmentations primarily to color pathway inputs
   - Apply brightness augmentations primarily to brightness pathway inputs

## Model Variants

### Fusion Methods

- **Concatenate**: Simple feature concatenation
- **Add**: Element-wise addition (requires same dimensions)
- **Adaptive**: Learned gating mechanism that weights pathways
- **Attention**: Attention-based fusion for dynamic weighting

### Depth Configurations

| Config | Blocks | Channels | Parameters | Use Case |
|--------|--------|----------|------------|----------|
| Shallow | [2, 2] | [64, 128] | ~2M | Quick experiments, small datasets |
| Medium | [2, 2, 2] | [64, 128, 256] | ~8M | Standard tasks, CIFAR-like datasets |
| Deep | [3, 4, 6, 3] | [64, 128, 256, 512] | ~25M | Complex tasks, ImageNet-scale |

## Evaluation Metrics

The model provides several specialized metrics:

- **Pathway Correlation**: Measures independence of pathways
- **Fusion Weights Entropy**: Indicates uncertainty in pathway selection
- **Pathway Activation Statistics**: Mean, std, sparsity per pathway

## Citation

If you use this model, please cite:
```bibtex
@article{multiweight2024,
  title={Multi-Channel Neural Networks for Color-Brightness Decomposition},
  author={Gabriel Clinger},
  year={2025}
}
```
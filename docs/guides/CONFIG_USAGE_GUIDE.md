# Using Model Configuration Files - Quick Start Guide

## 📋 **Overview**

All models in the Multi-Weight Neural Networks project now have standardized `config.yaml` files that define their default parameters, training settings, and architectural configurations.

## 🚀 **Quick Start**

### **1. Load a Model Config**

```python
from src.utils.config import get_model_config

# Load default configuration for any model
config = get_model_config('continuous_integration')
print(config['model']['params'])
```

### **2. Create a Model from Config**

```python
from src.models.continuous_integration.model import ContinuousIntegrationModel

# Get config and create model
config = get_model_config('continuous_integration')
model = ContinuousIntegrationModel(**config['model']['params'])
```

### **3. Training with Config**

```python
from src.utils.config import get_model_config
from src.training.trainer import Trainer

# Load model config
config = get_model_config('attention_based')

# Create model
model = AttentionBasedModel(**config['model']['params'])

# Create trainer with config settings
trainer = Trainer(
    model=model,
    learning_rate=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay'],
    scheduler=config['training']['scheduler']
)
```

## 🎛️ **Available Models & Their Configs**

### **1. Multi-Channel Model**
```yaml
model:
  name: MultiChannelModel
  params:
    fusion_method: adaptive  # concatenate, add, adaptive, attention
    feature_extraction_method: hsv
    depth: medium
```

### **2. Continuous Integration Model**  
```yaml
model:
  name: ContinuousIntegrationModel
  params:
    integration_points: ['early', 'middle', 'late']
    integration_weight_regularization: 0.01
    depth: medium
```

### **3. Attention-Based Model**
```yaml
model:
  name: AttentionBasedModel
  params:
    attention_dim: 64
    num_attention_heads: 4
    attention_dropout: 0.1
```

### **4. Cross-Modal Model**
```yaml
model:
  name: CrossModalModel
  params:
    cross_influence: 0.1
    influence_mechanism: linear
    depth: medium
```

### **5. Single-Output Model**
```yaml
model:
  name: SingleOutputModel
  params:
    adaptive_fusion: false
    weight_specialization_loss: 0.01
    depth: medium
```

## 🔧 **Customizing Configurations**

### **Method 1: Override Parameters**

```python
from src.utils.config import get_model_config, merge_configs

# Load base config
base_config = get_model_config('multi_channel')

# Create override config
override = {
    'model': {
        'params': {
            'depth': 'deep',
            'fusion_method': 'attention'
        }
    },
    'training': {
        'learning_rate': 0.0005
    }
}

# Merge configurations
final_config = merge_configs(base_config, override)
```

### **Method 2: Custom Config File**

```yaml
# custom_experiment.yaml
model:
  name: ContinuousIntegrationModel
  params:
    input_channels: 3
    num_classes: 100  # CIFAR-100
    depth: deep
    integration_points: ['all']

training:
  learning_rate: 0.0001
  num_epochs: 200
  
data:
  dataset: CIFAR100
```

```python
from src.utils.config import load_config

# Load custom config
config = load_config('custom_experiment.yaml')
model = ContinuousIntegrationModel(**config['model']['params'])
```

## 🧪 **Model-Specific Features**

### **Continuous Integration Model**
- **Integration Points**: Control where pathways are integrated
- **Regularization**: Pathway consistency and integration weight regularization
- **Metrics**: Integration weight evolution, gradient flow analysis

### **Attention-Based Model**  
- **Multi-Head Attention**: Configurable number of attention heads
- **Cross-Modal Attention**: Attention between color/brightness pathways
- **Metrics**: Attention weight analysis, cross-modal correlation

### **Cross-Modal Model**
- **Cross-Influence**: Bidirectional pathway influence strength
- **Influence Scheduling**: Dynamic influence strength during training
- **Metrics**: Information flow analysis, pathway independence

### **Multi-Channel Model**
- **Fusion Methods**: Multiple pathway combination strategies
- **Pathway Augmentations**: Separate augmentations for each pathway
- **Metrics**: Pathway correlation, fusion weight entropy

### **Single-Output Model**
- **Adaptive Fusion**: Learnable pathway combination weights
- **Weight Specialization**: Specialized weights for different features
- **Metrics**: Weight specialization, pathway utilization

## 📊 **Common Training Patterns**

### **Standard Training**
```python
# Quick model creation and training
config = get_model_config('multi_channel')
model = MultiChannelModel(**config['model']['params'])

trainer = Trainer(model)
trainer.train(train_loader, val_loader, epochs=100)
```

### **Hyperparameter Sweep**
```python
# Test different depths
depths = ['shallow', 'medium', 'deep']
results = {}

for depth in depths:
    config = get_model_config('continuous_integration')
    config['model']['params']['depth'] = depth
    
    model = ContinuousIntegrationModel(**config['model']['params'])
    results[depth] = train_and_evaluate(model)
```

### **Multi-Model Comparison**
```python
models_to_test = [
    'multi_channel',
    'continuous_integration', 
    'attention_based',
    'cross_modal',
    'single_output'
]

results = {}
for model_type in models_to_test:
    config = get_model_config(model_type)
    # Create and train each model...
    results[model_type] = accuracy
```

## 🎯 **Best Practices**

1. **Start with Defaults**: Use the provided config files as starting points
2. **Version Control Configs**: Save experiment configs for reproducibility  
3. **Document Changes**: Comment configuration modifications
4. **Test Small First**: Try shallow models before deep ones
5. **Monitor Metrics**: Use model-specific evaluation metrics

## 🔍 **Troubleshooting**

### **Config Loading Issues**
```python
try:
    config = get_model_config('my_model')
except ValueError as e:
    print(f"Config error: {e}")
    # Use minimal default config instead
```

### **Model Creation Issues**
```python
try:
    model = MultiChannelModel(**config['model']['params'])
except TypeError as e:
    print(f"Parameter error: {e}")
    # Check parameter names and types
```

### **Training Issues**
```python
# Validate training config
required_keys = ['learning_rate', 'weight_decay', 'scheduler']
missing = [key for key in required_keys if key not in config['training']]
if missing:
    print(f"Missing training config: {missing}")
```

## 🎉 **Summary**

The standardized config files provide:
- ✅ **Consistent interface** across all model types
- ✅ **Model-specific parameters** for each architecture  
- ✅ **Default hyperparameters** for quick experimentation
- ✅ **Easy customization** via config overrides
- ✅ **Reproducible experiments** with version-controlled configs

Happy experimenting with your Multi-Weight Neural Networks! 🚀

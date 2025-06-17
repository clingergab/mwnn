"""Configuration utilities for Multi-Weight Neural Networks."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML or JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate configuration against a schema.
    
    Returns:
        Dict: Validated and potentially modified configuration
    """
    # Make a copy to avoid modifying the original
    config = config.copy()
    
    # Basic validation
    required_keys = ['model', 'training', 'data']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Model validation
    if 'params' not in config['model']:
        config['model']['params'] = {}
    
    # Training validation with defaults
    training_defaults = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    }
    
    if 'training' not in config:
        config['training'] = {}
    
    for key, default in training_defaults.items():
        if key not in config['training']:
            config['training'][key] = default
    
    # Data validation
    if 'data' not in config:
        config['data'] = {
            'dataset': 'CIFAR10',
            'data_dir': './data'
        }
    
    return config


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for a specific model."""
    config_path = Path(__file__).parent.parent / 'models' / model_name / 'config.yaml'
    
    if config_path.exists():
        config = load_config(str(config_path))
        return validate_config(config)
    else:
        # Return minimal default config
        default_config = {
            'model': {
                'name': model_name,
                'params': {}
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adamw',
                'scheduler': 'cosine'
            },
            'data': {
                'dataset': 'CIFAR10',
                'augmentation': {
                    'random_crop': True,
                    'random_horizontal_flip': True,
                    'normalize': True
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'per_class_accuracy']
            }
        }
        return validate_config(default_config)
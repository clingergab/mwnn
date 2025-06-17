"""
Simple GPU detection for Colab training script
"""

import torch


def get_gpu_info():
    """Get GPU information"""
    info = {}
    
    if torch.cuda.is_available():
        info['name'] = torch.cuda.get_device_name(0)
        info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['device'] = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['name'] = 'Apple Silicon GPU (MPS)'
        info['memory_gb'] = 'Unknown'
        info['device'] = 'mps'
    else:
        info['name'] = 'CPU'
        info['memory_gb'] = 'N/A'
        info['device'] = 'cpu'
    
    return info


def get_optimal_settings():
    """Get optimal settings based on hardware"""
    gpu_info = get_gpu_info()
    
    if 'A100' in gpu_info.get('name', ''):
        return {'batch_size': 128, 'num_workers': 8}
    elif 'T4' in gpu_info.get('name', ''):
        return {'batch_size': 64, 'num_workers': 4}
    elif gpu_info['device'] == 'mps':
        return {'batch_size': 32, 'num_workers': 4}
    else:
        return {'batch_size': 16, 'num_workers': 2}

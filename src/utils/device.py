"""
Device utilities for MWNN
"""

import torch


def get_optimal_device():
    """
    Get the optimal device for training/inference.
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_gpu_info():
    """
    Get information about available GPUs.
    
    Returns:
        dict: GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'device_name': None,
        'memory_info': None
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name()
        
        if info['device_count'] > 0:
            info['memory_info'] = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
    
    return info

"""GPU optimization utilities for Continuous Integration Model."""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging


class GPUOptimizer:
    """Utility class for GPU optimizations in MWNN models."""
    
    @staticmethod
    def detect_optimal_device() -> torch.device:
        """Detect the optimal device for training/inference with priority for Mac M-series."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Mac M-series GPU (Apple Silicon) - preferred for Mac systems
            logging.info("Using Apple Silicon GPU (MPS backend)")
            return torch.device('mps')
        elif torch.cuda.is_available():
            # NVIDIA GPU
            device_name = torch.cuda.get_device_name(0)
            logging.info(f"Using NVIDIA GPU: {device_name}")
            return torch.device('cuda')
        else:
            # CPU fallback
            logging.info("Using CPU (no GPU available)")
            return torch.device('cpu')
    
    @staticmethod
    def configure_backends(device: torch.device):
        """Configure backends for optimal performance."""
        if device.type == 'cuda':
            # NVIDIA GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable TensorFloat-32 for better performance on A100/RTX 30xx+
            torch.backends.cuda.matmul.allow_tf32 = True
            
        elif device.type == 'mps':
            # Mac M-series optimizations
            # Note: MPS doesn't have equivalent settings to cuDNN
            # But we can optimize memory allocation
            try:
                # Set memory fraction if available
                torch.mps.set_per_process_memory_fraction(0.8)
            except AttributeError:
                # Fallback for older PyTorch versions
                pass
    
    @staticmethod
    def get_memory_info(device: torch.device) -> Dict[str, float]:
        """Get memory usage information for the device."""
        if device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved(device) / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024**3,
                'total_gb': torch.cuda.get_device_properties(device).total_memory / 1024**3
            }
        elif device.type == 'mps':
            try:
                return {
                    'allocated_gb': torch.mps.current_allocated_memory() / 1024**3,
                    'driver_allocated_gb': torch.mps.driver_allocated_memory() / 1024**3
                }
            except AttributeError:
                return {'mps_memory': 'not_available'}
        else:
            return {'device': 'cpu', 'memory_tracking': 'not_available'}
    
    @staticmethod
    def clear_cache(device: torch.device):
        """Clear GPU memory cache."""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except AttributeError:
                # Fallback for older PyTorch versions
                pass
    
    @staticmethod
    def compile_model(model: nn.Module, device: torch.device, mode: str = 'default') -> nn.Module:
        """Compile model for better performance if supported."""
        # Disable torch.compile for now due to stability issues with MPS
        if not hasattr(torch, 'compile'):
            return model
        
        # Temporarily disable compilation due to tensor stride issues
        # TODO: Re-enable when PyTorch MPS compilation is more stable
        return model
        
        # Original compilation code (disabled)
        # if device.type in ['cuda', 'mps']:
        #     try:
        #         if mode == 'max_performance':
        #             compiled_model = torch.compile(model, mode='max-autotune')
        #         elif mode == 'memory_efficient':
        #             compiled_model = torch.compile(model, mode='reduce-overhead')
        #         else:
        #             compiled_model = torch.compile(model)
        #         
        #         logging.info(f"Model compiled for {device.type} with mode: {mode}")
        #         return compiled_model
        #         
        #     except Exception as e:
        #         logging.warning(f"Model compilation failed: {e}. Using uncompiled model.")
        #         return model
        # 
        # return model
    
    @staticmethod
    def optimize_memory_layout(tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory layout for better GPU utilization."""
        if tensor.device.type in ['cuda', 'mps']:
            # Ensure tensor is contiguous for better memory access patterns
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
        
        return tensor
    
    @staticmethod
    def enable_mixed_precision_support(device: torch.device) -> bool:
        """Check if mixed precision is supported for the device."""
        if device.type == 'cuda':
            # Check for Tensor Core support (compute capability >= 7.0)
            try:
                major, minor = torch.cuda.get_device_capability(device)
                return major >= 7 or (major == 6 and minor >= 1)
            except Exception:
                return False
        elif device.type == 'mps':
            # MPS supports automatic mixed precision on newer PyTorch versions
            return hasattr(torch.amp, 'autocast')
        
        return False


class ModelProfiler:
    """Profiler for analyzing model performance."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.profiling_data = {}
    
    def profile_forward_pass(self, rgb_data: torch.Tensor, brightness_data: torch.Tensor, 
                           num_iterations: int = 10) -> Dict[str, Any]:
        """Profile forward pass performance."""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(rgb_data, brightness_data)
        
        # Synchronize GPU operations
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            try:
                torch.mps.synchronize()
            except AttributeError:
                pass
        
        # Profile
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = self.model(rgb_data, brightness_data)
                
                # Synchronize GPU operations
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    try:
                        torch.mps.synchronize()
                    except AttributeError:
                        pass
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        return {
            'mean_time_ms': sum(times) / len(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'throughput_fps': 1.0 / (sum(times) / len(times)),
            'memory_info': GPUOptimizer.get_memory_info(self.device)
        }

"""Continuous Integration Neural Network implementation - GPU Optimized."""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from ..components.blocks import ConvBlock
from .integration_module import IntegrationModule, IntegrationStage


class ContinuousIntegrationModel(nn.Module):
    """Model with continuous learnable integration of color and brightness pathways.
    
    Optimized for GPU usage including Mac M-series GPUs (Apple Silicon).
    Features:
    - Memory-efficient tensor operations
    - GPU-optimized forward pass
    - Automatic mixed precision support
    - Apple Silicon MPS backend support
    """
    
    def __init__(self,
                 num_classes: int = 1000,  # ImageNet-1K
                 base_channels: int = 64,
                 depth: str = 'medium',
                 dropout_rate: float = 0.2,
                 integration_points: List[str] = ['early', 'middle', 'late'],
                 enable_mixed_precision: bool = True,
                 memory_efficient: bool = True):
        
        super().__init__()
        
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.integration_points = integration_points
        self.num_classes = num_classes
        self.enable_mixed_precision = enable_mixed_precision
        self.memory_efficient = memory_efficient
        
        # Device detection for optimal GPU usage
        self.device = self._detect_optimal_device()
        
        # Architecture configurations - Memory optimized for MPS
        self.depth_configs = {
            'shallow': {'blocks': [1, 1], 'channels': [self.base_channels, self.base_channels * 2]},
            'medium': {'blocks': [2, 2, 2], 'channels': [self.base_channels, self.base_channels * 2, self.base_channels * 4]},
            'deep': {'blocks': [2, 3, 4, 2], 'channels': [self.base_channels * 2, self.base_channels * 4, self.base_channels * 8, self.base_channels * 16]}
        }
        
        self._build_model()
        
        # Automatically move to optimal device and apply optimizations
        self.to_device()
        
        # Apply GPU optimizations after model is built and moved
        self._apply_gpu_optimizations()
        
        # Enable gradient checkpointing for memory efficiency on MPS
        if self.memory_efficient and self.device.type == 'mps':
            self.enable_gradient_checkpointing(True)
    
    def _detect_optimal_device(self) -> torch.device:
        """Detect the optimal device for training/inference."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Mac M-series GPU (Apple Silicon)
            return torch.device('mps')
        elif torch.cuda.is_available():
            # NVIDIA GPU
            return torch.device('cuda')
        else:
            # CPU fallback
            return torch.device('cpu')
    
    def _apply_gpu_optimizations(self):
        """Apply GPU-specific optimizations."""
        # Enable cuDNN benchmark for consistent input sizes
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # For better performance
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        elif self.device.type == 'mps':
            # Mac M-series optimizations with more conservative memory usage
            try:
                torch.mps.set_per_process_memory_fraction(0.5)  # More conservative
            except AttributeError:
                pass
        
        # Note: torch.compile is disabled for now due to stability issues
        # Can be re-enabled in the future when more stable
        # if hasattr(torch, 'compile') and self.device.type in ['cuda', 'mps']:
        #     try:
        #         self = torch.compile(self, mode='max-autotune')
        #     except Exception:
        #         pass
    
    def _build_model(self):
        """Build the continuous integration architecture with GPU optimizations."""
        config = self.depth_configs[self.depth]
        
        # Initial separate processing - direct pathway inputs
        # Use groups=1 for better GPU utilization on separate pathways
        self.initial_color = ConvBlock(3, self.base_channels)      # RGB: 3 channels
        self.initial_brightness = ConvBlock(1, self.base_channels) # Brightness: 1 channel
        
        # Build processing stages with integration - use ModuleList for better memory management
        self.stages = nn.ModuleList()
        self.integration_modules = nn.ModuleDict()
        
        in_channels = self.base_channels
        
        for i, (out_channels, num_blocks) in enumerate(zip(config['channels'], config['blocks'])):
            stage_name = f'stage_{i}'
            
            # Create stage with separate pathways - optimized for parallel execution
            stage = IntegrationStage(
                in_channels, out_channels, num_blocks, 
                downsample=(i > 0),
                memory_efficient=self.memory_efficient
            )
            self.stages.append(stage)
            
            # Add integration module if specified - with GPU optimizations
            if self._should_integrate(i, len(config['channels'])):
                self.integration_modules[stage_name] = IntegrationModule(
                    out_channels, out_channels,
                    memory_efficient=self.memory_efficient
                )
            
            in_channels = out_channels
        
        # Global pooling - use adaptive for variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final integration and classification with optimizations
        final_channels = config['channels'][-1]
        self.final_integration = IntegrationModule(
            final_channels, final_channels,
            memory_efficient=self.memory_efficient
        )
        
        # Optimized classifier with better GPU utilization
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels, final_channels // 2),
            nn.ReLU(inplace=True),  # In-place operations save memory
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_channels // 2, self.num_classes)
        )
    
    def _should_integrate(self, stage_idx: int, total_stages: int) -> bool:
        """Determine if integration should happen at this stage."""
        if 'early' in self.integration_points and stage_idx == 0:
            return True
        if 'middle' in self.integration_points and stage_idx == total_stages // 2:
            return True
        if 'late' in self.integration_points and stage_idx == total_stages - 1:
            return True
        if 'all' in self.integration_points:
            return True
        return False
    
    def forward(self, rgb_data: torch.Tensor, brightness_data: torch.Tensor) -> torch.Tensor:
        """
        GPU-optimized forward pass with continuous integration.
        
        Args:
            rgb_data: RGB color data, shape (B, 3, H, W)
            brightness_data: Brightness/luminance data, shape (B, 1, H, W)
            
        Returns:
            Classification logits, shape (B, num_classes)
        """
        # Ensure tensors are on the correct device
        if rgb_data.device != self.device:
            rgb_data = rgb_data.to(self.device, non_blocking=True)
        if brightness_data.device != self.device:
            brightness_data = brightness_data.to(self.device, non_blocking=True)
        
        # Use autocast for mixed precision if enabled and supported
        use_autocast = (self.enable_mixed_precision and 
                       (self.device.type == 'cuda' or 
                        (self.device.type == 'mps' and hasattr(torch.amp, 'autocast'))))
        
        if use_autocast:
            with torch.amp.autocast(device_type=self.device.type):
                return self._forward_impl(rgb_data, brightness_data)
        else:
            return self._forward_impl(rgb_data, brightness_data)
    
    def _forward_impl(self, rgb_data: torch.Tensor, brightness_data: torch.Tensor) -> torch.Tensor:
        """Actual forward implementation - optimized for minimal bottlenecks."""
        # Skip unnecessary contiguity checks - assume inputs are already contiguous
        # from device transfer or previous operations
        
        # Initial processing of separate pathways - parallel execution
        color = self.initial_color(rgb_data)
        brightness = self.initial_brightness(brightness_data)
        
        # Process through stages with integration - optimized for GPU memory
        integrated = None
        
        for i, stage in enumerate(self.stages):
            stage_name = f'stage_{i}'
            
            # Process stage - GPU-optimized execution
            # Note: Don't pass integrated to stage processing, only use it for integration
            color, brightness, _ = stage(color, brightness, None)
            
            # Apply integration if available - optimized for GPU
            if stage_name in self.integration_modules:
                # Create integration from current stage outputs only
                integrated = self.integration_modules[stage_name](
                    color, brightness, None
                )
        
        # Fused global pooling and flattening for better GPU utilization
        batch_size = color.size(0)
        color_pooled = self.global_pool(color).view(batch_size, -1)
        brightness_pooled = self.global_pool(brightness).view(batch_size, -1)
        
        if integrated is not None:
            integrated_pooled = self.global_pool(integrated).view(batch_size, -1)
        else:
            integrated_pooled = None
        
        # Final integration - GPU-optimized
        final_features = self.final_integration(color_pooled, brightness_pooled, integrated_pooled)
        
        # Classification
        output = self.classifier(final_features)
        
        return output
    
    def get_integration_weights(self) -> dict:
        """Get current integration weights for analysis."""
        weights = {}
        
        for name, module in self.integration_modules.items():
            weights[name] = module.get_weights()
        
        weights['final'] = self.final_integration.get_weights()
        
        return weights
    
    def get_pathway_contributions(self, rgb_data: torch.Tensor, brightness_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the contribution of each pathway to the final output."""
        # Initial processing
        color = self.initial_color(rgb_data)
        brightness = self.initial_brightness(brightness_data)
        
        # Track contributions
        color_contribution = torch.zeros_like(color)
        brightness_contribution = torch.zeros_like(brightness)
        
        integrated = None
        
        for i, stage in enumerate(self.stages):
            stage_name = f'stage_{i}'
            
            # Process stage
            color, brightness, stage_integrated = stage(color, brightness, integrated)
            
            # Apply integration if available
            if stage_name in self.integration_modules:
                integrated = self.integration_modules[stage_name](
                    color, brightness, stage_integrated
                )
                
                # Track contributions based on integration weights
                weights = self.integration_modules[stage_name].get_weights()
                color_contribution += color * weights['color']
                brightness_contribution += brightness * weights['brightness']
        
        return color_contribution, brightness_contribution
    
    def to_device(self, device: Optional[torch.device] = None):
        """Move model to specified device with optimizations. If device is None, auto-detects best GPU."""
        if device is None:
            device = self._detect_optimal_device()
            print(f"🔍 Auto-detected optimal device: {device}")
        
        self.device = device
        
        # Move the model to the device
        self.to(device)
        
        return self
    
    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing to save memory during training."""
        for stage in self.stages:
            if hasattr(stage, 'gradient_checkpointing'):
                stage.gradient_checkpointing = enable
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage statistics."""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved(self.device) / 1024**3,      # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            }
        elif self.device.type == 'mps':
            return {
                'allocated': torch.mps.current_allocated_memory() / 1024**3,  # GB
                'driver_allocated': torch.mps.driver_allocated_memory() / 1024**3  # GB
            }
        else:
            return {'device': 'cpu', 'memory_tracking': 'not_available'}
    
    def optimize_for_inference(self):
        """Optimize model specifically for inference."""
        self.eval()
        
        # Disable dropout and batch norm updates
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
        
        # Enable inference-specific optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        return self
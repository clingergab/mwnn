"""Model export utilities for Multi-Weight Neural Networks."""

import torch
import torch.onnx
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def export_to_onnx(model: torch.nn.Module,
                   input_shape: Tuple[int, ...],
                   output_path: str,
                   opset_version: int = 11,
                   dynamic_axes: Optional[dict] = None,
                   verbose: bool = False) -> None:
    """Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        input_shape: Shape of input tensor (including batch dimension)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version to use
        dynamic_axes: Dynamic axes specification for variable input sizes
        verbose: Whether to print verbose output during export
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )
        logger.info(f"Model exported to ONNX format: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise


def export_to_torchscript(model: torch.nn.Module,
                          output_path: str,
                          example_input: Optional[torch.Tensor] = None,
                          method: str = 'trace') -> None:
    """Export model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        example_input: Example input for tracing (required if method='trace')
        method: Export method ('trace' or 'script')
    """
    model.eval()
    
    try:
        if method == 'trace':
            if example_input is None:
                raise ValueError("example_input is required for tracing method")
            
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
            traced_model.save(output_path)
            
        elif method == 'script':
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            
        else:
            raise ValueError(f"Unknown export method: {method}")
            
        logger.info(f"Model exported to TorchScript format: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to export model to TorchScript: {e}")
        raise


def optimize_model_for_inference(model: torch.nn.Module,
                                example_input: torch.Tensor) -> torch.nn.Module:
    """Optimize model for inference using TorchScript optimizations.
    
    Args:
        model: PyTorch model to optimize
        example_input: Example input tensor for optimization
        
    Returns:
        Optimized TorchScript model
    """
    model.eval()
    
    with torch.no_grad():
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Apply optimizations
        optimized_model = torch.jit.optimize_for_inference(traced_model)
        
    logger.info("Model optimized for inference")
    return optimized_model

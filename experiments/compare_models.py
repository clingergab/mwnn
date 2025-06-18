"""Model comparison experiments for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Callable
import time
import numpy as np
from pathlib import Path
import json
import logging

from src.training.metrics import (
    calculate_accuracy,
    evaluate_robustness,
    calculate_feature_diversity,
    evaluate_color_brightness_specialization
)
from src.training.trainer import Trainer


logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Analyzer for Multi-Weight Neural Networks."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_pathway_features(self, inputs: torch.Tensor):
        """Get features from different pathways."""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'get_pathway_outputs'):
                return self.model.get_pathway_outputs(inputs.to(self.device))
            else:
                # For models without separate pathways
                output = self.model(inputs.to(self.device))
                return output, output
    
    def visualize_attention(self, inputs: torch.Tensor):
        """Visualize attention weights for attention-based models."""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'get_attention_weights'):
                return self.model.get_attention_weights(inputs.to(self.device))
            else:
                logger.warning("Model does not support attention visualization")
                return None
    
    def track_integration_weights(self, data_loader: DataLoader, num_batches: int = 100):
        """Track integration weights over time for integration-based models."""
        if not hasattr(self.model, 'integration_weights'):
            logger.warning("Model does not have integration weights")
            return None
        
        self.model.eval()
        weights_history = []
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                # Forward pass to update any adaptive weights
                _ = self.model(inputs.to(self.device))
                
                # Record current integration weights
                if hasattr(self.model, 'integration_weights'):
                    weights = self.model.integration_weights.clone().cpu().numpy()
                    weights_history.append(weights)
        
        return np.array(weights_history)


def compare_models(models: Dict[str, nn.Module],
                   test_loader: DataLoader,
                   metrics: List[str] = ['accuracy', 'robustness', 'efficiency'],
                   device: torch.device = None) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on various metrics.
    
    Args:
        models: Dictionary of model_name -> model pairs
        test_loader: Test data loader
        metrics: List of metrics to evaluate
        device: Device to run evaluation on
        
    Returns:
        Dictionary of results for each model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        model.to(device)
        model.eval()
        
        model_results = {}
        
        # Basic accuracy
        if 'accuracy' in metrics:
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()
                    total_samples += targets.size(0)
            
            accuracy = 100. * total_correct / total_samples
            model_results['accuracy'] = accuracy
        
        # Robustness evaluation
        if 'robustness' in metrics:
            robustness_results = evaluate_robustness(
                model, test_loader, device,
                perturbation_types=['brightness', 'color', 'noise']
            )
            model_results.update(robustness_results)
        
        # Efficiency metrics
        if 'efficiency' in metrics:
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_results['total_parameters'] = total_params
            model_results['trainable_parameters'] = trainable_params
            
            # Inference time
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Measure inference time
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(100):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / 100 * 1000  # ms
            model_results['inference_time_ms'] = avg_inference_time
        
        # Specialization metrics (for multi-pathway models)
        if hasattr(model, 'get_pathway_outputs'):
            try:
                specialization_results = evaluate_color_brightness_specialization(
                    model, test_loader, device
                )
                model_results.update(specialization_results)
            except Exception as e:
                logger.warning(f"Could not evaluate specialization for {model_name}: {e}")
        
        results[model_name] = model_results
        logger.info(f"Completed evaluation for {model_name}")
    
    return results


def benchmark_models(models: Dict[str, nn.Module],
                     test_loader: DataLoader,
                     save_path: str = None) -> Dict[str, Any]:
    """Run comprehensive benchmark of models.
    
    Args:
        models: Dictionary of model_name -> model pairs
        test_loader: Test data loader
        save_path: Optional path to save results
        
    Returns:
        Comprehensive benchmark results
    """
    logger.info("Starting comprehensive model benchmark")
    
    # Run comparison
    results = compare_models(
        models, test_loader,
        metrics=['accuracy', 'robustness', 'efficiency']
    )
    
    # Create summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models_evaluated': list(models.keys()),
        'results': results,
        'summary_stats': {}
    }
    
    # Calculate summary statistics
    if results:
        accuracy_scores = [r.get('accuracy', 0) for r in results.values()]
        param_counts = [r.get('total_parameters', 0) for r in results.values()]
        inference_times = [r.get('inference_time_ms', 0) for r in results.values()]
        
        summary['summary_stats'] = {
            'best_accuracy': max(accuracy_scores) if accuracy_scores else 0,
            'avg_accuracy': np.mean(accuracy_scores) if accuracy_scores else 0,
            'min_parameters': min(param_counts) if param_counts else 0,
            'max_parameters': max(param_counts) if param_counts else 0,
            'fastest_inference': min(inference_times) if inference_times else 0,
            'slowest_inference': max(inference_times) if inference_times else 0
        }
    
    # Save results if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Benchmark results saved to {save_path}")
    
    return summary
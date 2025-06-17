"""Metrics for evaluating Multi-Weight Neural Networks."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return 100. * correct / total


def calculate_per_class_accuracy(outputs: torch.Tensor, targets: torch.Tensor, 
                               num_classes: int) -> Dict[int, float]:
    """Calculate per-class accuracy."""
    _, predicted = outputs.max(1)
    
    per_class_accuracy = {}
    
    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            correct = (predicted[mask] == class_idx).sum().item()
            total = mask.sum().item()
            per_class_accuracy[class_idx] = 100. * correct / total
        else:
            per_class_accuracy[class_idx] = 0.0
    
    return per_class_accuracy


def calculate_pathway_statistics(model: torch.nn.Module, 
                               data_loader: torch.utils.data.DataLoader,
                               device: torch.device) -> Dict[str, Dict[str, float]]:
    """Calculate statistics about pathway activations."""
    model.eval()
    
    color_activations = []
    brightness_activations = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            
            # Get pathway outputs if model supports it
            if hasattr(model, 'get_pathway_outputs'):
                color_out, brightness_out = model.get_pathway_outputs(inputs)
                
                color_activations.append(color_out.cpu())
                brightness_activations.append(brightness_out.cpu())
    
    if not color_activations:
        return {}
    
    # Concatenate all activations
    color_acts = torch.cat(color_activations, dim=0)
    brightness_acts = torch.cat(brightness_activations, dim=0)
    
    # Calculate statistics
    stats = {
        'color': {
            'mean': color_acts.mean().item(),
            'std': color_acts.std().item(),
            'min': color_acts.min().item(),
            'max': color_acts.max().item(),
            'sparsity': (color_acts == 0).float().mean().item()
        },
        'brightness': {
            'mean': brightness_acts.mean().item(),
            'std': brightness_acts.std().item(),
            'min': brightness_acts.min().item(),
            'max': brightness_acts.max().item(),
            'sparsity': (brightness_acts == 0).float().mean().item()
        }
    }
    
    # Calculate correlation between pathways
    color_flat = color_acts.flatten()
    brightness_flat = brightness_acts.flatten()
    correlation = torch.corrcoef(torch.stack([color_flat, brightness_flat]))[0, 1].item()
    
    stats['pathway_correlation'] = correlation
    
    return stats


def evaluate_robustness(model: torch.nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       device: torch.device,
                       perturbation_types: List[str] = ['brightness', 'color', 'noise']) -> Dict[str, float]:
    """Evaluate model robustness to various perturbations."""
    model.eval()
    
    results = {}
    
    for perturb_type in perturbation_types:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Apply perturbation
                perturbed_inputs = apply_perturbation(inputs, perturb_type)
                
                # Get predictions
                outputs = model(perturbed_inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        results[f'{perturb_type}_robustness'] = accuracy
    
    return results


def apply_perturbation(inputs: torch.Tensor, perturbation_type: str) -> torch.Tensor:
    """Apply specific perturbation to inputs."""
    if perturbation_type == 'brightness':
        # Random brightness change
        factor = torch.rand(1).item() * 0.6 + 0.7  # 0.7 to 1.3
        return torch.clamp(inputs * factor, 0, 1)
    
    elif perturbation_type == 'color':
        # Random color shift
        shift = torch.randn(1, 3, 1, 1).to(inputs.device) * 0.1
        return torch.clamp(inputs + shift, 0, 1)
    
    elif perturbation_type == 'noise':
        # Gaussian noise
        noise = torch.randn_like(inputs) * 0.1
        return torch.clamp(inputs + noise, 0, 1)
    
    else:
        return inputs


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_integration_entropy(integration_weights: Dict[str, float]) -> float:
    """Calculate entropy of integration weights to measure uncertainty."""
    weights = torch.tensor(list(integration_weights.values()))
    weights = weights / weights.sum()  # Normalize
    
    # Calculate entropy
    entropy = -torch.sum(weights * torch.log(weights + 1e-8))
    
    return entropy.item()


def pathway_contribution_analysis(model: torch.nn.Module,
                                data_loader: torch.utils.data.DataLoader,
                                device: torch.device) -> Dict[str, np.ndarray]:
    """Analyze the contribution of each pathway to final predictions."""
    model.eval()
    
    color_contributions = []
    brightness_contributions = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            
            if hasattr(model, 'get_pathway_contributions'):
                color_contrib, brightness_contrib = model.get_pathway_contributions(inputs)
                color_contributions.append(color_contrib.cpu().numpy())
                brightness_contributions.append(brightness_contrib.cpu().numpy())
    
    if not color_contributions:
        return {}
    
    return {
        'color_contributions': np.concatenate(color_contributions),
        'brightness_contributions': np.concatenate(brightness_contributions)
    }


class MetricsTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, metric_name: str, value: float, step: Optional[int] = None):
        """Update a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.history[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        if step is not None:
            self.history[metric_name].append((step, value))
    
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric over last n updates."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = self.metrics[metric_name]
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values) if values else 0.0
    
    def reset(self, metric_name: Optional[str] = None):
        """Reset metrics."""
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name] = []
        else:
            self.metrics = {k: [] for k in self.metrics}
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages of all tracked metrics."""
        return {name: self.get_average(name) for name in self.metrics}


def calculate_feature_diversity(features: torch.Tensor) -> float:
    """Calculate diversity of features using average pairwise distance."""
    # Normalize features
    features_norm = F.normalize(features, dim=1)
    
    # Calculate pairwise distances
    distances = torch.cdist(features_norm, features_norm)
    
    # Get upper triangular part (excluding diagonal)
    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
    pairwise_distances = distances[mask]
    
    # Average distance
    diversity = pairwise_distances.mean().item()
    
    return diversity


def evaluate_color_brightness_specialization(model: torch.nn.Module,
                                           data_loader: torch.utils.data.DataLoader,
                                           device: torch.device) -> Dict[str, float]:
    """Evaluate how well pathways specialize in color vs brightness."""
    model.eval()
    
    results = {
        'color_pathway_color_sensitivity': 0.0,
        'color_pathway_brightness_sensitivity': 0.0,
        'brightness_pathway_color_sensitivity': 0.0,
        'brightness_pathway_brightness_sensitivity': 0.0,
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            if hasattr(model, 'get_pathway_outputs'):
                # Get baseline outputs
                color_base, brightness_base = model.get_pathway_outputs(inputs)
                
                # Test color perturbation
                color_perturbed = apply_perturbation(inputs, 'color')
                color_pert_c, brightness_pert_c = model.get_pathway_outputs(color_perturbed)
                
                # Test brightness perturbation
                brightness_perturbed = apply_perturbation(inputs, 'brightness')
                color_pert_b, brightness_pert_b = model.get_pathway_outputs(brightness_perturbed)
                
                # Calculate sensitivities
                results['color_pathway_color_sensitivity'] += torch.norm(
                    color_pert_c - color_base
                ).item() / batch_size
                
                results['color_pathway_brightness_sensitivity'] += torch.norm(
                    color_pert_b - color_base
                ).item() / batch_size
                
                results['brightness_pathway_color_sensitivity'] += torch.norm(
                    brightness_pert_c - brightness_base
                ).item() / batch_size
                
                results['brightness_pathway_brightness_sensitivity'] += torch.norm(
                    brightness_pert_b - brightness_base
                ).item() / batch_size
                
                num_batches += 1
    
    # Average over batches
    if num_batches > 0:
        for key in results:
            results[key] /= num_batches
    
    # Calculate specialization scores
    color_specialization = (
        results['color_pathway_color_sensitivity'] / 
        (results['color_pathway_brightness_sensitivity'] + 1e-8)
    )
    
    brightness_specialization = (
        results['brightness_pathway_brightness_sensitivity'] / 
        (results['brightness_pathway_color_sensitivity'] + 1e-8)
    )
    
    results['color_specialization_ratio'] = color_specialization
    results['brightness_specialization_ratio'] = brightness_specialization
    
    return results
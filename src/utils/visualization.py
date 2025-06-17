"""Visualization utilities for Multi-Weight Neural Networks."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_pathway_activations(color_activations: torch.Tensor,
                           brightness_activations: torch.Tensor,
                           save_path: Optional[str] = None):
    """Plot color and brightness pathway activations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color pathway
    color_act = color_activations.detach().cpu().numpy()
    axes[0].imshow(color_act.mean(axis=0), cmap='viridis')
    axes[0].set_title('Color Pathway Activations')
    axes[0].axis('off')
    
    # Brightness pathway
    brightness_act = brightness_activations.detach().cpu().numpy()
    axes[1].imshow(brightness_act.mean(axis=0), cmap='viridis')
    axes[1].set_title('Brightness Pathway Activations')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
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


def plot_attention_maps(attention_weights: Dict[str, torch.Tensor],
                       save_path: Optional[str] = None):
    """Plot attention maps from cross-modal attention."""
    num_stages = len(attention_weights)
    fig, axes = plt.subplots(2, num_stages, figsize=(4*num_stages, 8))
    
    if num_stages == 1:
        axes = axes.reshape(2, 1)
    
    for i, (stage_name, weights) in enumerate(attention_weights.items()):
        if 'color_to_brightness' in weights:
            # Color to brightness attention
            attn_map = weights['color_to_brightness'].detach().cpu().numpy()
            if len(attn_map.shape) > 2:
                attn_map = attn_map.mean(axis=0)
            
            axes[0, i].imshow(attn_map, cmap='hot')
            axes[0, i].set_title(f'{stage_name}\nColor→Brightness')
            axes[0, i].axis('off')
            
            # Brightness to color attention
            attn_map = weights['brightness_to_color'].detach().cpu().numpy()
            if len(attn_map.shape) > 2:
                attn_map = attn_map.mean(axis=0)
            
            axes[1, i].imshow(attn_map, cmap='hot')
            axes[1, i].set_title(f'Brightness→Color')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_integration_weights(integration_history: List[Dict[str, float]],
                           save_path: Optional[str] = None):
    """Plot the evolution of integration weights over time."""
    # Extract weights for each component
    epochs = list(range(len(integration_history)))
    color_weights = [w.get('color', 0) for w in integration_history]
    brightness_weights = [w.get('brightness', 0) for w in integration_history]
    integrated_weights = [w.get('integrated', 0) for w in integration_history]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, color_weights, 'r-', label='Color Weight', linewidth=2)
    plt.plot(epochs, brightness_weights, 'b-', label='Brightness Weight', linewidth=2)
    
    if any(integrated_weights):
        plt.plot(epochs, integrated_weights, 'g-', label='Integrated Weight', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title('Integration Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_model_diagram(model_name: str, save_path: Optional[str] = None):
    """Create a simple diagram of the model architecture."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define components based on model type
    if model_name == 'MultiChannelModel':
        components = [
            ('Input Image', 0, 0.5),
            ('Feature\nExtraction', 1, 0.5),
            ('Color\nPathway', 2, 0.7),
            ('Brightness\nPathway', 2, 0.3),
            ('Fusion', 3, 0.5),
            ('Classifier', 4, 0.5),
            ('Output', 5, 0.5)
        ]
        connections = [
            (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6)
        ]
    elif model_name == 'ContinuousIntegrationModel':
        components = [
            ('Input Image', 0, 0.5),
            ('Feature\nExtraction', 1, 0.5),
            ('Stage 1\nColor', 2, 0.7),
            ('Stage 1\nBrightness', 2, 0.3),
            ('Integration 1', 2.5, 0.5),
            ('Stage 2\nColor', 3, 0.7),
            ('Stage 2\nBrightness', 3, 0.3),
            ('Integration 2', 3.5, 0.5),
            ('Final\nIntegration', 4, 0.5),
            ('Output', 5, 0.5)
        ]
        connections = [
            (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6),
            (5, 7), (6, 7), (7, 8), (8, 9)
        ]
    else:
        # Generic diagram
        components = [
            ('Input', 0, 0.5),
            ('Processing', 1, 0.5),
            ('Output', 2, 0.5)
        ]
        connections = [(0, 1), (1, 2)]
    
    # Draw components
    for i, (name, x, y) in enumerate(components):
        if 'Color' in name:
            color = 'lightcoral'
        elif 'Brightness' in name:
            color = 'lightblue'
        elif 'Integration' in name or 'Fusion' in name:
            color = 'lightgreen'
        else:
            color = 'lightgray'
        
        circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw connections
    for start, end in connections:
        start_comp = components[start]
        end_comp = components[end]
        ax.arrow(start_comp[1] + 0.15, start_comp[2], 
                end_comp[1] - start_comp[1] - 0.3, 
                end_comp[2] - start_comp[2],
                head_width=0.05, head_length=0.05, fc='black', ec='black')
    
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{model_name} Architecture', fontsize=16, weight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None):
    """Plot training history with loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_weight_specialization(model: torch.nn.Module,
                                  save_path: Optional[str] = None):
    """Visualize weight specialization in the model."""
    color_weights = []
    brightness_weights = []
    
    for name, param in model.named_parameters():
        if 'color_weights' in name:
            color_weights.append(param.detach().cpu().numpy().flatten())
        elif 'brightness_weights' in name:
            brightness_weights.append(param.detach().cpu().numpy().flatten())
    
    if not color_weights or not brightness_weights:
        print("No specialized weights found in model")
        return None
    
    # Concatenate all weights
    all_color = np.concatenate(color_weights)
    all_brightness = np.concatenate(brightness_weights)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Weight distributions
    axes[0, 0].hist(all_color, bins=50, alpha=0.7, color='red', label='Color')
    axes[0, 0].hist(all_brightness, bins=50, alpha=0.7, color='blue', label='Brightness')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Weight Distributions')
    axes[0, 0].legend()
    
    # Weight magnitudes
    axes[0, 1].bar(['Color', 'Brightness'], 
                   [np.mean(np.abs(all_color)), np.mean(np.abs(all_brightness))],
                   color=['red', 'blue'])
    axes[0, 1].set_ylabel('Mean Absolute Weight')
    axes[0, 1].set_title('Average Weight Magnitudes')
    
    # Weight correlation
    min_len = min(len(all_color), len(all_brightness))
    axes[1, 0].scatter(all_color[:min_len], all_brightness[:min_len], alpha=0.1)
    axes[1, 0].set_xlabel('Color Weights')
    axes[1, 0].set_ylabel('Brightness Weights')
    axes[1, 0].set_title('Weight Correlation')
    
    # Sparsity comparison
    color_sparsity = np.mean(np.abs(all_color) < 0.01)
    brightness_sparsity = np.mean(np.abs(all_brightness) < 0.01)
    axes[1, 1].bar(['Color', 'Brightness'], 
                   [color_sparsity, brightness_sparsity],
                   color=['red', 'blue'])
    axes[1, 1].set_ylabel('Sparsity (% near zero)')
    axes[1, 1].set_title('Weight Sparsity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
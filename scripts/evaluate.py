#!/usr/bin/env python3
"""Evaluation script for Multi-Weight Neural Network models."""

import argparse
import logging
import os
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    MultiChannelModel,
    ContinuousIntegrationModel,
    CrossModalModel,
    SingleOutputModel,
    AttentionBasedModel
)
from src.training.metrics import (
    calculate_accuracy,
    calculate_per_class_accuracy,
    calculate_pathway_statistics,
    evaluate_robustness,
    evaluate_color_brightness_specialization
)
from src.utils import load_config
from src.utils.visualization import (
    plot_attention_maps,
    visualize_weight_specialization,
    plot_pathway_activations,
    plot_confusion_matrix
)


def load_model(model_type: str, checkpoint_path: str, config: dict, device: torch.device):
    """Load a trained model from checkpoint."""
    model_classes = {
        'multi_channel': MultiChannelModel,
        'continuous_integration': ContinuousIntegrationModel,
        'cross_modal': CrossModalModel,
        'single_output': SingleOutputModel,
        'attention_based': AttentionBasedModel
    }
    
    # Create model
    model_class = model_classes[model_type]
    model = model_class(**config['model']['params'])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def create_data_loader(config: dict):
    """Create test data loader."""
    data_config = config.get('data', {})
    
    # Validation transform (no augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Load dataset
    dataset_name = data_config.get('dataset', 'CIFAR10')
    data_dir = data_config.get('data_dir', './data')
    
    if dataset_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        class_names = test_dataset.classes
    elif dataset_name == 'CIFAR100':
        test_dataset = datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        class_names = [f'class_{i}' for i in range(100)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('evaluation', {}).get('test_batch_size', 64),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    return test_loader, class_names


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model performance."""
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    
    # Calculate metrics
    accuracy = calculate_accuracy(all_outputs, torch.tensor(all_targets))
    per_class_acc = calculate_per_class_accuracy(
        all_outputs, torch.tensor(all_targets), len(class_names)
    )
    
    # Classification report
    report = classification_report(
        all_targets, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets
    }


def evaluate_model_specific_metrics(model, test_loader, device, model_type):
    """Evaluate model-specific metrics."""
    results = {}
    
    if model_type == 'multi_channel' or model_type == 'cross_modal':
        # Pathway statistics
        stats = calculate_pathway_statistics(model, test_loader, device)
        results['pathway_statistics'] = stats
        
    if model_type == 'continuous_integration':
        # Integration weights
        if hasattr(model, 'get_integration_weights'):
            results['integration_weights'] = model.get_integration_weights()
    
    if model_type == 'attention_based':
        # Get attention maps for a sample batch
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)[:8]  # Just first 8 samples
                if hasattr(model, 'get_attention_maps'):
                    results['attention_maps'] = model.get_attention_maps(inputs)
                break
    
    # Evaluate specialization for all models
    specialization = evaluate_color_brightness_specialization(
        model, test_loader, device
    )
    results['specialization_scores'] = specialization
    
    return results


def evaluate_robustness_suite(model, test_loader, device):
    """Evaluate model robustness to various perturbations."""
    perturbations = ['brightness', 'color', 'noise']
    robustness_results = evaluate_robustness(
        model, test_loader, device, perturbation_types=perturbations
    )
    return robustness_results


def save_results(results, output_dir, model_name):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics = {
        'accuracy': results['accuracy'],
        'per_class_accuracy': results['per_class_accuracy'],
        'classification_report': results['classification_report']
    }
    
    with open(output_dir / f'{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix plot
    plot_confusion_matrix(
        results['targets'], 
        results['predictions'],
        class_names=results.get('class_names', None),
        save_path=output_dir / f'{model_name}_confusion_matrix.png'
    )
    
    # Save model-specific visualizations
    if 'pathway_statistics' in results:
        with open(output_dir / f'{model_name}_pathway_stats.json', 'w') as f:
            json.dump(results['pathway_statistics'], f, indent=2)
    
    if 'integration_weights' in results:
        with open(output_dir / f'{model_name}_integration_weights.json', 'w') as f:
            json.dump(results['integration_weights'], f, indent=2)
    
    if 'attention_maps' in results and results['attention_maps']:
        plot_attention_maps(
            results['attention_maps'],
            save_path=output_dir / f'{model_name}_attention_maps.png'
        )
    
    if 'robustness_results' in results:
        with open(output_dir / f'{model_name}_robustness.json', 'w') as f:
            json.dump(results['robustness_results'], f, indent=2)
    
    print(f"Results saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Weight Neural Network')
    parser.add_argument('model', type=str, 
                       choices=['multi_channel', 'continuous_integration', 
                               'cross_modal', 'single_output', 'attention_based'],
                       help='Model type to evaluate')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['CIFAR10', 'CIFAR100'],
                       help='Dataset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--robustness', action='store_true',
                       help='Evaluate robustness to perturbations')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Use default config
            config = {
                'model': {'params': {'num_classes': 10}},
                'data': {'dataset': 'CIFAR10'},
                'evaluation': {'test_batch_size': args.batch_size}
            }
    
    # Override with command line arguments
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.batch_size:
        config['evaluation']['test_batch_size'] = args.batch_size
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, checkpoint_data = load_model(args.model, args.checkpoint, config, device)
    
    # Create data loader
    logger.info("Creating data loader...")
    test_loader, class_names = create_data_loader(config)
    
    # Basic evaluation
    logger.info("Evaluating model performance...")
    results = evaluate_model(model, test_loader, device, class_names)
    results['class_names'] = class_names
    
    logger.info(f"Test Accuracy: {results['accuracy']:.2f}%")
    
    # Model-specific metrics
    logger.info("Evaluating model-specific metrics...")
    specific_results = evaluate_model_specific_metrics(
        model, test_loader, device, args.model
    )
    results.update(specific_results)
    
    # Robustness evaluation
    if args.robustness:
        logger.info("Evaluating robustness...")
        robustness_results = evaluate_robustness_suite(model, test_loader, device)
        results['robustness_results'] = robustness_results
        
        for perturb, acc in robustness_results.items():
            logger.info(f"{perturb}: {acc:.2f}%")
    
    # Visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Weight specialization
        if args.model != 'attention_based':  # Skip for attention model
            fig = visualize_weight_specialization(
                model, 
                save_path=output_dir / f'{args.model}_weight_specialization.png'
            )
        
        # Model-specific visualizations
        if hasattr(model, 'get_pathway_outputs'):
            # Get sample pathway outputs
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)[:1]
                    color, brightness = model.get_pathway_outputs(inputs)
                    plot_pathway_activations(
                        color, brightness,
                        save_path=output_dir / f'{args.model}_pathway_activations.png'
                    )
                    break
    
    # Save results
    logger.info("Saving results...")
    save_results(results, args.output_dir, args.model)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    
    if 'pathway_statistics' in results:
        stats = results['pathway_statistics']
        if 'pathway_correlation' in stats:
            print(f"Pathway Correlation: {stats['pathway_correlation']:.3f}")
    
    if 'specialization_scores' in results:
        scores = results['specialization_scores']
        if 'color_specialization_ratio' in scores:
            print(f"Color Specialization: {scores['color_specialization_ratio']:.3f}")
            print(f"Brightness Specialization: {scores['brightness_specialization_ratio']:.3f}")
    
    print("="*50)


if __name__ == '__main__':
    main()
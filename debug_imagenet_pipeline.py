#!/usr/bin/env python3
"""
ImageNet Pipeline Debugging Script
Comprehensive analysis of why MWNN fails on ImageNet but succeeds on MNIST
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path

sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel
from test_mnist_csv import MNISTCSVDataset


def analyze_model_architecture(model, model_name):
    """Analyze model architecture and parameters"""
    print(f"\n🔍 Analyzing {model_name} Architecture:")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # Layer analysis
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_count += 1
    
    print(f"   Total Layers (Conv + Linear): {layer_count}")
    
    # Memory analysis
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"   Model Size: {model_size_mb:.2f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_count': layer_count,
        'model_size_mb': model_size_mb
    }


def test_gradient_flow(model, sample_input, sample_target, loss_fn):
    """Test gradient flow through the model"""
    print(f"\n🔍 Testing Gradient Flow:")
    
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)
    
    model.train()
    
    # Forward pass
    try:
        output = model(sample_input)
        loss = loss_fn(output, sample_target)
        print(f"   Forward Pass: ✅ Loss = {loss.item():.4f}")
    except Exception as e:
        print(f"   Forward Pass: ❌ Error = {e}")
        return None
    
    # Backward pass
    try:
        loss.backward()
        print(f"   Backward Pass: ✅")
    except Exception as e:
        print(f"   Backward Pass: ❌ Error = {e}")
        return None
    
    # Analyze gradients
    grad_stats = {}
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_norm += grad_norm ** 2
            param_count += 1
            
            if grad_norm == 0:
                zero_grad_count += 1
            
            grad_stats[name] = {
                'grad_norm': grad_norm,
                'param_shape': list(param.shape),
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item()
            }
    
    total_norm = total_norm ** 0.5
    
    print(f"   Total Gradient Norm: {total_norm:.6f}")
    print(f"   Parameters with Gradients: {param_count}")
    print(f"   Parameters with Zero Gradients: {zero_grad_count}")
    
    # Check for problematic gradients
    nan_grads = [name for name, stats in grad_stats.items() if stats['has_nan']]
    inf_grads = [name for name, stats in grad_stats.items() if stats['has_inf']]
    
    if nan_grads:
        print(f"   ⚠️  NaN Gradients in: {nan_grads}")
    if inf_grads:
        print(f"   ⚠️  Inf Gradients in: {inf_grads}")
    
    return {
        'total_gradient_norm': total_norm,
        'param_count': param_count,
        'zero_grad_count': zero_grad_count,
        'nan_gradients': nan_grads,
        'inf_gradients': inf_grads,
        'gradient_stats': grad_stats
    }


def test_data_preprocessing_pipeline():
    """Test and compare data preprocessing between MNIST and ImageNet-like data"""
    print(f"\n🔍 Testing Data Preprocessing:")
    
    # Load MNIST sample
    try:
        mnist_dataset = MNISTCSVDataset('data/MNIST/mnist_train.csv', max_samples=10)
        mnist_sample = mnist_dataset[0]
        print(f"   MNIST Sample Shape: {mnist_sample[0].shape}")
        print(f"   MNIST Sample Range: [{mnist_sample[0].min():.3f}, {mnist_sample[0].max():.3f}]")
        print(f"   MNIST Sample Mean: {mnist_sample[0].mean():.3f}")
        print(f"   MNIST Sample Std: {mnist_sample[0].std():.3f}")
    except Exception as e:
        print(f"   ❌ MNIST Loading Error: {e}")
        return None
    
    # Simulate ImageNet-like preprocessing
    try:
        # Create synthetic ImageNet-like data
        imagenet_sample = torch.randn(3, 224, 224)
        # Apply ImageNet normalization
        imagenet_normalized = (imagenet_sample - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        print(f"   ImageNet Sample Shape: {imagenet_normalized.shape}")
        print(f"   ImageNet Sample Range: [{imagenet_normalized.min():.3f}, {imagenet_normalized.max():.3f}]")
        print(f"   ImageNet Sample Mean: {imagenet_normalized.mean():.3f}")
        print(f"   ImageNet Sample Std: {imagenet_normalized.std():.3f}")
        
        # Test extreme values
        if imagenet_normalized.min() < -10 or imagenet_normalized.max() > 10:
            print(f"   ⚠️  Extreme normalization values detected!")
            
    except Exception as e:
        print(f"   ❌ ImageNet Preprocessing Error: {e}")
        return None
    
    return {
        'mnist_shape': list(mnist_sample[0].shape),
        'mnist_range': [float(mnist_sample[0].min()), float(mnist_sample[0].max())],
        'mnist_stats': {'mean': float(mnist_sample[0].mean()), 'std': float(mnist_sample[0].std())},
        'imagenet_shape': list(imagenet_normalized.shape),
        'imagenet_range': [float(imagenet_normalized.min()), float(imagenet_normalized.max())],
        'imagenet_stats': {'mean': float(imagenet_normalized.mean()), 'std': float(imagenet_normalized.std())}
    }


def test_model_capacity():
    """Test if model has sufficient capacity for the task"""
    print(f"\n🔍 Testing Model Capacity:")
    
    # Test overfitting on small dataset
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    # MNIST-like test (should easily overfit)
    print("   Testing on MNIST-like data (10 classes):")
    mnist_model = ContinuousIntegrationModel(num_classes=10, depth='shallow')
    mnist_model = mnist_model.to(device)
    
    # Create small synthetic dataset
    small_dataset = []
    for i in range(50):  # 50 samples, 5 per class
        label = i % 10
        data = torch.randn(3, 28, 28) * 0.1 + label * 0.1  # Slight pattern
        small_dataset.append((data, label))
    
    # Train for overfitting
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mnist_model.parameters(), lr=0.01)
    
    mnist_model.train()
    initial_loss = None
    final_loss = None
    
    for epoch in range(50):  # Many epochs to test overfitting
        total_loss = 0
        for data, label in small_dataset:
            data = data.unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)
            
            optimizer.zero_grad()
            output = mnist_model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(small_dataset)
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 49:
            final_loss = avg_loss
        
        if epoch % 10 == 9:
            print(f"      Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    mnist_overfitting = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
    print(f"   MNIST Overfitting Ratio: {mnist_overfitting:.3f}")
    
    # ImageNet-like test (1000 classes)
    print("   Testing on ImageNet-like data (1000 classes):")
    imagenet_model = ContinuousIntegrationModel(num_classes=1000, depth='medium')
    imagenet_model = imagenet_model.to(device)
    
    # Create small ImageNet-like dataset
    imagenet_dataset = []
    for i in range(100):  # 100 samples across 1000 classes
        label = i % 1000
        data = torch.randn(3, 224, 224) * 0.1 + (label / 1000.0) * 0.1
        imagenet_dataset.append((data, label))
    
    optimizer2 = torch.optim.Adam(imagenet_model.parameters(), lr=0.001)
    imagenet_model.train()
    
    initial_loss2 = None
    final_loss2 = None
    
    for epoch in range(20):  # Fewer epochs due to complexity
        total_loss = 0
        batch_count = 0
        for data, label in imagenet_dataset:
            data = data.unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)
            
            try:
                optimizer2.zero_grad()
                output = imagenet_model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer2.step()
                
                total_loss += loss.item()
                batch_count += 1
            except Exception as e:
                print(f"      Error in epoch {epoch}, sample {label.item()}: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            if epoch == 0:
                initial_loss2 = avg_loss
            if epoch == 19:
                final_loss2 = avg_loss
            
            if epoch % 5 == 4:
                print(f"      Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    imagenet_overfitting = (initial_loss2 - final_loss2) / initial_loss2 if initial_loss2 and initial_loss2 > 0 else 0
    print(f"   ImageNet Overfitting Ratio: {imagenet_overfitting:.3f}")
    
    return {
        'mnist_overfitting_ratio': mnist_overfitting,
        'imagenet_overfitting_ratio': imagenet_overfitting,
        'mnist_loss_reduction': initial_loss - final_loss if initial_loss and final_loss else 0,
        'imagenet_loss_reduction': initial_loss2 - final_loss2 if initial_loss2 and final_loss2 else 0
    }


def analyze_training_dynamics():
    """Analyze training dynamics and optimization landscape"""
    print(f"\n🔍 Analyzing Training Dynamics:")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with different optimizers
    optimizers_to_test = ['Adam', 'SGD', 'AdamW']
    results = {}
    
    for opt_name in optimizers_to_test:
        print(f"   Testing {opt_name} optimizer:")
        
        model = ContinuousIntegrationModel(num_classes=10, depth='shallow')
        model = model.to(device)
        
        if opt_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        elif opt_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif opt_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Quick training test
        criterion = nn.CrossEntropyLoss()
        model.train()
        
        losses = []
        for epoch in range(10):
            # Synthetic batch
            batch_data = torch.randn(32, 3, 28, 28).to(device)
            batch_labels = torch.randint(0, 10, (32,)).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        loss_reduction = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
        print(f"      Loss Reduction: {loss_reduction:.3f}")
        print(f"      Final Loss: {losses[-1]:.4f}")
        
        results[opt_name] = {
            'loss_reduction': loss_reduction,
            'final_loss': losses[-1],
            'loss_history': losses
        }
    
    return results


def main():
    """Run comprehensive ImageNet debugging analysis"""
    print("🔍 ImageNet Pipeline Debugging Analysis")
    print("Investigating why MWNN fails on ImageNet but succeeds on MNIST")
    
    debug_results = {}
    
    # Create models for testing
    try:
        mnist_model = ContinuousIntegrationModel(num_classes=10, depth='shallow')
        imagenet_model = ContinuousIntegrationModel(num_classes=1000, depth='medium')
        
        print("✅ Models created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Architecture Analysis
    print(f"\n{'='*60}")
    print("ARCHITECTURE ANALYSIS")
    print(f"{'='*60}")
    
    mnist_arch = analyze_model_architecture(mnist_model, "MNIST Model")
    imagenet_arch = analyze_model_architecture(imagenet_model, "ImageNet Model")
    
    debug_results['architecture_analysis'] = {
        'mnist': mnist_arch,
        'imagenet': imagenet_arch
    }
    
    # Gradient Flow Analysis
    print(f"\n{'='*60}")
    print("GRADIENT FLOW ANALYSIS")
    print(f"{'='*60}")
    
    # Test gradient flow
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    
    mnist_model = mnist_model.to(device)
    imagenet_model = imagenet_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # MNIST gradient test
    mnist_input = torch.randn(1, 3, 28, 28)
    mnist_target = torch.tensor([5])
    mnist_grad_stats = test_gradient_flow(mnist_model, mnist_input, mnist_target, criterion)
    
    # ImageNet gradient test
    imagenet_input = torch.randn(1, 3, 224, 224)
    imagenet_target = torch.tensor([500])
    imagenet_grad_stats = test_gradient_flow(imagenet_model, imagenet_input, imagenet_target, criterion)
    
    debug_results['gradient_analysis'] = {
        'mnist': mnist_grad_stats,
        'imagenet': imagenet_grad_stats
    }
    
    # Data Preprocessing Analysis
    print(f"\n{'='*60}")
    print("DATA PREPROCESSING ANALYSIS")
    print(f"{'='*60}")
    
    preprocessing_stats = test_data_preprocessing_pipeline()
    debug_results['preprocessing_analysis'] = preprocessing_stats
    
    # Model Capacity Analysis
    print(f"\n{'='*60}")
    print("MODEL CAPACITY ANALYSIS")
    print(f"{'='*60}")
    
    capacity_stats = test_model_capacity()
    debug_results['capacity_analysis'] = capacity_stats
    
    # Training Dynamics Analysis
    print(f"\n{'='*60}")
    print("TRAINING DYNAMICS ANALYSIS")
    print(f"{'='*60}")
    
    dynamics_stats = analyze_training_dynamics()
    debug_results['dynamics_analysis'] = dynamics_stats
    
    # Save results
    output_file = Path('checkpoints/imagenet_debug_results.json')
    with open(output_file, 'w') as f:
        json.dump(debug_results, f, indent=2, default=str)
    
    # Generate Summary Report
    print(f"\n{'='*60}")
    print("🎯 DEBUGGING SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Parameter comparison
    mnist_params = debug_results['architecture_analysis']['mnist']['total_params']
    imagenet_params = debug_results['architecture_analysis']['imagenet']['total_params']
    param_ratio = imagenet_params / mnist_params
    print(f"\n📊 Parameter Analysis:")
    print(f"   MNIST Model: {mnist_params:,} parameters")
    print(f"   ImageNet Model: {imagenet_params:,} parameters")
    print(f"   Complexity Ratio: {param_ratio:.1f}x")
    
    # Gradient analysis
    if mnist_grad_stats and imagenet_grad_stats:
        print(f"\n🔄 Gradient Analysis:")
        print(f"   MNIST Gradient Norm: {mnist_grad_stats['total_gradient_norm']:.6f}")
        print(f"   ImageNet Gradient Norm: {imagenet_grad_stats['total_gradient_norm']:.6f}")
        
        if imagenet_grad_stats['nan_gradients'] or imagenet_grad_stats['inf_gradients']:
            print(f"   ⚠️  ImageNet has problematic gradients!")
    
    # Capacity analysis
    print(f"\n🧠 Model Capacity:")
    print(f"   MNIST Overfitting: {capacity_stats['mnist_overfitting_ratio']:.3f}")
    print(f"   ImageNet Overfitting: {capacity_stats['imagenet_overfitting_ratio']:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if param_ratio > 10:
        print("   1. ⚠️  ImageNet model is significantly more complex - consider simplification")
    
    if imagenet_grad_stats and imagenet_grad_stats['total_gradient_norm'] < 1e-6:
        print("   2. ⚠️  Very small gradients detected - try higher learning rate")
    elif imagenet_grad_stats and imagenet_grad_stats['total_gradient_norm'] > 100:
        print("   2. ⚠️  Very large gradients detected - try lower learning rate or gradient clipping")
    
    if capacity_stats['imagenet_overfitting_ratio'] < 0.1:
        print("   3. ⚠️  Model struggles to overfit - insufficient capacity or optimization issues")
    
    if preprocessing_stats:
        imagenet_range = preprocessing_stats['imagenet_range']
        if abs(imagenet_range[0]) > 5 or abs(imagenet_range[1]) > 5:
            print("   4. ⚠️  Extreme preprocessing values - check normalization")
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    print(f"\n🎯 Next steps: Address identified issues and retest")


if __name__ == "__main__":
    main()

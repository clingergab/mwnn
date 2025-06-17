#!/usr/bin/env python3
"""
Robustness Testing for MWNN Architecture
Tests the model's stability and robustness across different conditions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from pathlib import Path
import sys

sys.path.append('.')

from test_mnist_csv import MNISTCSVDataset
from src.models.continuous_integration.model import ContinuousIntegrationModel


def add_noise_to_data(dataloader, noise_level=0.1):
    """Add Gaussian noise to data for robustness testing"""
    noisy_data = []
    for inputs, labels in dataloader:
        noise = torch.randn_like(inputs) * noise_level
        noisy_inputs = inputs + noise
        noisy_inputs = torch.clamp(noisy_inputs, -1, 1)  # Keep in valid range
        noisy_data.append((noisy_inputs, labels))
    return noisy_data


def test_learning_rate_sensitivity(model_class, dataset, learning_rates):
    """Test model sensitivity to different learning rates"""
    results = []
    
    for lr in learning_rates:
        print(f"\n🔬 Testing Learning Rate: {lr}")
        
        # Create fresh model
        model = model_class(num_classes=10, depth='shallow')
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                             'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Quick training (3 epochs)
        model.train()
        total_loss = 0
        batch_count = 0
        
        for epoch in range(3):
            for i, (inputs, labels) in enumerate(dataset):
                if i >= 20:  # Limit to 20 batches for speed
                    break
                    
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        results.append({
            'learning_rate': lr,
            'final_loss': avg_loss,
            'converged': avg_loss < 10.0  # Simple convergence criterion
        })
        
        print(f"   Final Loss: {avg_loss:.4f}")
    
    return results


def test_noise_robustness(model, dataloader, noise_levels):
    """Test model robustness to input noise"""
    results = []
    device = next(model.parameters()).device
    
    # First get clean performance
    model.eval()
    clean_correct = 0
    clean_total = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= 10:  # Limit for speed
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            clean_total += labels.size(0)
            clean_correct += (predicted == labels).sum().item()
    
    clean_acc = 100 * clean_correct / clean_total if clean_total > 0 else 0
    
    # Test with different noise levels
    for noise_level in noise_levels:
        print(f"\n🔬 Testing Noise Level: {noise_level}")
        
        noisy_correct = 0
        noisy_total = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if i >= 10:  # Limit for speed
                    break
                    
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Add noise
                noise = torch.randn_like(inputs) * noise_level
                noisy_inputs = inputs + noise
                noisy_inputs = torch.clamp(noisy_inputs, -1, 1)
                
                outputs = model(noisy_inputs)
                _, predicted = torch.max(outputs.data, 1)
                noisy_total += labels.size(0)
                noisy_correct += (predicted == labels).sum().item()
        
        noisy_acc = 100 * noisy_correct / noisy_total if noisy_total > 0 else 0
        acc_drop = clean_acc - noisy_acc
        
        results.append({
            'noise_level': noise_level,
            'clean_accuracy': clean_acc,
            'noisy_accuracy': noisy_acc,
            'accuracy_drop': acc_drop
        })
        
        print(f"   Accuracy: {noisy_acc:.2f}% (drop: {acc_drop:.2f}%)")
    
    return results


def test_batch_size_sensitivity(model_class, dataset_fn, batch_sizes):
    """Test model sensitivity to different batch sizes"""
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n🔬 Testing Batch Size: {batch_size}")
        
        try:
            # Create data loader with specific batch size
            train_data = dataset_fn()
            train_loader = DataLoader(train_data, batch_size=batch_size, 
                                    shuffle=True, num_workers=0)
            
            # Create fresh model
            model = model_class(num_classes=10, depth='shallow')
            device = torch.device('mps' if torch.backends.mps.is_available() else 
                                 'cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Quick training
            model.train()
            total_loss = 0
            batch_count = 0
            start_time = time.time()
            
            for epoch in range(2):
                for i, (inputs, labels) in enumerate(train_loader):
                    if i >= 10:  # Limit batches
                        break
                        
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
            
            training_time = time.time() - start_time
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            results.append({
                'batch_size': batch_size,
                'final_loss': avg_loss,
                'training_time': training_time,
                'stable': avg_loss < 10.0
            })
            
            print(f"   Final Loss: {avg_loss:.4f}, Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed with batch size {batch_size}: {e}")
            results.append({
                'batch_size': batch_size,
                'final_loss': float('inf'),
                'training_time': 0,
                'stable': False,
                'error': str(e)
            })
    
    return results


def create_mnist_dataset():
    """Helper function to create MNIST dataset"""
    return MNISTCSVDataset('data/MNIST/mnist_train.csv', max_samples=1000)


def main():
    """Run comprehensive robustness testing"""
    print("🧪 Starting MWNN Robustness Testing Suite")
    print("Testing stability and robustness across different conditions")
    
    all_results = {}
    
    # Setup basic dataset
    try:
        train_dataset = create_mnist_dataset()
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        print(f"✅ Loaded MNIST dataset with {len(train_dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load MNIST dataset: {e}")
        return
    
    # Test 1: Learning Rate Sensitivity
    print(f"\n{'='*60}")
    print("TEST 1: Learning Rate Sensitivity")
    print(f"{'='*60}")
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    lr_results = test_learning_rate_sensitivity(
        ContinuousIntegrationModel, 
        train_loader, 
        learning_rates
    )
    all_results['learning_rate_sensitivity'] = lr_results
    
    # Test 2: Noise Robustness
    print(f"\n{'='*60}")
    print("TEST 2: Noise Robustness")
    print(f"{'='*60}")
    
    # Train a model first
    model = ContinuousIntegrationModel(num_classes=10, depth='shallow')
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Quick training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(3):
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= 20:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Test noise robustness
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    noise_results = test_noise_robustness(model, train_loader, noise_levels)
    all_results['noise_robustness'] = noise_results
    
    # Test 3: Batch Size Sensitivity
    print(f"\n{'='*60}")
    print("TEST 3: Batch Size Sensitivity")
    print(f"{'='*60}")
    
    batch_sizes = [16, 32, 64, 128, 256]
    batch_results = test_batch_size_sensitivity(
        ContinuousIntegrationModel,
        create_mnist_dataset,
        batch_sizes
    )
    all_results['batch_size_sensitivity'] = batch_results
    
    # Save results
    output_file = Path('checkpoints/robustness_test_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 ROBUSTNESS TESTING SUMMARY")
    print(f"{'='*60}")
    
    # Learning Rate Summary
    print("\n🔬 Learning Rate Sensitivity:")
    stable_lrs = [r for r in lr_results if r['converged']]
    if stable_lrs:
        best_lr = min(stable_lrs, key=lambda x: x['final_loss'])
        print(f"   Best LR: {best_lr['learning_rate']} (loss: {best_lr['final_loss']:.4f})")
        print(f"   Stable LRs: {[r['learning_rate'] for r in stable_lrs]}")
    else:
        print("   ❌ No stable learning rates found")
    
    # Noise Robustness Summary
    print("\n🔬 Noise Robustness:")
    if noise_results:
        for result in noise_results:
            if result['noise_level'] == 0.0:
                continue
            print(f"   Noise {result['noise_level']}: {result['noisy_accuracy']:.1f}% "
                  f"(drop: {result['accuracy_drop']:.1f}%)")
    
    # Batch Size Summary
    print("\n🔬 Batch Size Sensitivity:")
    stable_batches = [r for r in batch_results if r.get('stable', False)]
    if stable_batches:
        best_batch = min(stable_batches, key=lambda x: x['final_loss'])
        print(f"   Best Batch Size: {best_batch['batch_size']} (loss: {best_batch['final_loss']:.4f})")
        print(f"   Stable Batch Sizes: {[r['batch_size'] for r in stable_batches]}")
    else:
        print("   ❌ No stable batch sizes found")
    
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()

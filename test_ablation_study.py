#!/usr/bin/env python3
"""
Ablation Study: Test MWNN architecture with different complexities
to isolate the source of ImageNet training failures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json
import time
from pathlib import Path
import sys
sys.path.append('.')

from src.models.continuous_integration.model import ContinuousIntegrationModel


class SimplifiedMWNN(nn.Module):
    """Simplified MWNN for ablation testing"""
    
    def __init__(self, num_classes=10, use_dual_pathway=False):
        super().__init__()
        self.use_dual_pathway = use_dual_pathway
        
        # RGB pathway
        self.rgb_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.rgb_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.rgb_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Luminance pathway (if enabled)
        if use_dual_pathway:
            self.lum_conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.lum_conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.lum_pool = nn.AdaptiveAvgPool2d((4, 4))
            fc_input = 64 * 16 + 32 * 16  # RGB + Luminance
        else:
            fc_input = 64 * 16  # RGB only
            
        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Assume x is RGB image
        rgb_features = torch.relu(self.rgb_conv1(x))
        rgb_features = torch.relu(self.rgb_conv2(rgb_features))
        rgb_features = self.rgb_pool(rgb_features)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        
        if self.use_dual_pathway:
            # Convert RGB to luminance (simple method)
            luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            lum_features = torch.relu(self.lum_conv1(luminance))
            lum_features = torch.relu(self.lum_conv2(lum_features))
            lum_features = self.lum_pool(lum_features)
            lum_features = lum_features.view(lum_features.size(0), -1)
            
            # Concatenate features
            combined_features = torch.cat([rgb_features, lum_features], dim=1)
        else:
            combined_features = rgb_features
            
        return self.classifier(combined_features)


def run_ablation_experiment(name, model, dataset_name='CIFAR10', epochs=5):
    """Run a single ablation experiment"""
    print(f"\n{'='*50}")
    print(f"🧪 Running Experiment: {name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'='*50}")
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    model = model.to(device)
    
    # Load dataset
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        
    elif dataset_name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    start_time = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss = {loss.item():.4f}')
        
        train_acc = 100 * correct / total
        avg_loss = epoch_loss / len(trainloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(testloader)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        model.train()
    
    training_time = time.time() - start_time
    
    results = {
        'experiment_name': name,
        'dataset': dataset_name,
        'epochs': epochs,
        'training_time': training_time,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'history': history
    }
    
    return results


def main():
    """Run comprehensive ablation study"""
    print("🚀 Starting MWNN Ablation Study")
    print("Testing architecture complexity on progressively harder datasets")
    
    all_results = []
    
    # Experiment 1: Simple CNN (baseline) on CIFAR-10
    simple_cnn = SimplifiedMWNN(num_classes=10, use_dual_pathway=False)
    results1 = run_ablation_experiment("Simple CNN (RGB only)", simple_cnn, "CIFAR10")
    all_results.append(results1)
    
    # Experiment 2: Dual pathway CNN on CIFAR-10
    dual_cnn = SimplifiedMWNN(num_classes=10, use_dual_pathway=True)
    results2 = run_ablation_experiment("Dual Pathway CNN (RGB + Luminance)", dual_cnn, "CIFAR10")
    all_results.append(results2)
    
    # Experiment 3: Simple CNN on CIFAR-100 (more classes)
    simple_cnn_100 = SimplifiedMWNN(num_classes=100, use_dual_pathway=False)
    results3 = run_ablation_experiment("Simple CNN (RGB only)", simple_cnn_100, "CIFAR100")
    all_results.append(results3)
    
    # Experiment 4: Dual pathway CNN on CIFAR-100
    dual_cnn_100 = SimplifiedMWNN(num_classes=100, use_dual_pathway=True)
    results4 = run_ablation_experiment("Dual Pathway CNN (RGB + Luminance)", dual_cnn_100, "CIFAR100")
    all_results.append(results4)
    
    # Experiment 5: Full MWNN on CIFAR-10
    full_mwnn = ContinuousIntegrationModel(num_classes=10, depth='shallow')
    results5 = run_ablation_experiment("Full MWNN (Shallow)", full_mwnn, "CIFAR10")
    all_results.append(results5)
    
    # Save all results
    output_file = Path('checkpoints/ablation_study_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"🧪 {result['experiment_name']} ({result['dataset']})")
        print(f"   Best Val Acc: {result['best_val_acc']:.2f}%")
        print(f"   Training Time: {result['training_time']:.1f}s")
        print()
    
    print(f"📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()

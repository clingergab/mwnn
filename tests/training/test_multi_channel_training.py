#!/usr/bin/env python3
"""Test multi-channel model training setup without downloading data."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.utils.config import get_model_config
from src.models.multi_channel.model import MultiChannelModel
from src.training.trainer import Trainer

def test_multi_channel_training_setup():
    """Test multi-channel model training setup without data download."""
    print("🧪 Testing Multi-Channel Model Training Setup")
    print("=" * 55)
    
    # Load config
    print("\n1️⃣ Loading Configuration...")
    config = get_model_config('multi_channel')
    print(f"✅ Config loaded: {config['model']['name']}")
    
    # Create model
    print("\n2️⃣ Creating Model...")
    model = MultiChannelModel(**config['model']['params'])
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create mock data
    print("\n3️⃣ Creating Mock Data...")
    # Create synthetic CIFAR-10 style data
    train_data = torch.randn(100, 3, 32, 32)  # 100 samples
    train_labels = torch.randint(0, 10, (100,))  # Random labels 0-9
    
    val_data = torch.randn(20, 3, 32, 32)  # 20 validation samples
    val_labels = torch.randint(0, 10, (20,))
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"✅ Mock data created: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Test forward pass
    print("\n4️⃣ Testing Forward Pass...")
    model.eval()
    with torch.no_grad():
        batch_data, batch_labels = next(iter(train_loader))
        output = model(batch_data)
        print(f"✅ Forward pass: {batch_data.shape} -> {output.shape}")
    
    # Create trainer
    print("\n5️⃣ Creating Trainer...")
    trainer = Trainer(
        model=model,
        device='cpu',
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    print("✅ Trainer created successfully")
    
    # Test one training step
    print("\n6️⃣ Testing Training Step...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Manual training step
    batch_data, batch_labels = next(iter(train_loader))
    
    # Forward pass
    output = model(batch_data)
    loss = criterion(output, batch_labels)
    
    # Backward pass
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()
    
    print(f"✅ Training step completed. Loss: {loss.item():.4f}")
    
    # Test evaluation
    print("\n7️⃣ Testing Evaluation...")
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            
            pred = output.argmax(dim=1)
            correct = (pred == batch_labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    
    print(f"✅ Evaluation completed. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("\n🎉 All Multi-Channel Training Components Working!")
    print("✅ Configuration loading")
    print("✅ Model creation")
    print("✅ Data loading")
    print("✅ Forward pass")
    print("✅ Trainer setup")
    print("✅ Training step")
    print("✅ Evaluation")

if __name__ == "__main__":
    try:
        test_multi_channel_training_setup()
        print("\n✅ SUCCESS: Multi-channel model is ready for training!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

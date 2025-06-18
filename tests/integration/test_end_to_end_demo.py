#!/usr/bin/env python3
"""
Simple Demo Test
Demonstrate MWNN API with synthetic data for quick validation.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from mwnn import MWNN


class SyntheticDataset(Dataset):
    """Synthetic dataset for quick API demonstration."""
    
    def __init__(self, num_samples=100, num_classes=10, image_size=32):
        """
        Args:
            num_samples: Number of synthetic samples
            num_classes: Number of classes
            image_size: Size of square images
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate random data
        self.rgb_data = torch.randn(num_samples, 3, image_size, image_size)
        self.brightness_data = torch.randn(num_samples, 1, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (self.rgb_data[idx], self.brightness_data[idx]), self.labels[idx]


def create_synthetic_dataloaders(num_samples=200, num_classes=10, batch_size=16):
    """Create synthetic data loaders for testing."""
    
    # Create train and test datasets
    train_dataset = SyntheticDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=32
    )
    
    test_dataset = SyntheticDataset(
        num_samples=num_samples // 4,  # Smaller test set
        num_classes=num_classes,
        image_size=32
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def demo_basic_api():
    """Demonstrate basic MWNN API usage."""
    print("\n=== MWNN API Demo - Basic Usage ===")
    
    # Create synthetic data
    print("📊 Creating synthetic dataset...")
    train_loader, test_loader = create_synthetic_dataloaders(
        num_samples=100,
        num_classes=5,
        batch_size=8
    )
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Test data format
    data_batch, label_batch = next(iter(train_loader))
    rgb_batch, brightness_batch = data_batch
    print(f"RGB shape: {rgb_batch.shape}, Brightness shape: {brightness_batch.shape}")
    
    # Create model
    print("🧠 Creating MWNN model...")
    model = MWNN(num_classes=5, depth='shallow', base_channels=16)
    model.summary()
    
    # Train model
    print("🎯 Training model...")
    with tempfile.TemporaryDirectory() as temp_dir:
        history = model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=2,
            learning_rate=0.01,
            save_checkpoints=True,
            checkpoint_dir=temp_dir
        )
        
        print(f"Training completed! Final accuracy: {history['val_acc'][-1]:.2f}%")
        
        # Evaluate
        print("📈 Evaluating model...")
        results = model.evaluate(test_loader)
        print(f"Test accuracy: {results['accuracy']:.2f}%")
        
        # Make predictions
        print("🔮 Making predictions...")
        predictions = model.predict(test_loader)
        print(f"Predictions: {predictions[:10].tolist()}")
        
        # Save and load
        print("💾 Testing save/load...")
        model_path = Path(temp_dir) / 'demo_model.pth'
        model.save(model_path)
        
        loaded_model = MWNN.load(model_path)
        loaded_results = loaded_model.evaluate(test_loader)
        print(f"Loaded model accuracy: {loaded_results['accuracy']:.2f}%")
        
        print("✅ Basic API demo completed successfully!")


def demo_different_configurations():
    """Demonstrate different model configurations."""
    print("\n=== MWNN API Demo - Different Configurations ===")
    
    # Create shared data
    train_loader, test_loader = create_synthetic_dataloaders(
        num_samples=80,
        num_classes=3,
        batch_size=8
    )
    
    configurations = [
        {'depth': 'shallow', 'base_channels': 8, 'description': 'Lightweight'},
        {'depth': 'medium', 'base_channels': 16, 'description': 'Balanced'},
        {'depth': 'deep', 'base_channels': 24, 'description': 'Complex'}
    ]
    
    results = []
    
    for config in configurations:
        desc = config.pop('description')
        print(f"\n🔧 Testing {desc} configuration: {config}")
        
        # Create model
        model = MWNN(num_classes=3, **config)
        
        # Quick training
        _ = model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=1,
            learning_rate=0.01,
            save_checkpoints=False
        )
        
        # Evaluate
        test_results = model.evaluate(test_loader)
        results.append((desc, test_results['accuracy']))
        
        print(f"{desc} model accuracy: {test_results['accuracy']:.2f}%")
    
    print(f"\n📊 Configuration comparison:")
    for desc, accuracy in results:
        print(f"   {desc:12}: {accuracy:6.2f}%")
    
    print("✅ Configuration demo completed!")


def demo_class_scalability():
    """Demonstrate model scalability with different class counts."""
    print("\n=== MWNN API Demo - Class Scalability ===")
    
    class_counts = [2, 5, 10, 50]
    results = []
    
    for num_classes in class_counts:
        print(f"\n🎯 Testing with {num_classes} classes...")
        
        # Create data for this class count
        train_loader, test_loader = create_synthetic_dataloaders(
            num_samples=60,
            num_classes=num_classes,
            batch_size=6
        )
        
        # Create model
        model = MWNN(num_classes=num_classes, depth='shallow', base_channels=12)
        
        # Quick training
        _ = model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=1,
            learning_rate=0.01,
            save_checkpoints=False
        )
        
        # Evaluate
        test_results = model.evaluate(test_loader)
        results.append((num_classes, test_results['accuracy']))
        
        print(f"{num_classes} classes -> accuracy: {test_results['accuracy']:.2f}%")
    
    print(f"\n📊 Scalability results:")
    for num_classes, accuracy in results:
        print(f"   {num_classes:3} classes: {accuracy:6.2f}%")
    
    print("✅ Scalability demo completed!")


def demo_error_handling():
    """Demonstrate error handling and edge cases."""
    print("\n=== MWNN API Demo - Error Handling ===")
    
    # Test model creation with invalid parameters
    print("🚫 Testing invalid parameters...")
    
    try:
        # This should work
        model = MWNN(num_classes=10, depth='shallow')
        print("✅ Valid model creation succeeded")
    except Exception as e:
        print(f"❌ Unexpected error with valid parameters: {e}")
    
    # Test evaluation without training
    print("🚫 Testing evaluation without training...")
    try:
        model = MWNN(num_classes=5, depth='shallow')
        train_loader, test_loader = create_synthetic_dataloaders(num_samples=20, num_classes=5)
        
        # This should fail
        model.evaluate(test_loader)
        print("❌ Evaluation without training should have failed")
    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
    
    # Test saving without training
    print("🚫 Testing save without training...")
    try:
        model = MWNN(num_classes=5, depth='shallow')
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save(Path(temp_dir) / 'untrained.pth')
        print("❌ Save without training should have failed")
    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
    
    print("✅ Error handling demo completed!")


if __name__ == "__main__":
    """Run all demo tests."""
    print("🎭 MWNN API Demo Test Suite")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_basic_api()
        demo_different_configurations()
        demo_class_scalability()
        demo_error_handling()
        
        print("\n" + "=" * 50)
        print("🎉 All demo tests completed successfully!")
        print("MWNN API is working correctly with synthetic data.")
        print("\n💡 Next steps:")
        print("   • Run MNIST tests: python test_end_to_end_mnist.py")
        print("   • Run ImageNet tests: python test_end_to_end_imagenet.py")
        print("   • Run comprehensive tests: python test_end_to_end_comprehensive.py")
        
    except Exception as e:
        print(f"\n❌ Demo test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

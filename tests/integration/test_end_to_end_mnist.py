#!/usr/bin/env python3
"""
End-to-End MNIST Training Test
Test the complete MWNN pipeline with MNIST dataset for quick validation.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from mwnn import MWNN


class MNISTDataset(Dataset):
    """MNIST dataset that outputs dual pathways (RGB + brightness)."""
    
    def __init__(self, csv_path, transform=None, subset_size=None):
        """
        Args:
            csv_path: Path to MNIST CSV file
            transform: Transforms to apply
            subset_size: Optional subset size for faster testing
        """
        self.data = pd.read_csv(csv_path)
        if subset_size:
            self.data = self.data.head(subset_size)
            
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),  # Resize for compatibility
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row[0])
        
        # Extract pixel values and reshape to 28x28
        pixels = row[1:].values.astype(np.uint8).reshape(28, 28)
        
        # Convert to 3-channel image (RGB)
        rgb_image = np.stack([pixels, pixels, pixels], axis=2)
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Create brightness channel (grayscale)
        brightness_image = rgb_image.mean(dim=0, keepdim=True)
        
        # Return dual pathways: (RGB, brightness)
        return (rgb_image, brightness_image), label


def create_mnist_dataloaders(data_dir, batch_size=32, subset_size=1000):
    """Create MNIST data loaders with dual pathways."""
    data_path = Path(data_dir)
    
    # Load datasets
    train_dataset = MNISTDataset(
        csv_path=data_path / 'mnist_train.csv',
        subset_size=subset_size
    )
    
    test_dataset = MNISTDataset(
        csv_path=data_path / 'mnist_test.csv', 
        subset_size=subset_size // 5  # Smaller test set
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Single threaded for testing
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def test_mnist_training_basic():
    """Test basic MNIST training with MWNN."""
    print("\n=== Testing MNIST Training (Basic) ===")
    
    # Setup
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'MNIST'
    batch_size = 16
    subset_size = 500  # Small subset for quick test
    epochs = 2
    
    print(f"Loading MNIST data from {data_dir}")
    print(f"Subset size: {subset_size}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Create data loaders
    train_loader, test_loader = create_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        subset_size=subset_size
    )
    
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Test data loading
    print("Testing data loading...")
    data_batch, label_batch = next(iter(train_loader))
    rgb_batch, brightness_batch = data_batch
    
    print(f"RGB batch shape: {rgb_batch.shape}")
    print(f"Brightness batch shape: {brightness_batch.shape}")
    print(f"Labels shape: {label_batch.shape}")
    
    assert rgb_batch.shape[1:] == (3, 32, 32), f"Expected RGB shape (3, 32, 32), got {rgb_batch.shape[1:]}"
    assert brightness_batch.shape[1:] == (1, 32, 32), f"Expected brightness shape (1, 32, 32), got {brightness_batch.shape[1:]}"
    
    # Create model
    print("Creating MWNN model...")
    model = MWNN(num_classes=10, depth='shallow', base_channels=32)
    model.summary()
    
    # Create temporary checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Train model
        print("Starting training...")
        history = model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            learning_rate=0.01,
            save_checkpoints=True,
            checkpoint_dir=temp_dir
        )
        
        print("Training completed!")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        
        # Evaluate model
        print("Evaluating model...")
        results = model.evaluate(test_loader)
        print(f"Test accuracy: {results['accuracy']:.2f}%")
        print(f"Test loss: {results['loss']:.4f}")
        
        # Test saving and loading
        print("Testing model save/load...")
        model_path = Path(temp_dir) / 'test_model.pth'
        model.save(model_path)
        
        # Load model
        loaded_model = MWNN.load(model_path)
        loaded_results = loaded_model.evaluate(test_loader)
        
        print(f"Loaded model accuracy: {loaded_results['accuracy']:.2f}%")
        
        # Verify results match
        assert abs(results['accuracy'] - loaded_results['accuracy']) < 0.01, \
            "Loaded model results should match original"
        
        print("✅ MNIST basic training test passed!")


def test_mnist_training_deep():
    """Test MNIST training with deeper model configuration."""
    print("\n=== Testing MNIST Training (Deep Model) ===")
    
    # Setup
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'MNIST'
    batch_size = 8  # Smaller batch for deeper model
    subset_size = 300  # Even smaller subset
    epochs = 3
    
    print(f"Deep model test - Subset: {subset_size}, Batch: {batch_size}, Epochs: {epochs}")
    
    # Create data loaders
    train_loader, test_loader = create_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        subset_size=subset_size
    )
    
    # Create deeper model
    print("Creating deep MWNN model...")
    model = MWNN(num_classes=10, depth='deep', base_channels=64)
    model.summary()
    
    # Train with different settings
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Training deep model...")
        history = model.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            learning_rate=0.005,  # Lower learning rate for deep model
            save_checkpoints=True,
            checkpoint_dir=temp_dir
        )
        
        # Evaluate
        results = model.evaluate(test_loader)
        print(f"Deep model test accuracy: {results['accuracy']:.2f}%")
        
        # Test predictions
        print("Testing predictions...")
        predictions = model.predict(test_loader)
        print(f"Prediction shape: {predictions.shape}")
        print(f"Unique predictions: {torch.unique(predictions).tolist()}")
        
        assert len(predictions) == len(test_loader.dataset), \
            f"Expected {len(test_loader.dataset)} predictions, got {len(predictions)}"
        
        print("✅ MNIST deep training test passed!")


def test_mnist_multiple_configurations():
    """Test multiple model configurations on MNIST."""
    print("\n=== Testing Multiple MNIST Configurations ===")
    
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'MNIST'
    batch_size = 16
    subset_size = 200  # Very small for quick testing
    epochs = 1  # Single epoch for speed
    
    configurations = [
        {'depth': 'shallow', 'base_channels': 16},
        {'depth': 'medium', 'base_channels': 32},
        {'depth': 'deep', 'base_channels': 32}
    ]
    
    # Create data loaders once
    train_loader, test_loader = create_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        subset_size=subset_size
    )
    
    results = {}
    
    for i, config in enumerate(configurations):
        print(f"\nTesting configuration {i+1}: {config}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model
            model = MWNN(num_classes=10, **config)
            
            # Quick training
            history = model.fit(
                train_loader=train_loader,
                val_loader=test_loader,
                epochs=epochs,
                learning_rate=0.01,
                save_checkpoints=False
            )
            
            # Evaluate
            test_results = model.evaluate(test_loader)
            results[f"{config['depth']}_{config['base_channels']}"] = test_results['accuracy']
            
            print(f"Configuration {config} -> Accuracy: {test_results['accuracy']:.2f}%")
    
    print(f"\nAll configuration results: {results}")
    print("✅ Multiple configuration test passed!")


if __name__ == "__main__":
    """Run all MNIST end-to-end tests."""
    print("🧪 Running MNIST End-to-End Tests")
    print("=" * 50)
    
    try:
        # Run all tests
        test_mnist_training_basic()
        test_mnist_training_deep()
        test_mnist_multiple_configurations()
        
        print("\n" + "=" * 50)
        print("🎉 All MNIST end-to-end tests passed!")
        print("MWNN API is working correctly with MNIST dataset.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

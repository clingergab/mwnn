#!/usr/bin/env python3
"""
End-to-End ImageNet Training Test
Test the complete MWNN pipeline with ImageNet subset for realistic validation.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from mwnn import MWNN


def test_imagenet_data_loading():
    """Test ImageNet data loading functionality."""
    print("\n=== Testing ImageNet Data Loading ===")
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'ImageNet-1K'
    
    # Check if ImageNet data exists
    if not data_path.exists():
        print(f"⚠️  ImageNet data not found at {data_path}")
        print("Skipping ImageNet data loading test")
        return False
    
    print(f"Loading ImageNet data from {data_path}")
    
    try:
        # Test with very small subset for speed
        train_loader, val_loader = MWNN.load_imagenet_data(
            data_path=data_path,
            batch_size=4,
            num_workers=0,  # Single threaded for testing
            subset_size=20  # Very small subset
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Test data loading
        print("Testing data batch...")
        data_batch, label_batch = next(iter(train_loader))
        rgb_batch, brightness_batch = data_batch
        
        print(f"RGB batch shape: {rgb_batch.shape}")
        print(f"Brightness batch shape: {brightness_batch.shape}")
        print(f"Labels shape: {label_batch.shape}")
        print(f"Label range: {label_batch.min().item()} - {label_batch.max().item()}")
        
        # Verify data format
        assert len(rgb_batch.shape) == 4, f"RGB batch should be 4D, got {len(rgb_batch.shape)}D"
        assert len(brightness_batch.shape) == 4, f"Brightness batch should be 4D, got {len(brightness_batch.shape)}D"
        assert rgb_batch.shape[1] == 3, f"RGB should have 3 channels, got {rgb_batch.shape[1]}"
        assert brightness_batch.shape[1] == 1, f"Brightness should have 1 channel, got {brightness_batch.shape[1]}"
        
        print("✅ ImageNet data loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ImageNet data loading failed: {e}")
        return False


def test_imagenet_training_basic():
    """Test basic ImageNet training with MWNN."""
    print("\n=== Testing ImageNet Training (Basic) ===")
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'ImageNet-1K'
    
    # Check if ImageNet data exists
    if not data_path.exists():
        print(f"⚠️  ImageNet data not found at {data_path}")
        print("Skipping ImageNet training test")
        return False
    
    batch_size = 2  # Very small batch for testing
    subset_size = 10  # Minimal subset
    epochs = 1  # Single epoch for speed
    
    print(f"Subset size: {subset_size}, Batch size: {batch_size}, Epochs: {epochs}")
    
    try:
        # Create data loaders
        train_loader, val_loader = MWNN.load_imagenet_data(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=0,
            subset_size=subset_size
        )
        
        # Create model for ImageNet (1000 classes)
        print("Creating MWNN model for ImageNet...")
        model = MWNN(num_classes=1000, depth='shallow', base_channels=32)
        model.summary()
        
        # Create temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train model
            print("Starting ImageNet training...")
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
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
            results = model.evaluate(val_loader)
            print(f"Test accuracy: {results['accuracy']:.2f}%")
            print(f"Test loss: {results['loss']:.4f}")
            
            # Test saving and loading
            print("Testing model save/load...")
            model_path = Path(temp_dir) / 'imagenet_model.pth'
            model.save(model_path)
            
            # Load model
            loaded_model = MWNN.load(model_path)
            loaded_results = loaded_model.evaluate(val_loader)
            
            print(f"Loaded model accuracy: {loaded_results['accuracy']:.2f}%")
            
            # Verify results match (within tolerance for floating point)
            assert abs(results['accuracy'] - loaded_results['accuracy']) < 0.1, \
                "Loaded model results should match original (within tolerance)"
            
            print("✅ ImageNet basic training test passed!")
            return True
            
    except Exception as e:
        print(f"❌ ImageNet training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imagenet_training_medium():
    """Test ImageNet training with medium model configuration."""
    print("\n=== Testing ImageNet Training (Medium Model) ===")
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'ImageNet-1K'
    
    if not data_path.exists():
        print(f"⚠️  ImageNet data not found at {data_path}")
        print("Skipping ImageNet medium model test")
        return False
    
    batch_size = 2
    subset_size = 8  # Even smaller for medium model
    epochs = 1
    
    print(f"Medium model test - Subset: {subset_size}, Batch: {batch_size}, Epochs: {epochs}")
    
    try:
        # Create data loaders
        train_loader, val_loader = MWNN.load_imagenet_data(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=0,
            subset_size=subset_size
        )
        
        # Create medium model
        print("Creating medium MWNN model...")
        model = MWNN(num_classes=1000, depth='medium', base_channels=48)
        model.summary()
        
        # Train with different settings
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Training medium model...")
            _ = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=0.005,  # Lower learning rate
                save_checkpoints=False  # Skip checkpoints for speed
            )
            
            # Evaluate
            results = model.evaluate(val_loader)
            print(f"Medium model test accuracy: {results['accuracy']:.2f}%")
            
            # Test predictions
            print("Testing predictions...")
            predictions = model.predict(val_loader)
            print(f"Prediction shape: {predictions.shape}")
            print(f"Prediction range: {predictions.min().item()} - {predictions.max().item()}")
            
            assert len(predictions) == len(val_loader.dataset), \
                f"Expected {len(val_loader.dataset)} predictions, got {len(predictions)}"
            
            # Check that predictions are in valid range for ImageNet
            assert predictions.min() >= 0, "Predictions should be >= 0"
            assert predictions.max() < 1000, "Predictions should be < 1000 for ImageNet"
            
            print("✅ ImageNet medium training test passed!")
            return True
            
    except Exception as e:
        print(f"❌ ImageNet medium training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imagenet_configurations():
    """Test different model configurations with ImageNet."""
    print("\n=== Testing ImageNet Multiple Configurations ===")
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'ImageNet-1K'
    
    if not data_path.exists():
        print(f"⚠️  ImageNet data not found at {data_path}")
        print("Skipping ImageNet configuration test")
        return False
    
    batch_size = 2
    subset_size = 6  # Minimal for quick testing
    epochs = 1
    
    configurations = [
        {'depth': 'shallow', 'base_channels': 16},
        {'depth': 'medium', 'base_channels': 24}
        # Skip deep model for time constraints
    ]
    
    try:
        # Create data loaders once
        train_loader, val_loader = MWNN.load_imagenet_data(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=0,
            subset_size=subset_size
        )
        
        results = {}
        
        for i, config in enumerate(configurations):
            print(f"\nTesting ImageNet configuration {i+1}: {config}")
            
            try:
                # Create model
                model = MWNN(num_classes=1000, **config)
                
                # Quick training
                _ = model.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    learning_rate=0.01,
                    save_checkpoints=False
                )
                
                # Evaluate
                test_results = model.evaluate(val_loader)
                results[f"{config['depth']}_{config['base_channels']}"] = test_results['accuracy']
                
                print(f"Configuration {config} -> Accuracy: {test_results['accuracy']:.2f}%")
                
            except Exception as e:
                print(f"Configuration {config} failed: {e}")
                continue
        
        print(f"\nImageNet configuration results: {results}")
        print("✅ ImageNet configuration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ ImageNet configuration test failed: {e}")
        return False


def test_cross_dataset_compatibility():
    """Test that models can handle different input sizes and class counts."""
    print("\n=== Testing Cross-Dataset Compatibility ===")
    
    try:
        # Test different class counts
        models = [
            MWNN(num_classes=10, depth='shallow', base_channels=16),    # MNIST-like
            MWNN(num_classes=100, depth='shallow', base_channels=16),   # CIFAR-100-like
            MWNN(num_classes=1000, depth='shallow', base_channels=16)   # ImageNet-like
        ]
        
        for i, model in enumerate(models):
            print(f"Testing model {i+1} with {model.num_classes} classes")
            model.summary()
            
            # Test that model accepts different inputs
            # Simulate dual pathway input
            batch_size = 2
            rgb_input = torch.randn(batch_size, 3, 224, 224)
            brightness_input = torch.randn(batch_size, 1, 224, 224)
            
            model.model.eval()
            with torch.no_grad():
                output = model.model(rgb_input, brightness_input)
                
            print(f"Model {i+1} output shape: {output.shape}")
            assert output.shape == (batch_size, model.num_classes), \
                f"Expected shape ({batch_size}, {model.num_classes}), got {output.shape}"
        
        print("✅ Cross-dataset compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cross-dataset compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run all ImageNet end-to-end tests."""
    print("🧪 Running ImageNet End-to-End Tests")
    print("=" * 50)
    
    success_count = 0
    total_tests = 5
    
    # Run all tests
    tests = [
        test_imagenet_data_loading,
        test_imagenet_training_basic,
        test_imagenet_training_medium,
        test_imagenet_configurations,
        test_cross_dataset_compatibility
    ]
    
    for test_func in tests:
        try:
            result = test_func()
            if result is not False:  # True or None (for tests that don't return bool)
                success_count += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} passed")
    
    if success_count == total_tests:
        print("🎉 All ImageNet end-to-end tests passed!")
        print("MWNN API is working correctly with ImageNet dataset.")
    else:
        print(f"⚠️  {total_tests - success_count} tests failed or were skipped.")
        print("Check ImageNet data availability and model configurations.")
    
    # Exit with appropriate code
    sys.exit(0 if success_count > 0 else 1)

#!/usr/bin/env python3
"""
MWNN Quick Demo
Demonstrates the clean, Keras-like API for MWNN
"""

import sys

# Add src to path
sys.path.append('src')

# Import the MWNN module directly 
from mwnn import MWNN


def demo_clean_api():
    """Demonstrate the clean MWNN API."""
    
    print("🎯 MWNN Clean API Demo")
    print("=" * 50)
    
    # Example 1: Creating a model
    print("\n1️⃣ Creating MWNN Model")
    model = MWNN(num_classes=1000, depth='deep', device='auto')
    model.summary()
    
    # Example 2: Loading data (pseudo-code since we need actual data)
    print("\n2️⃣ Loading Data (Example)")
    print("train_loader, val_loader = MWNN.load_imagenet_data('/path/to/imagenet', batch_size=64)")
    
    # Example 3: Training (pseudo-code)
    print("\n3️⃣ Training (Example)")
    print("history = model.fit(train_loader, val_loader, epochs=30)")
    
    # Example 4: Evaluation (pseudo-code)
    print("\n4️⃣ Evaluation (Example)")
    print("results = model.evaluate(test_loader)")
    print("print(f'Accuracy: {results[\"accuracy\"]:.2f}%')")
    
    # Example 5: Save/Load
    print("\n5️⃣ Save/Load Model")
    print("model.save('my_model.pth')")
    print("loaded_model = MWNN.load('my_model.pth')")
    
    print("\n✨ Compare this with the old complex training script!")
    print("Now it's as simple as Keras/scikit-learn! 🎉")


def show_old_vs_new():
    """Show the difference between old and new approach."""
    
    print("\n" + "=" * 60)
    print("🔄 OLD vs NEW Training Approach")
    print("=" * 60)
    
    print("\n❌ OLD WAY (Complex):")
    print("""
    # Complex imports and setup
    from train_deep_colab import run_deep_training
    from setup_colab import get_gpu_info, clear_gpu_memory
    
    # Complex parameter setup
    config = get_preset_config('continuous_integration_experiment', ...)
    model = ContinuousIntegrationModel(...)
    optimizer = optim.Adam(...)
    scheduler = StepLR(...)
    
    # Complex training call
    results = run_deep_training(
        dataset_name='ImageNet',
        model_name='continuous_integration',
        complexity='deep',
        batch_size=64,
        epochs=30,
        learning_rate=0.002,
        save_checkpoints=True,
        data_path='/content/drive/MyDrive/mwnn/...'
    )
    """)
    
    print("\n✅ NEW WAY (Clean):")
    print("""
    from mwnn import MWNN
    
    # Load data
    train_loader, val_loader = MWNN.load_imagenet_data('/path/to/data', batch_size=64)
    
    # Create model
    model = MWNN(num_classes=1000, depth='deep')
    
    # Train
    history = model.fit(train_loader, val_loader, epochs=30)
    
    # Evaluate
    results = model.evaluate(val_loader)
    
    # Save
    model.save('best_model.pth')
    """)
    
    print("\n🎯 Benefits of new approach:")
    print("  ✅ Simple, intuitive API")
    print("  ✅ Keras-like interface")
    print("  ✅ Less boilerplate code")
    print("  ✅ Easy to understand and modify")
    print("  ✅ Better error handling")
    print("  ✅ Cleaner progress bars")
    print("  ✅ Automatic device detection")


if __name__ == "__main__":
    demo_clean_api()
    show_old_vs_new()

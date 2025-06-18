#!/usr/bin/env python3
"""
Test the clean MWNN imports
"""

import sys
import os

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Add src to path
sys.path.insert(0, 'src')

try:
    # Test individual imports
    print("Testing individual imports...")
    
    from utils.device import get_optimal_device
    print("✅ utils.device imported")
    
    from models.continuous_integration import ContinuousIntegrationModel
    print("✅ continuous_integration model imported")
    
    from preprocessing.imagenet_dataset import create_imagenet_separate_pathway_dataloaders
    print("✅ imagenet_dataset imported")
    
    from training.trainer import MWNNTrainer  
    print("✅ trainer imported")
    
    # Now test the main module
    from mwnn import MWNN
    print("✅ MWNN imported successfully!")
    
    # Test creating a model
    model = MWNN(num_classes=10, depth='shallow', device='cpu')
    print("✅ MWNN model created successfully!")
    
    model.summary()
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

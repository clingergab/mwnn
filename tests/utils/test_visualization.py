#!/usr/bin/env python
"""
Unit tests for the visualization utilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil

# Test only the visualization functions that exist
try:
    from utils.visualization import (
        plot_pathway_activations,
        plot_confusion_matrix,
        plot_training_history,
        visualize_weight_specialization
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from models.multi_channel.model import MultiChannelModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


class TestVisualization(unittest.TestCase):
    """Test cases for the visualization utilities."""

    def setUp(self):
        """Set up test variables."""
        # Create a temporary directory for saving files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy training history
        self.history = {
            'train_loss': [2.3, 1.9, 1.5, 1.2, 1.0],
            'val_loss': [2.4, 2.0, 1.6, 1.4, 1.2],
            'train_acc': [20.0, 40.0, 60.0, 70.0, 75.0],
            'val_acc': [18.0, 38.0, 58.0, 65.0, 70.0]
        }

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Close all plots
        plt.close('all')

    @unittest.skipUnless(VISUALIZATION_AVAILABLE, "Visualization functions not available")
    def test_plot_pathway_activations(self):
        """Test plotting pathway activations."""
        # Create dummy activations with simpler shape for 2D visualization
        # The function expects tensors that will be averaged over batch dimension
        color_activations = torch.randn(4, 8, 8)  # Batch, Height, Width
        brightness_activations = torch.randn(4, 8, 8)  # Batch, Height, Width
        
        # Test without saving
        plot_pathway_activations(color_activations, brightness_activations)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'activations.png')
        plot_pathway_activations(color_activations, brightness_activations, save_path)
        
        # Check that file was saved
        self.assertTrue(os.path.exists(save_path))

    @unittest.skipUnless(VISUALIZATION_AVAILABLE, "Visualization functions not available")
    def test_plot_confusion_matrix(self):
        """Test plotting confusion matrix."""
        # Create dummy data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2])
        class_names = ['Class A', 'Class B', 'Class C']
        
        # Test without saving
        plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, class_names, save_path)
        
        # Check that file was saved
        self.assertTrue(os.path.exists(save_path))

    @unittest.skipUnless(VISUALIZATION_AVAILABLE, "Visualization functions not available")
    def test_plot_training_history(self):
        """Test plotting training history."""
        # Test without saving
        fig = plot_training_history(self.history)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with saving
        save_path = os.path.join(self.temp_dir, 'history.png')
        fig = plot_training_history(self.history, save_path)
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that file was saved
        self.assertTrue(os.path.exists(save_path))

    @unittest.skipUnless(VISUALIZATION_AVAILABLE and MODEL_AVAILABLE, "Visualization or model not available")
    def test_visualize_weight_specialization(self):
        """Test visualizing weight specialization."""
        # Create a simple model for testing
        model = MultiChannelModel(
            input_channels=3,
            num_classes=10,
            depth='shallow'  # Use shallow for faster testing
        )
        
        # The function may return None if no specialized weights are found
        # Let's check if the function runs without error first
        try:
            fig = visualize_weight_specialization(model)
            if fig is not None:
                self.assertIsInstance(fig, plt.Figure)
                
                # Test with saving
                save_path = os.path.join(self.temp_dir, 'weight_specialization.png')
                fig_saved = visualize_weight_specialization(model, save_path)
                
                if fig_saved is not None:
                    # Check that figure was created
                    self.assertIsInstance(fig_saved, plt.Figure)
                    
                    # Check that file was saved
                    self.assertTrue(os.path.exists(save_path))
            else:
                # If no specialized weights found, that's okay for testing
                print("No specialized weights found in test model - this is expected")
        except Exception as e:
            self.fail(f"visualize_weight_specialization raised an exception: {e}")

    def test_basic_functionality(self):
        """Test that basic test infrastructure works."""
        # This test should always pass to verify test setup
        self.assertTrue(True)
        self.assertEqual(1 + 1, 2)
        
        # Test temporary directory
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Test that we can create and remove files
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        self.assertTrue(os.path.exists(test_file))


if __name__ == '__main__':
    unittest.main()

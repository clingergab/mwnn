"""
Unit tests for ImageNet dataset preprocessing functionality.

This module contains comprehensive tests for the ImageNet-1K dataset preprocessing
pipeline, including dataset creation, configuration management, and data loading.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader

# Import the modules we're testing
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.imagenet_dataset import (
    ImageNetMWNNDataset,
    ImageNetRGBLuminanceDataset,
    create_imagenet_mwnn_dataset,
    create_imagenet_rgb_luminance_dataset,
    create_imagenet_dataloaders,
    create_imagenet_rgb_luminance_dataloaders,
    get_imagenet_transforms,
    analyze_imagenet_dataset
)
from src.preprocessing.imagenet_config import (
    ImageNetPreprocessingConfig,
    get_preset_config,
    create_default_configs
)


class TestImageNetPreprocessingConfig(unittest.TestCase):
    """Test cases for ImageNet preprocessing configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.devkit_dir = Path(self.temp_dir) / "devkit"
        
        # Create test directories
        self.data_dir.mkdir(parents=True)
        self.devkit_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_creation_with_valid_paths(self):
        """Test configuration creation with valid paths."""
        config = ImageNetPreprocessingConfig(
            data_dir="/path/to/data",  # Placeholder path
            devkit_dir="/path/to/devkit",  # Placeholder path
            batch_size=32,
            input_size=224
        )
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.input_size, 224)
        self.assertEqual(config.feature_method, 'rgb_luminance')
    
    def test_config_validation_invalid_batch_size(self):
        """Test configuration validation with invalid batch size."""
        with self.assertRaises(ValueError):
            ImageNetPreprocessingConfig(
                data_dir="/path/to/data",
                devkit_dir="/path/to/devkit",
                batch_size=0  # Invalid
            )
    
    def test_config_validation_invalid_val_split(self):
        """Test configuration validation with invalid validation split."""
        with self.assertRaises(ValueError):
            ImageNetPreprocessingConfig(
                data_dir="/path/to/data",
                devkit_dir="/path/to/devkit",
                val_split=1.5  # Invalid
            )
    
    def test_preset_config_development(self):
        """Test development preset configuration."""
        config = get_preset_config(
            'development',
            data_dir="/path/to/data",
            devkit_dir="/path/to/devkit"
        )
        
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.load_subset, 100)
        self.assertFalse(config.augment)
    
    def test_preset_config_training(self):
        """Test training preset configuration."""
        config = get_preset_config(
            'training',
            data_dir="/path/to/data",
            devkit_dir="/path/to/devkit"
        )
        
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.augment)
        self.assertTrue(config.augment_features)
    
    def test_preset_config_evaluation(self):
        """Test evaluation preset configuration."""
        config = get_preset_config(
            'evaluation',
            data_dir="/path/to/data",
            devkit_dir="/path/to/devkit"
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertFalse(config.augment)
        self.assertEqual(config.val_split, 0.0)
    
    def test_preset_config_invalid(self):
        """Test invalid preset configuration."""
        with self.assertRaises(ValueError):
            get_preset_config(
                'invalid_preset',
                data_dir="/path/to/data",
                devkit_dir="/path/to/devkit"
            )
    
    def test_config_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        config = ImageNetPreprocessingConfig(
            data_dir="/path/to/data",
            devkit_dir="/path/to/devkit",
            batch_size=64,
            input_size=256
        )
        
        yaml_file = Path(self.temp_dir) / "test_config.yaml"
        config.to_yaml(str(yaml_file))
        
        # Verify file was created
        self.assertTrue(yaml_file.exists())
        
        # Load and verify
        loaded_config = ImageNetPreprocessingConfig.from_yaml(str(yaml_file))
        self.assertEqual(loaded_config.batch_size, 64)
        self.assertEqual(loaded_config.input_size, 256)


class TestImageNetTransforms(unittest.TestCase):
    """Test cases for ImageNet transforms."""
    
    def test_get_imagenet_transforms_no_augmentation(self):
        """Test getting transforms without augmentation."""
        transforms = get_imagenet_transforms(input_size=224, augment=False)
        
        # Should have Resize, CenterCrop, ToTensor, Normalize
        self.assertEqual(len(transforms.transforms), 4)
    
    def test_get_imagenet_transforms_with_augmentation(self):
        """Test getting transforms with augmentation."""
        transforms = get_imagenet_transforms(input_size=224, augment=True)
        
        # Should have RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
        self.assertEqual(len(transforms.transforms), 5)
    
    def test_custom_input_size(self):
        """Test transforms with custom input size."""
        transforms = get_imagenet_transforms(input_size=256, augment=False)
        
        # Check that resize transform uses correct size
        resize_transform = transforms.transforms[0]
        expected_size = int(256 * 1.143)  # 256 * 1.143 ≈ 293
        self.assertEqual(resize_transform.size, expected_size)


class MockImageNetDataset:
    """Mock ImageNet dataset for testing."""
    
    def setUp(self):
        """Set up mock dataset structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "ImageNet-1K"
        self.devkit_dir = self.data_dir / "ILSVRC2013_devkit"
        
        # Create directory structure
        val_images_dir = self.data_dir / "val_images"
        val_images_dir.mkdir(parents=True)
        
        devkit_data_dir = self.devkit_dir / "data"
        devkit_data_dir.mkdir(parents=True)
        
        # Create mock images
        for i in range(1, 11):  # Create 10 mock images
            img_file = val_images_dir / f"ILSVRC2012_val_{i:08d}_n{i:08d}.JPEG"
            img_file.touch()
        
        # Create mock ground truth file
        ground_truth_file = devkit_data_dir / "ILSVRC2013_clsloc_validation_ground_truth.txt"
        with open(ground_truth_file, 'w') as f:
            for i in range(1, 11):
                f.write(f"{i}\n")  # Labels 1-10
        
        return self.temp_dir, str(self.data_dir), str(self.devkit_dir)
    
    def tearDown(self):
        """Clean up mock dataset."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir)


class TestImageNetMWNNDataset(unittest.TestCase):
    """Test cases for ImageNet MWNN dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dataset = MockImageNetDataset()
        self.temp_dir, self.data_dir, self.devkit_dir = self.mock_dataset.setUp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_dataset.tearDown()
    
    @patch('src.preprocessing.imagenet_dataset.Image')
    def test_dataset_creation_basic(self, mock_image):
        """Test basic dataset creation."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        # Mock transforms to return a tensor
        with patch('src.preprocessing.imagenet_dataset.transforms'):
            dataset = ImageNetMWNNDataset(
                data_dir=self.data_dir,
                devkit_dir=self.devkit_dir,
                load_subset=5
            )
            
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset.num_classes, 1000)
    
    def test_dataset_creation_invalid_paths(self):
        """Test dataset creation with invalid paths."""
        with self.assertRaises(FileNotFoundError):
            ImageNetMWNNDataset(
                data_dir="/nonexistent/path",
                devkit_dir="/nonexistent/devkit"
            )
    
    @patch('src.preprocessing.imagenet_dataset.Image')
    def test_dataset_class_mappings(self, mock_image):
        """Test class mapping functionality."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        with patch('src.preprocessing.imagenet_dataset.transforms'):
            dataset = ImageNetMWNNDataset(
                data_dir=self.data_dir,
                devkit_dir=self.devkit_dir,
                load_subset=5
            )
            
            # Test class name retrieval
            class_name = dataset.get_class_name(0)
            self.assertIsInstance(class_name, str)
            
            # Test class distribution
            distribution = dataset.get_class_distribution()
            self.assertIsInstance(distribution, dict)
    
    @patch('src.preprocessing.imagenet_dataset.Image')
    def test_dataset_different_feature_methods(self, mock_image):
        """Test dataset with different feature methods."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        feature_methods = ['rgb_luminance', 'hsv', 'lab', 'yuv']
        
        for method in feature_methods:
            with patch('src.preprocessing.imagenet_dataset.transforms'):
                dataset = ImageNetMWNNDataset(
                    data_dir=self.data_dir,
                    devkit_dir=self.devkit_dir,
                    feature_method=method,
                    load_subset=3
                )
                
                self.assertEqual(len(dataset), 3)
                self.assertEqual(dataset.feature_extractor.method, method)
    
    @patch('src.preprocessing.imagenet_dataset.Image')
    def test_rgb_luminance_dataset(self, mock_image):
        """Test RGB+Luminance dataset creation and functionality."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        with patch('src.preprocessing.imagenet_dataset.transforms'):
            dataset = ImageNetRGBLuminanceDataset(
                data_dir=self.data_dir,
                devkit_dir=self.devkit_dir,
                load_subset=3
            )
            
            self.assertEqual(len(dataset), 3)
            # RGB+Luminance should return 4-channel tensors
            # This would be tested with actual data loading
    
    @patch('src.preprocessing.imagenet_dataset.ImageNetRGBLuminanceDataset')
    def test_create_rgb_luminance_dataloaders(self, mock_dataset_class):
        """Test creation of RGB+Luminance data loaders."""
        # Mock dataset instance
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset_class.return_value = mock_dataset
        
        train_loader, val_loader = create_imagenet_rgb_luminance_dataloaders(
            data_dir=self.data_dir,
            devkit_dir=self.devkit_dir,
            batch_size=4,
            train_split=0.7,
            load_subset=10
        )
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertEqual(train_loader.batch_size, 4)
        self.assertEqual(val_loader.batch_size, 4)


class TestImageNetDataLoaders(unittest.TestCase):
    """Test cases for ImageNet data loaders."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dataset = MockImageNetDataset()
        self.temp_dir, self.data_dir, self.devkit_dir = self.mock_dataset.setUp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_dataset.tearDown()
    
    @patch('src.preprocessing.imagenet_dataset.ImageNetMWNNDataset')
    def test_create_imagenet_mwnn_dataset(self, mock_dataset_class):
        """Test create_imagenet_mwnn_dataset function."""
        # Mock dataset instance
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset
        
        dataset = create_imagenet_mwnn_dataset(
            data_dir=self.data_dir,
            devkit_dir=self.devkit_dir,
            load_subset=10
        )
        
        # Verify dataset was created with correct parameters
        mock_dataset_class.assert_called_once()
        self.assertEqual(len(dataset), 10)
    
    @patch('src.preprocessing.imagenet_dataset.ImageNetMWNNDataset')
    @patch('torch.utils.data.random_split')
    def test_create_imagenet_dataloaders(self, mock_split, mock_dataset_class):
        """Test create_imagenet_dataloaders function."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset
        
        # Mock split subsets with proper length
        mock_train_subset = MagicMock()
        mock_train_subset.__len__.return_value = 80
        mock_val_subset = MagicMock()
        mock_val_subset.__len__.return_value = 20
        mock_split.return_value = [mock_train_subset, mock_val_subset]
        
        train_loader, val_loader = create_imagenet_dataloaders(
            data_dir=self.data_dir,
            devkit_dir=self.devkit_dir,
            batch_size=16,
            load_subset=100
        )
        
        # Verify loaders were created
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        
        # Verify split was called
        mock_split.assert_called()


class TestImageNetAnalysis(unittest.TestCase):
    """Test cases for ImageNet analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dataset = MockImageNetDataset()
        self.temp_dir, self.data_dir, self.devkit_dir = self.mock_dataset.setUp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_dataset.tearDown()
    
    @patch('src.preprocessing.imagenet_dataset.create_imagenet_mwnn_dataset')
    @patch('builtins.print')
    def test_analyze_imagenet_dataset(self, mock_print, mock_create_dataset):
        """Test analyze_imagenet_dataset function."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.num_classes = 1000
        mock_dataset.get_class_distribution.return_value = {0: 5, 1: 3, 2: 2}
        mock_dataset.get_class_name.return_value = "test_class"
        mock_dataset.__getitem__.return_value = (torch.randn(3, 224, 224), 0)
        mock_create_dataset.return_value = mock_dataset
        
        # Run analysis
        analyze_imagenet_dataset(
            data_dir=self.data_dir,
            devkit_dir=self.devkit_dir,
            sample_size=10
        )
        
        # Verify dataset was created and analysis ran
        mock_create_dataset.assert_called_once()
        mock_print.assert_called()  # Should have printed analysis results


class TestImageNetValidation(unittest.TestCase):
    """Test cases for ImageNet preprocessing validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dataset = MockImageNetDataset()
        self.temp_dir, self.data_dir, self.devkit_dir = self.mock_dataset.setUp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_dataset.tearDown()
    
    def test_directory_structure_validation(self):
        """Test directory structure validation."""
        # Test valid structure
        self.assertTrue(Path(self.data_dir).exists())
        self.assertTrue(Path(self.devkit_dir).exists())
        self.assertTrue((Path(self.data_dir) / "val_images").exists())
        self.assertTrue((Path(self.devkit_dir) / "data" / "ILSVRC2013_clsloc_validation_ground_truth.txt").exists())
    
    def test_ground_truth_file_format(self):
        """Test ground truth file format."""
        ground_truth_file = Path(self.devkit_dir) / "data" / "ILSVRC2013_clsloc_validation_ground_truth.txt"
        
        with open(ground_truth_file, 'r') as f:
            lines = f.readlines()
        
        # Should have 10 lines (one per mock image)
        self.assertEqual(len(lines), 10)
        
        # Each line should be a valid integer
        for i, line in enumerate(lines):
            label = int(line.strip())
            self.assertEqual(label, i + 1)  # Labels 1-10
    
    @patch('src.preprocessing.imagenet_dataset.ImageNetMWNNDataset')
    def test_configuration_validation(self, mock_dataset_class):
        """Test configuration validation."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5
        mock_dataset_class.return_value = mock_dataset
        
        presets = ['development', 'training', 'evaluation', 'research']
        
        for preset in presets:
            config = get_preset_config(
                preset=preset,
                data_dir=self.data_dir,
                devkit_dir=self.devkit_dir
            )
            
            # Verify config is valid
            self.assertIsInstance(config, ImageNetPreprocessingConfig)
            self.assertGreater(config.batch_size, 0)
            self.assertGreater(config.input_size, 0)
            self.assertIn(config.feature_method, ['rgb_luminance', 'hsv', 'lab', 'yuv'])


class TestImageNetIntegration(unittest.TestCase):
    """Integration tests for ImageNet preprocessing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_dataset = MockImageNetDataset()
        self.temp_dir, self.data_dir, self.devkit_dir = self.mock_dataset.setUp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.mock_dataset.tearDown()
    
    @patch('src.preprocessing.imagenet_dataset.Image')
    @patch('src.preprocessing.imagenet_dataset.transforms')
    def test_end_to_end_pipeline(self, mock_transforms, mock_image):
        """Test complete end-to-end preprocessing pipeline."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        # Mock transforms to return tensors
        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        mock_transforms.Compose.return_value = mock_transform
        
        # Create configuration
        config = get_preset_config(
            'development',
            data_dir=self.data_dir,
            devkit_dir=self.devkit_dir
        )
        
        # Create dataset
        dataset = create_imagenet_mwnn_dataset(
            data_dir=config.data_dir,
            devkit_dir=config.devkit_dir,
            input_size=config.input_size,
            feature_method=config.feature_method,
            augment=config.augment,
            load_subset=5
        )
        
        # Verify dataset
        self.assertEqual(len(dataset), 5)
        
        # Test data loading
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertGreaterEqual(label, 0)
        self.assertLess(label, 1000)
    
    @patch('src.preprocessing.imagenet_dataset.ImageNetMWNNDataset')
    def test_dataloader_integration(self, mock_dataset_class):
        """Test dataloader integration."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 20
        mock_dataset.__getitem__.return_value = (torch.randn(3, 224, 224), 0)
        mock_dataset_class.return_value = mock_dataset
        
        # Create dataloaders
        train_loader, val_loader = create_imagenet_dataloaders(
            data_dir=self.data_dir,
            devkit_dir=self.devkit_dir,
            batch_size=4,
            load_subset=20,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        # Test batch loading
        for images, labels in train_loader:
            self.assertEqual(images.shape[0], 4)  # Batch size
            self.assertEqual(images.shape[1:], (3, 224, 224))  # Image shape
            self.assertEqual(labels.shape[0], 4)  # Batch size
            break  # Only test first batch


class TestImageNetConfigFiles(unittest.TestCase):
    """Test cases for ImageNet configuration file management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_default_configs(self):
        """Test creation of default configuration files."""
        create_default_configs(self.temp_dir)
        
        config_dir = Path(self.temp_dir) / "configs" / "preprocessing"
        
        # Verify all config files were created
        expected_files = [
            "imagenet_development.yaml",
            "imagenet_training.yaml",
            "imagenet_evaluation.yaml",
            "imagenet_research.yaml",
            "imagenet_template.yaml",
            "README.md"
        ]
        
        for filename in expected_files:
            file_path = config_dir / filename
            self.assertTrue(file_path.exists(), f"Missing config file: {filename}")
    
    def test_config_file_content(self):
        """Test configuration file content validity."""
        create_default_configs(self.temp_dir)
        
        config_dir = Path(self.temp_dir) / "configs" / "preprocessing"
        config_file = config_dir / "imagenet_training.yaml"
        
        # Load and validate config
        config = ImageNetPreprocessingConfig.from_yaml(str(config_file))
        
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.augment)
        self.assertEqual(config.feature_method, 'rgb_luminance')


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestImageNetPreprocessingConfig,
        TestImageNetTransforms,
        TestImageNetMWNNDataset,
        TestImageNetDataLoaders,
        TestImageNetAnalysis,
        TestImageNetValidation,
        TestImageNetIntegration,
        TestImageNetConfigFiles
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)

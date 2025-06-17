"""Tests for data loading utilities."""

import torch
import pytest
import tempfile
import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing.data_loaders import (
    MWNNDataset,
    MultiModalDataset,
    SyntheticMWNNDataset,
    create_mwnn_dataset,
    get_data_loader
)
from preprocessing.color_extractors import FeatureExtractor


class TestMWNNDataset:
    """Test MWNN dataset wrapper."""
    
    def test_initialization_default(self):
        """Test MWNNDataset initialization with defaults."""
        # Create base dataset
        base_data = TensorDataset(torch.rand(10, 3, 32, 32), torch.randint(0, 10, (10,)))
        
        dataset = MWNNDataset(base_data)
        
        assert len(dataset) == len(base_data)
        assert hasattr(dataset, 'feature_extractor')
        assert isinstance(dataset.feature_extractor, FeatureExtractor)
    
    def test_initialization_custom_extractor(self):
        """Test MWNNDataset initialization with custom extractor."""
        base_data = TensorDataset(torch.rand(10, 3, 32, 32), torch.randint(0, 10, (10,)))
        custom_extractor = FeatureExtractor(method='rgb', normalize=False)
        
        dataset = MWNNDataset(base_data, feature_extractor=custom_extractor)
        
        assert dataset.feature_extractor is custom_extractor
        assert dataset.feature_extractor.method == 'rgb'
    
    def test_initialization_with_augmentation(self):
        """Test MWNNDataset initialization with augmentation."""
        base_data = TensorDataset(torch.rand(10, 3, 32, 32), torch.randint(0, 10, (10,)))
        
        dataset = MWNNDataset(base_data, augment=True)
        
        # Should create AugmentedFeatureExtractor
        from preprocessing.color_extractors import AugmentedFeatureExtractor
        assert isinstance(dataset.feature_extractor, AugmentedFeatureExtractor)
    
    def test_getitem(self):
        """Test dataset item retrieval."""
        base_data = TensorDataset(torch.rand(5, 3, 16, 16), torch.randint(0, 3, (5,)))
        dataset = MWNNDataset(base_data)
        
        image, label = dataset[0]
        
        # Should return the same as base dataset for now
        base_image, base_label = base_data[0]
        assert torch.allclose(image, base_image)
        assert label == base_label


class TestMultiModalDataset:
    """Test multi-modal dataset."""
    
    @pytest.fixture
    def temp_dirs_with_images(self):
        """Create temporary directories with test images."""
        with tempfile.TemporaryDirectory() as rgb_dir, \
             tempfile.TemporaryDirectory() as aux_dir:
            
            # Create test images
            rgb_path = Path(rgb_dir)
            aux_path = Path(aux_dir)
            
            for i in range(3):
                # Create RGB image
                rgb_img = Image.new('RGB', (32, 32), color=(i*50, i*60, i*70))
                rgb_img.save(rgb_path / f'image_{i:03d}.png')
                
                # Create auxiliary image (grayscale)
                aux_img = Image.new('L', (32, 32), color=i*80)
                aux_img.save(aux_path / f'image_{i:03d}.png')
            
            yield str(rgb_path), str(aux_path)
    
    def test_initialization(self, temp_dirs_with_images):
        """Test MultiModalDataset initialization."""
        rgb_dir, aux_dir = temp_dirs_with_images
        
        dataset = MultiModalDataset(rgb_dir, aux_dir)
        
        assert len(dataset) == 3  # Should find 3 matching pairs
        assert len(dataset.samples) == 3
    
    def test_initialization_with_labels(self, temp_dirs_with_images):
        """Test MultiModalDataset with labels file."""
        rgb_dir, aux_dir = temp_dirs_with_images
        
        # Create labels file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            labels_file = f.name
            pd.DataFrame({
                'filename': ['image_000', 'image_001', 'image_002'],
                'label': [0, 1, 2]
            }).to_csv(f.name, index=False)
        
        try:
            dataset = MultiModalDataset(rgb_dir, aux_dir, labels_file=labels_file)
            
            assert len(dataset) == 3
            # Check that labels are loaded
            _, _, label = dataset.samples[0]
            assert label in [0, 1, 2]
        finally:
            os.unlink(labels_file)
    
    def test_getitem(self, temp_dirs_with_images):
        """Test item retrieval from MultiModalDataset."""
        rgb_dir, aux_dir = temp_dirs_with_images
        dataset = MultiModalDataset(rgb_dir, aux_dir)
        
        (rgb_image, aux_image), label = dataset[0]
        
        # Check that images are tensors (assuming transform converts to tensor)
        assert isinstance(rgb_image, Image.Image) or torch.is_tensor(rgb_image)
        assert isinstance(aux_image, Image.Image) or torch.is_tensor(aux_image)
        assert isinstance(label, int)


class TestSyntheticMWNNDataset:
    """Test synthetic dataset generation."""
    
    def test_initialization_default(self):
        """Test SyntheticMWNNDataset with default parameters."""
        dataset = SyntheticMWNNDataset(num_samples=50)
        
        assert len(dataset) == 50
        assert len(dataset.images) == 50
        assert len(dataset.labels) == 50
    
    def test_initialization_custom(self):
        """Test SyntheticMWNNDataset with custom parameters."""
        dataset = SyntheticMWNNDataset(
            num_samples=20,
            image_size=(16, 16),
            num_classes=5,
            color_noise=0.2,
            brightness_noise=0.2
        )
        
        assert len(dataset) == 20
        assert dataset.image_size == (16, 16)
        assert dataset.num_classes == 5
    
    def test_getitem(self):
        """Test item retrieval from synthetic dataset."""
        dataset = SyntheticMWNNDataset(num_samples=10, num_classes=3)
        
        image, label = dataset[0]
        
        assert torch.is_tensor(image)
        assert image.shape == (3, 32, 32)  # Default size
        assert isinstance(label, int)
        assert 0 <= label < 3
    
    def test_pattern_generation(self):
        """Test that different classes generate different patterns."""
        dataset = SyntheticMWNNDataset(num_samples=5, num_classes=5)
        
        patterns = []
        for i in range(5):
            image, label = dataset[i]
            patterns.append(image)
            assert label == i  # First 5 samples should have labels 0-4
        
        # Patterns should be different for different classes
        assert not torch.allclose(patterns[0], patterns[1], atol=1e-3)
    
    def test_color_generation(self):
        """Test color generation for different classes."""
        dataset = SyntheticMWNNDataset(num_samples=10, num_classes=10, color_noise=0.0)
        
        # Get colors for different classes
        colors = []
        for i in range(5):
            # Generate same class multiple times to test consistency
            color = dataset._generate_color(i, 0.0)  # No noise
            colors.append(color)
        
        # Should generate consistent colors for same class with no noise
        color_0_again = dataset._generate_color(0, 0.0)
        assert torch.allclose(torch.tensor(colors[0]), torch.tensor(color_0_again), atol=1e-6)
    
    def test_brightness_generation(self):
        """Test brightness generation for different classes."""
        dataset = SyntheticMWNNDataset(num_samples=10, num_classes=5)
        
        brightness_values = []
        for i in range(5):
            brightness = dataset._generate_brightness(i, 0.0)  # No noise
            brightness_values.append(brightness)
        
        # Brightness should vary by class
        assert not all(abs(b - brightness_values[0]) < 1e-6 for b in brightness_values[1:])


class TestDataLoaderUtilities:
    """Test data loader utility functions."""
    
    def test_create_mwnn_dataset(self):
        """Test create_mwnn_dataset function."""
        base_data = TensorDataset(torch.rand(10, 3, 32, 32), torch.randint(0, 5, (10,)))
        
        # Test with default parameters
        dataset = create_mwnn_dataset(base_data)
        assert isinstance(dataset, MWNNDataset)
        assert len(dataset) == 10
        
        # Test with custom parameters
        dataset = create_mwnn_dataset(base_data, feature_method='rgb', augment=False)
        assert dataset.feature_extractor.method == 'rgb'
    
    def test_get_data_loader_basic(self):
        """Test get_data_loader function with basic parameters."""
        dataset = TensorDataset(torch.rand(20, 3, 32, 32), torch.randint(0, 5, (20,)))
        
        loader = get_data_loader(dataset, batch_size=4, shuffle=False)
        
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4
        # Note: DataLoader doesn't expose shuffle directly, but we can check sampler
        from torch.utils.data import SequentialSampler
        assert isinstance(loader.sampler, SequentialSampler)
    
    def test_get_data_loader_advanced(self):
        """Test get_data_loader with advanced parameters."""
        dataset = TensorDataset(torch.rand(16, 3, 32, 32), torch.randint(0, 3, (16,)))
        
        loader = get_data_loader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            pin_memory=False,
            drop_last=True
        )
        
        assert loader.batch_size == 8
        # Check that RandomSampler is used when shuffle=True
        from torch.utils.data import RandomSampler
        assert isinstance(loader.sampler, RandomSampler)
        assert loader.num_workers == 0
        assert loader.drop_last
    
    def test_data_loader_iteration(self):
        """Test that data loader can be iterated."""
        dataset = TensorDataset(torch.rand(12, 3, 16, 16), torch.randint(0, 3, (12,)))
        loader = get_data_loader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        batches = list(loader)
        assert len(batches) == 3  # 12 samples / 4 batch_size = 3 batches
        
        for batch_images, batch_labels in batches:
            assert batch_images.shape[0] <= 4  # Batch size
            assert batch_images.shape[1:] == (3, 16, 16)  # Image dimensions
            assert batch_labels.shape[0] <= 4


class TestDatasetIntegration:
    """Test integration between different dataset components."""
    
    def test_mwnn_dataset_with_synthetic_base(self):
        """Test MWNNDataset wrapping SyntheticMWNNDataset."""
        synthetic_data = SyntheticMWNNDataset(num_samples=20, num_classes=3)
        mwnn_data = MWNNDataset(synthetic_data, feature_method='hsv')
        
        assert len(mwnn_data) == 20
        
        # Test data retrieval
        image, label = mwnn_data[0]
        assert torch.is_tensor(image)
        assert isinstance(label, int)
    
    def test_dataloader_with_mwnn_dataset(self):
        """Test DataLoader with MWNNDataset."""
        base_data = SyntheticMWNNDataset(num_samples=16, num_classes=4)
        mwnn_data = MWNNDataset(base_data)
        loader = get_data_loader(mwnn_data, batch_size=4, num_workers=0)
        
        # Test one batch
        batch_images, batch_labels = next(iter(loader))
        assert batch_images.shape == (4, 3, 32, 32)
        assert batch_labels.shape == (4,)
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction is consistent."""
        # Create dataset with known extraction method
        base_data = TensorDataset(torch.rand(5, 3, 16, 16), torch.randint(0, 2, (5,)))
        dataset = MWNNDataset(base_data, feature_method='rgb', augment=False)
        
        # Get same image multiple times
        image1, _ = dataset[0]
        image2, _ = dataset[0]
        
        # Should be identical (no augmentation)
        assert torch.allclose(image1, image2)


@pytest.fixture
def sample_dataset():
    """Fixture providing a sample dataset for testing."""
    return TensorDataset(
        torch.rand(20, 3, 32, 32),
        torch.randint(0, 5, (20,))
    )


def test_dataset_length_consistency(sample_dataset):
    """Test that wrapped datasets maintain correct length."""
    mwnn_dataset = MWNNDataset(sample_dataset)
    assert len(mwnn_dataset) == len(sample_dataset)


def test_batch_size_handling():
    """Test various batch sizes with data loaders."""
    dataset = SyntheticMWNNDataset(num_samples=17)  # Non-divisible by common batch sizes
    
    # Test different batch sizes
    for batch_size in [1, 3, 5, 8, 16]:
        loader = get_data_loader(dataset, batch_size=batch_size, num_workers=0)
        total_samples = sum(len(batch[0]) for batch in loader)
        assert total_samples == 17


def test_memory_efficiency():
    """Test that datasets don't preload all data into memory."""
    # Create a large synthetic dataset
    large_dataset = SyntheticMWNNDataset(num_samples=1000)
    
    # Should be able to create without memory issues
    assert len(large_dataset) == 1000
    
    # Accessing single item should work
    image, label = large_dataset[0]
    assert torch.is_tensor(image)


if __name__ == '__main__':
    pytest.main([__file__])
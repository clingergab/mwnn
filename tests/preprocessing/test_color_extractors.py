"""Tests for color extraction utilities."""

import torch
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from preprocessing.color_extractors import (
    ColorSpaceConverter, 
    FeatureExtractor, 
    AugmentedFeatureExtractor,
    MultiModalFeatureExtractor
)


class TestColorSpaceConverter:
    """Test color space conversion functionality."""
    
    def test_initialization(self):
        """Test ColorSpaceConverter initialization."""
        # Valid initialization
        converter = ColorSpaceConverter('rgb', 'hsv')
        assert converter.input_space == 'rgb'
        assert converter.output_space == 'hsv'
        
        # Invalid input space
        with pytest.raises(ValueError, match="Input space must be one of"):
            ColorSpaceConverter('invalid', 'hsv')
        
        # Invalid output space
        with pytest.raises(ValueError, match="Output space must be one of"):
            ColorSpaceConverter('rgb', 'invalid')
    
    def test_rgb_to_hsv_conversion(self):
        """Test RGB to HSV conversion."""
        converter = ColorSpaceConverter('rgb', 'hsv')
        
        # Test with known values
        # Pure red (255, 0, 0) -> (0, 1, 1) in HSV
        rgb = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], 
                           [[0.0, 0.0], [0.0, 0.0]], 
                           [[0.0, 0.0], [0.0, 0.0]]]])  # [1, 3, 2, 2]
        
        hsv = converter.rgb_to_hsv(rgb)
        
        assert hsv.shape == rgb.shape
        assert torch.allclose(hsv[0, 0], torch.zeros_like(hsv[0, 0]), atol=1e-6)  # Hue = 0 for red
        assert torch.allclose(hsv[0, 1], torch.ones_like(hsv[0, 1]), atol=1e-6)   # Saturation = 1
        assert torch.allclose(hsv[0, 2], torch.ones_like(hsv[0, 2]), atol=1e-6)   # Value = 1
    
    def test_rgb_to_yuv_conversion(self):
        """Test RGB to YUV conversion."""
        converter = ColorSpaceConverter('rgb', 'yuv')
        
        # Test with grayscale input
        rgb = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], 
                           [[0.5, 0.5], [0.5, 0.5]], 
                           [[0.5, 0.5], [0.5, 0.5]]]])  # Gray
        
        yuv = converter.rgb_to_yuv(rgb)
        
        assert yuv.shape == rgb.shape
        # For grayscale, U and V should be close to 0.5 (neutral)
        assert torch.allclose(yuv[0, 1], torch.full_like(yuv[0, 1], 0.5), atol=1e-2)
        assert torch.allclose(yuv[0, 2], torch.full_like(yuv[0, 2], 0.5), atol=1e-2)
    
    def test_forward_pass(self):
        """Test forward pass with different color spaces."""
        # RGB to HSV
        converter_hsv = ColorSpaceConverter('rgb', 'hsv')
        rgb_input = torch.rand(2, 3, 32, 32)
        hsv_output = converter_hsv(rgb_input)
        
        assert hsv_output.shape == rgb_input.shape
        assert hsv_output.dtype == rgb_input.dtype
        
        # RGB to YUV
        converter_yuv = ColorSpaceConverter('rgb', 'yuv')
        yuv_output = converter_yuv(rgb_input)
        
        assert yuv_output.shape == rgb_input.shape
        assert yuv_output.dtype == rgb_input.dtype
        
        # Same space conversion
        converter_same = ColorSpaceConverter('rgb', 'rgb')
        same_output = converter_same(rgb_input)
        
        assert torch.allclose(same_output, rgb_input)


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        # Default initialization
        extractor = FeatureExtractor()
        assert extractor.method == 'hsv'
        assert extractor.normalize
        
        # Custom initialization
        extractor = FeatureExtractor(method='rgb', normalize=False)
        assert extractor.method == 'rgb'
        assert not extractor.normalize
    
    def test_hsv_feature_extraction(self):
        """Test HSV-based feature extraction."""
        extractor = FeatureExtractor(method='hsv', normalize=False)
        
        # Create test image
        rgb_input = torch.rand(2, 3, 16, 16)
        
        color, brightness = extractor(rgb_input)
        
        # Check output shapes
        assert color.shape == (2, 2, 16, 16)  # H, S channels
        assert brightness.shape == (2, 1, 16, 16)  # V channel
        
        # Check value ranges for non-normalized output
        assert color.min() >= 0 and color.max() <= 1
        assert brightness.min() >= 0 and brightness.max() <= 1
    
    def test_yuv_feature_extraction(self):
        """Test YUV-based feature extraction."""
        extractor = FeatureExtractor(method='yuv', normalize=False)
        
        rgb_input = torch.rand(2, 3, 16, 16)
        color, brightness = extractor(rgb_input)
        
        # Check output shapes
        assert color.shape == (2, 2, 16, 16)  # U, V channels
        assert brightness.shape == (2, 1, 16, 16)  # Y channel
    
    def test_rgb_feature_extraction(self):
        """Test RGB-based feature extraction."""
        extractor = FeatureExtractor(method='rgb', normalize=False)
        
        rgb_input = torch.rand(2, 3, 16, 16)
        color, brightness = extractor(rgb_input)
        
        # Check output shapes
        assert color.shape == (2, 3, 16, 16)  # Normalized RGB
        assert brightness.shape == (2, 1, 16, 16)  # Average brightness
        
        # Check that brightness is mean of RGB channels
        expected_brightness = torch.mean(rgb_input, dim=1, keepdim=True)
        assert torch.allclose(brightness, expected_brightness, atol=1e-6)
    
    def test_normalization(self):
        """Test feature normalization."""
        extractor_norm = FeatureExtractor(method='hsv', normalize=True)
        extractor_no_norm = FeatureExtractor(method='hsv', normalize=False)
        
        rgb_input = torch.rand(2, 3, 16, 16)
        
        color_norm, brightness_norm = extractor_norm(rgb_input)
        color_no_norm, brightness_no_norm = extractor_no_norm(rgb_input)
        
        # Normalized features should have different statistics
        assert not torch.allclose(color_norm, color_no_norm, atol=1e-3)
        assert not torch.allclose(brightness_norm, brightness_no_norm, atol=1e-3)
    
    def test_invalid_method(self):
        """Test error handling for invalid extraction method."""
        extractor = FeatureExtractor(method='invalid')
        rgb_input = torch.rand(2, 3, 16, 16)
        
        with pytest.raises(ValueError, match="Unknown extraction method"):
            extractor(rgb_input)


class TestAugmentedFeatureExtractor:
    """Test augmented feature extraction."""
    
    def test_initialization(self):
        """Test AugmentedFeatureExtractor initialization."""
        extractor = AugmentedFeatureExtractor(
            method='hsv', 
            augment_color=True, 
            augment_brightness=True
        )
        
        assert extractor.method == 'hsv'
        assert extractor.augment_color
        assert extractor.augment_brightness
    
    def test_training_mode_augmentation(self):
        """Test that augmentation only applies in training mode."""
        extractor = AugmentedFeatureExtractor(method='hsv')
        rgb_input = torch.rand(2, 3, 16, 16)
        
        # Set to evaluation mode
        extractor.eval()
        color_eval, brightness_eval = extractor(rgb_input)
        
        # Set to training mode  
        extractor.train()
        # Note: Since augmentation uses random values, we can't test exact equality
        # But we can test that the function runs without error
        color_train, brightness_train = extractor(rgb_input)
        
        assert color_train.shape == color_eval.shape
        assert brightness_train.shape == brightness_eval.shape
    
    def test_augmentation_disabled(self):
        """Test extractor with augmentation disabled."""
        extractor = AugmentedFeatureExtractor(
            method='hsv',
            augment_color=False,
            augment_brightness=False
        )
        
        rgb_input = torch.rand(2, 3, 16, 16)
        extractor.train()
        
        color, brightness = extractor(rgb_input)
        
        # Should work without error even with augmentation disabled
        assert color.shape == (2, 2, 16, 16)
        assert brightness.shape == (2, 1, 16, 16)


class TestMultiModalFeatureExtractor:
    """Test multi-modal feature extraction."""
    
    def test_initialization(self):
        """Test MultiModalFeatureExtractor initialization."""
        extractor = MultiModalFeatureExtractor(rgb_method='hsv')
        assert hasattr(extractor, 'rgb_extractor')
        assert extractor.normalize
    
    def test_rgb_only_extraction(self):
        """Test extraction with only RGB input."""
        extractor = MultiModalFeatureExtractor()
        rgb_input = torch.rand(2, 3, 16, 16)
        
        color, brightness = extractor(rgb_input)
        
        assert color.shape == (2, 2, 16, 16)  # HSV color channels
        assert brightness.shape == (2, 1, 16, 16)  # HSV brightness channel
    
    def test_multimodal_extraction(self):
        """Test extraction with RGB and additional modality."""
        extractor = MultiModalFeatureExtractor()
        rgb_input = torch.rand(2, 3, 16, 16)
        additional_input = torch.rand(2, 1, 16, 16)  # Additional modality
        
        color, brightness = extractor(rgb_input, additional_input)
        
        assert color.shape == (2, 2, 16, 16)  # From RGB
        assert brightness.shape == (2, 1, 16, 16)  # From additional modality
        
        # Should use additional modality as brightness
        if extractor.normalize:
            # Can't directly compare due to normalization, but shapes should match
            assert brightness.shape == additional_input.shape
        else:
            assert torch.allclose(brightness, additional_input)


@pytest.fixture
def sample_rgb_batch():
    """Fixture providing sample RGB batch."""
    return torch.rand(4, 3, 32, 32)


def test_feature_extractor_batch_processing(sample_rgb_batch):
    """Test feature extraction with batch input."""
    extractor = FeatureExtractor(method='hsv')
    
    color, brightness = extractor(sample_rgb_batch)
    
    assert color.shape[0] == sample_rgb_batch.shape[0]  # Same batch size
    assert brightness.shape[0] == sample_rgb_batch.shape[0]


def test_device_compatibility():
    """Test that extractors work on different devices."""
    extractor = FeatureExtractor(method='hsv')
    
    # CPU test
    rgb_cpu = torch.rand(2, 3, 16, 16)
    color_cpu, brightness_cpu = extractor(rgb_cpu)
    
    assert color_cpu.device == rgb_cpu.device
    assert brightness_cpu.device == rgb_cpu.device
    
    # GPU test (if available)
    if torch.cuda.is_available():
        extractor_gpu = extractor.cuda()
        rgb_gpu = rgb_cpu.cuda()
        
        color_gpu, brightness_gpu = extractor_gpu(rgb_gpu)
        
        assert color_gpu.device == rgb_gpu.device
        assert brightness_gpu.device == rgb_gpu.device


def test_gradient_flow():
    """Test that gradients flow through feature extractors."""
    extractor = FeatureExtractor(method='hsv', normalize=False)
    
    rgb_input = torch.rand(2, 3, 16, 16, requires_grad=True)
    color, brightness = extractor(rgb_input)
    
    # Create a simple loss
    loss = color.sum() + brightness.sum()
    loss.backward()
    
    # Check that gradients exist
    assert rgb_input.grad is not None
    assert not torch.allclose(rgb_input.grad, torch.zeros_like(rgb_input.grad))


if __name__ == '__main__':
    pytest.main([__file__])
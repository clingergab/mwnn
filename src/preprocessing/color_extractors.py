"""Color and brightness feature extraction utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ColorSpaceConverter(nn.Module):
    """Convert between different color spaces."""
    
    def __init__(self, input_space: str = 'rgb', output_space: str = 'hsv'):
        super().__init__()
        self.input_space = input_space.lower()
        self.output_space = output_space.lower()
        
        # Supported color spaces
        self.supported_spaces = ['rgb', 'hsv', 'lab', 'yuv', 'hls']
        
        if self.input_space not in self.supported_spaces:
            raise ValueError(f"Input space must be one of {self.supported_spaces}")
        if self.output_space not in self.supported_spaces:
            raise ValueError(f"Output space must be one of {self.supported_spaces}")
    
    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV color space."""
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        
        max_rgb, _ = torch.max(rgb, dim=1)
        min_rgb, _ = torch.min(rgb, dim=1)
        diff = max_rgb - min_rgb
        
        # Value channel
        v = max_rgb
        
        # Saturation channel
        s = torch.where(max_rgb > 0, diff / max_rgb, torch.zeros_like(max_rgb))
        
        # Hue channel
        h = torch.zeros_like(max_rgb)
        
        # Red is max
        mask = (max_rgb == r) & (diff > 0)
        h[mask] = ((g[mask] - b[mask]) / diff[mask]) % 6
        
        # Green is max
        mask = (max_rgb == g) & (diff > 0)
        h[mask] = (b[mask] - r[mask]) / diff[mask] + 2
        
        # Blue is max
        mask = (max_rgb == b) & (diff > 0)
        h[mask] = (r[mask] - g[mask]) / diff[mask] + 4
        
        h = h / 6  # Normalize to [0, 1]
        
        return torch.stack([h, s, v], dim=1)
    
    def rgb_to_yuv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to YUV color space."""
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        
        return torch.stack([y, u, v], dim=1)
    
    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space (approximate)."""
        # This is a simplified version - for production use opencv or skimage
        xyz = self.rgb_to_xyz(rgb)
        return self.xyz_to_lab(xyz)
    
    def rgb_to_xyz(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to XYZ color space."""
        # Apply gamma correction
        rgb_linear = torch.where(rgb > 0.04045,
                                torch.pow((rgb + 0.055) / 1.055, 2.4),
                                rgb / 12.92)
        
        # Transformation matrix
        transform = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], device=rgb.device, dtype=rgb.dtype)
        
        # Apply transformation
        xyz = torch.einsum('ij,bjhw->bihw', transform, rgb_linear)
        return xyz
    
    def xyz_to_lab(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to LAB color space."""
        # Reference white (D65)
        ref_white = torch.tensor([0.95047, 1.00000, 1.08883], 
                                device=xyz.device, dtype=xyz.dtype)
        
        # Normalize by reference white
        xyz_norm = xyz / ref_white.view(1, 3, 1, 1)
        
        # Apply function
        fx = torch.where(xyz_norm > 0.008856,
                        torch.pow(xyz_norm, 1/3),
                        7.787 * xyz_norm + 16/116)
        
        L = 116 * fx[:, 1, :, :] - 16
        a = 500 * (fx[:, 0, :, :] - fx[:, 1, :, :])
        b = 200 * (fx[:, 1, :, :] - fx[:, 2, :, :])
        
        # Normalize to [0, 1]
        L = L / 100
        a = (a + 128) / 255
        b = (b + 128) / 255
        
        return torch.stack([L, a, b], dim=1)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image from input space to output space."""
        if self.input_space == self.output_space:
            return image
        
        # Convert to RGB first if needed
        if self.input_space != 'rgb':
            # Implement reverse conversions if needed
            raise NotImplementedError(f"Conversion from {self.input_space} not implemented")
        
        # Convert from RGB to target space
        if self.output_space == 'hsv':
            return self.rgb_to_hsv(image)
        elif self.output_space == 'yuv':
            return self.rgb_to_yuv(image)
        elif self.output_space == 'lab':
            return self.rgb_to_lab(image)
        else:
            raise NotImplementedError(f"Conversion to {self.output_space} not implemented")


class FeatureExtractor(nn.Module):
    """Extract color and brightness features from images."""
    
    def __init__(self, method: str = 'hsv', normalize: bool = True):
        super().__init__()
        self.method = method.lower()
        self.normalize = normalize
        
        # Initialize color space converter if needed
        if self.method in ['hsv', 'yuv', 'lab']:
            self.converter = ColorSpaceConverter('rgb', self.method)
        
    def extract_hsv_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color (HS) and brightness (V) from HSV space."""
        hsv = self.converter(image)
        
        # Color components: Hue and Saturation
        color = hsv[:, :2, :, :]
        
        # Brightness component: Value
        brightness = hsv[:, 2:3, :, :]
        
        return color, brightness
    
    def extract_yuv_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color (UV) and brightness (Y) from YUV space."""
        yuv = self.converter(image)
        
        # Brightness component: Y
        brightness = yuv[:, 0:1, :, :]
        
        # Color components: U and V
        color = yuv[:, 1:3, :, :]
        
        return color, brightness
    
    def extract_lab_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color (AB) and brightness (L) from LAB space."""
        lab = self.converter(image)
        
        # Brightness component: L
        brightness = lab[:, 0:1, :, :]
        
        # Color components: A and B
        color = lab[:, 1:3, :, :]
        
        return color, brightness
    
    def extract_rgb_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color and brightness directly from RGB."""
        # Simple brightness: average of RGB channels
        brightness = torch.mean(image, dim=1, keepdim=True)
        
        # Color: normalized RGB (remove brightness)
        color = image / (brightness + 1e-7)
        
        return color, brightness
    
    def extract_rgb_luminance_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract RGB color and ITU-R BT.709 luminance features (lossless approach)."""
        
        if image.shape[1] == 4:
            # Input is already RGB+Luminance, just split the channels
            color = image[:, :3, :, :]      # RGB channels (B, 3, H, W)
            brightness = image[:, 3:4, :, :]  # Luminance channel (B, 1, H, W)
            return color, brightness
            
        elif image.shape[1] == 3:
            # Input is RGB, compute luminance
            # ITU-R BT.709 standard luminance weights
            luminance_weights = torch.tensor([0.2126, 0.7152, 0.0722], 
                                            device=image.device, dtype=image.dtype)
            
            # Compute luminance using standard weights
            # image shape: (B, 3, H, W), weights shape: (3,)
            luminance = torch.sum(image * luminance_weights.view(1, 3, 1, 1), dim=1, keepdim=True)
            
            # Color pathway: Original RGB channels
            color = image  # (B, 3, H, W)
            
            # Brightness pathway: Luminance channel  
            brightness = luminance  # (B, 1, H, W)
            
            return color, brightness
        else:
            raise ValueError(f"Expected 3 or 4 channel input, got {image.shape[1]} channels")
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color and brightness features based on selected method."""
        if self.method == 'rgb_luminance':
            color, brightness = self.extract_rgb_luminance_features(image)
        elif self.method == 'hsv':
            color, brightness = self.extract_hsv_features(image)
        elif self.method == 'yuv':
            color, brightness = self.extract_yuv_features(image)
        elif self.method == 'lab':
            color, brightness = self.extract_lab_features(image)
        elif self.method == 'rgb':
            color, brightness = self.extract_rgb_features(image)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
        
        if self.normalize:
            # Normalize features to have zero mean and unit variance
            color = F.normalize(color.flatten(2), dim=2).view_as(color)
            brightness = F.normalize(brightness.flatten(2), dim=2).view_as(brightness)
        
        return color, brightness


class AugmentedFeatureExtractor(FeatureExtractor):
    """Feature extractor with data augmentation capabilities."""
    
    def __init__(self, method: str = 'hsv', normalize: bool = True,
                 augment_color: bool = True, augment_brightness: bool = True):
        super().__init__(method, normalize)
        self.augment_color = augment_color
        self.augment_brightness = augment_brightness
    
    def apply_color_augmentation(self, color: torch.Tensor) -> torch.Tensor:
        """Apply color-specific augmentations."""
        if not self.training or not self.augment_color:
            return color
        
        # Random hue shift (for HSV)
        if self.method == 'hsv' and torch.rand(1) > 0.5:
            hue_shift = torch.rand(1) * 0.1 - 0.05  # ±5% shift
            color[:, 0, :, :] = (color[:, 0, :, :] + hue_shift) % 1.0
        
        # Random saturation adjustment
        if torch.rand(1) > 0.5:
            sat_factor = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            if self.method == 'hsv':
                color[:, 1, :, :] = torch.clamp(color[:, 1, :, :] * sat_factor, 0, 1)
            else:
                color = color * sat_factor
        
        return color
    
    def apply_brightness_augmentation(self, brightness: torch.Tensor) -> torch.Tensor:
        """Apply brightness-specific augmentations."""
        if not self.training or not self.augment_brightness:
            return brightness
        
        # Random brightness adjustment
        if torch.rand(1) > 0.5:
            brightness_factor = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            brightness = torch.clamp(brightness * brightness_factor, 0, 1)
        
        # Random gamma correction
        if torch.rand(1) > 0.5:
            gamma = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            brightness = torch.pow(brightness, gamma)
        
        return brightness
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features with optional augmentation."""
        color, brightness = super().forward(image)
        
        # Apply augmentations
        color = self.apply_color_augmentation(color)
        brightness = self.apply_brightness_augmentation(brightness)
        
        return color, brightness


class MultiModalFeatureExtractor(nn.Module):
    """Extract features from multiple modalities (RGB + additional sensors)."""
    
    def __init__(self, rgb_method: str = 'hsv', normalize: bool = True):
        super().__init__()
        self.rgb_extractor = FeatureExtractor(rgb_method, normalize)
        self.normalize = normalize
    
    def forward(self, rgb: torch.Tensor, 
                additional_modality: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from RGB and optional additional modality."""
        # Extract from RGB
        color, brightness_rgb = self.rgb_extractor(rgb)
        
        if additional_modality is not None:
            # Use additional modality as refined brightness
            brightness = additional_modality
            
            if self.normalize:
                brightness = F.normalize(brightness.flatten(2), dim=2).view_as(brightness)
        else:
            brightness = brightness_rgb
        
        return color, brightness


def rgb_to_rgb_luminance(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to RGB+Luminance format.
    
    Args:
        rgb_tensor: Input RGB tensor of shape (B, 3, H, W)
        
    Returns:
        RGB+Luminance tensor of shape (B, 4, H, W) where the 4th channel
        is luminance calculated using ITU-R BT.709 standard weights
    """
    if rgb_tensor.shape[1] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {rgb_tensor.shape[1]}")
    
    # ITU-R BT.709 luminance weights
    r_weight = 0.2126
    g_weight = 0.7152
    b_weight = 0.0722
    
    # Extract RGB channels
    r_channel = rgb_tensor[:, 0:1, :, :]  # Shape: (B, 1, H, W)
    g_channel = rgb_tensor[:, 1:2, :, :]  # Shape: (B, 1, H, W)
    b_channel = rgb_tensor[:, 2:3, :, :]  # Shape: (B, 1, H, W)
    
    # Calculate luminance
    luminance = r_weight * r_channel + g_weight * g_channel + b_weight * b_channel
    
    # Concatenate RGB + Luminance
    rgb_luminance = torch.cat([rgb_tensor, luminance], dim=1)
    
    return rgb_luminance


def extract_color_brightness_from_rgb_luminance(rgb_luminance_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract color and brightness pathways from RGB+Luminance tensor.
    
    Args:
        rgb_luminance_tensor: Input RGB+Luminance tensor of shape (B, 4, H, W)
        
    Returns:
        Tuple of (color_pathway, brightness_pathway):
        - color_pathway: RGB channels (B, 3, H, W)
        - brightness_pathway: Luminance channel (B, 1, H, W)
    """
    if rgb_luminance_tensor.shape[1] != 4:
        raise ValueError(f"Expected 4 channels (RGB+L), got {rgb_luminance_tensor.shape[1]}")
    
    color_pathway = rgb_luminance_tensor[:, :3, :, :]      # RGB channels
    brightness_pathway = rgb_luminance_tensor[:, 3:4, :, :] # Luminance channel
    
    return color_pathway, brightness_pathway

def rgb_to_rgb_yuv_brightness(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to RGB+YUV-Brightness format (alternative to RGB+Luminance).
    
    Args:
        rgb_tensor: Input RGB tensor of shape (B, 3, H, W)
        
    Returns:
        4-channel tensor with RGB + YUV Y-channel: (B, 4, H, W)
        Channels: [R, G, B, Y_yuv]
    """
    if rgb_tensor.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB input, got {rgb_tensor.shape[1]} channels")
    
    # Extract RGB channels
    r, g, b = rgb_tensor[:, 0, :, :], rgb_tensor[:, 1, :, :], rgb_tensor[:, 2, :, :]
    
    # Calculate YUV Y-channel (ITU-R BT.601 standard)
    # Y = 0.299*R + 0.587*G + 0.114*B
    y_channel = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Combine RGB + YUV Y-channel
    rgb_yuv_tensor = torch.stack([r, g, b, y_channel], dim=1)
    
    return rgb_yuv_tensor

def extract_color_brightness_from_rgb_yuv(rgb_yuv_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract color and brightness pathways from RGB+YUV-Brightness data.
    
    Args:
        rgb_yuv_tensor: 4-channel tensor (B, 4, H, W) with [R, G, B, Y_yuv]
        
    Returns:
        Tuple of (color_pathway, brightness_pathway):
        - color_pathway: RGB channels (B, 3, H, W)
        - brightness_pathway: YUV Y channel (B, 1, H, W)
    """
    if rgb_yuv_tensor.shape[1] != 4:
        raise ValueError(f"Expected 4-channel RGB+YUV input, got {rgb_yuv_tensor.shape[1]} channels")
    
    # Split into color (RGB) and brightness (YUV Y) pathways
    color_pathway = rgb_yuv_tensor[:, :3, :, :]      # RGB channels
    brightness_pathway = rgb_yuv_tensor[:, 3:4, :, :] # YUV Y channel
    
    return color_pathway, brightness_pathway
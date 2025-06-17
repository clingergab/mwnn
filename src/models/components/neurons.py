"""Shared neuron components for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
from typing import Tuple, Union
from abc import ABC, abstractmethod


class BaseMultiWeightNeuron(nn.Module, ABC):
    """Abstract base class for multi-weight neurons."""
    
    def __init__(self, input_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.use_bias = bias
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through the neuron."""
        pass
    
    def extract_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color and brightness components from inputs."""
        if len(inputs.shape) == 4:  # Image input
            # Convert RGB to HSV-like representation
            rgb = inputs[:, :3]  # Assume first 3 channels are RGB
            
            # Extract brightness (value channel from HSV)
            brightness = torch.max(rgb, dim=1, keepdim=True)[0]
            
            # Extract color information (simplified - could use proper HSV conversion)
            color = rgb / (brightness + 1e-7)  # Normalize by brightness
            
            return color, brightness
        else:  # Feature vector input
            # Assume features are already separated or use learned separation
            mid = inputs.shape[1] // 2
            return inputs[:, :mid], inputs[:, mid:]


class MultiChannelNeuron(BaseMultiWeightNeuron):
    """Basic multi-channel neuron with separate outputs for each modality."""
    
    def __init__(self, input_size: int, activation: str = 'relu', bias: bool = True):
        super().__init__(input_size, bias)
        
        # Separate weights for color and brightness
        self.color_weights = nn.Parameter(torch.randn(input_size) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(input_size) * 0.01)
        
        if self.use_bias:
            self.color_bias = nn.Parameter(torch.zeros(1))
            self.brightness_bias = nn.Parameter(torch.zeros(1))
        
        # Activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, color_inputs: torch.Tensor, brightness_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing separate color and brightness outputs."""
        # Color processing
        color_output = torch.matmul(color_inputs, self.color_weights)
        if self.use_bias:
            color_output = color_output + self.color_bias
        color_output = self.activation(color_output)
        
        # Brightness processing
        brightness_output = torch.matmul(brightness_inputs, self.brightness_weights)
        if self.use_bias:
            brightness_output = brightness_output + self.brightness_bias
        brightness_output = self.activation(brightness_output)
        
        return color_output, brightness_output


class PracticalMultiWeightNeuron(BaseMultiWeightNeuron):
    """Single-output neuron with specialized weights for different features."""
    
    def __init__(self, input_size: int, activation: str = 'relu', bias: bool = True):
        super().__init__(input_size, bias)
        
        # For feature vector inputs, each pathway gets half the input features
        # For image inputs, this would be handled differently in extract_features
        feature_size = input_size // 2
        
        # Specialized weights
        self.color_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing single output from multiple weights."""
        color_components, brightness_components = self.extract_features(inputs)
        
        # Separate processing
        color_contribution = torch.matmul(color_components.flatten(1), self.color_weights)
        brightness_contribution = torch.matmul(brightness_components.flatten(1), self.brightness_weights)
        
        # Single output combining both contributions
        output = color_contribution + brightness_contribution
        
        if self.use_bias:
            output = output + self.bias
        
        output = self.activation(output)
        
        return output


class ContinuousIntegrationNeuron(BaseMultiWeightNeuron):
    """Neuron with continuous learnable integration (Option 1B from DESIGN.md)."""
    
    def __init__(self, input_size: int, activation: str = 'relu', bias: bool = True):
        super().__init__(input_size, bias)
        
        # For feature vector inputs, each pathway gets half the input features
        feature_size = input_size // 2
        
        # Separate processing weights
        self.color_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        
        # Learnable integration weights that get updated by training
        self.integration_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        if self.use_bias:
            self.color_bias = nn.Parameter(torch.zeros(1))
            self.brightness_bias = nn.Parameter(torch.zeros(1))
            self.integration_bias = nn.Parameter(torch.zeros(1))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass producing separate outputs and learnable integration."""
        color_components, brightness_components = self.extract_features(inputs)
        
        # Separate processing
        color_output = torch.matmul(color_components.flatten(1), self.color_weights)
        brightness_output = torch.matmul(brightness_components.flatten(1), self.brightness_weights)
        
        if self.use_bias:
            color_output = color_output + self.color_bias
            brightness_output = brightness_output + self.brightness_bias
        
        color_output = self.activation(color_output)
        brightness_output = self.activation(brightness_output)
        
        # Learnable integration (gets updated by backprop)
        integrated_output = (self.integration_weights[0] * color_output + 
                           self.integration_weights[1] * brightness_output)
        
        if self.use_bias:
            integrated_output = integrated_output + self.integration_bias
        
        integrated_output = self.activation(integrated_output)
        
        return color_output, brightness_output, integrated_output


class CrossModalMultiWeightNeuron(BaseMultiWeightNeuron):
    """Neuron with cross-modal influence (Option 1C from DESIGN.md)."""
    
    def __init__(self, input_size: int, cross_influence: float = 0.1, 
                 activation: str = 'relu', bias: bool = True):
        super().__init__(input_size, bias)
        
        # For feature vector inputs, each pathway gets half the input features
        feature_size = input_size // 2
        
        # Direct pathway weights
        self.color_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        
        # Cross-modal influence weights (smaller magnitude)
        self.cross_weights_cb = nn.Parameter(torch.randn(feature_size) * 0.01 * cross_influence)
        self.cross_weights_bc = nn.Parameter(torch.randn(feature_size) * 0.01 * cross_influence)
        
        if self.use_bias:
            self.color_bias = nn.Parameter(torch.zeros(1))
            self.brightness_bias = nn.Parameter(torch.zeros(1))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, color_input: torch.Tensor, 
                brightness_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-modal influence."""
        # Extract features from each input (each gets half the feature size)
        color_features, _ = self.extract_features(color_input)
        _, brightness_features = self.extract_features(brightness_input)
        
        # Direct pathways (maintain separation)
        color_direct = torch.matmul(color_features.flatten(1), self.color_weights)
        brightness_direct = torch.matmul(brightness_features.flatten(1), self.brightness_weights)
        
        # Cross-influence (interaction)
        color_from_brightness = torch.matmul(brightness_features.flatten(1), self.cross_weights_cb)
        brightness_from_color = torch.matmul(color_features.flatten(1), self.cross_weights_bc)
        
        # Combine direct and cross-influence
        color_output = color_direct + color_from_brightness
        brightness_output = brightness_direct + brightness_from_color
        
        if self.use_bias:
            color_output = color_output + self.color_bias
            brightness_output = brightness_output + self.brightness_bias
        
        # Apply activation
        color_output = self.activation(color_output)
        brightness_output = self.activation(brightness_output)
        
        return color_output, brightness_output


class AdaptiveMultiWeightNeuron(BaseMultiWeightNeuron):
    """Neuron with adaptive output selection (Option 2B from DESIGN.md)."""
    
    def __init__(self, input_size: int, activation: str = 'relu', bias: bool = True):
        super().__init__(input_size, bias)
        
        # For feature vector inputs, each pathway gets half the input features
        feature_size = input_size // 2
        
        # Processing weights
        self.color_processing_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        self.brightness_processing_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        
        # Output selector network
        self.output_selector = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive selection."""
        color_components, brightness_components = self.extract_features(inputs)
        
        # Process on separate pathways
        color_signal = torch.matmul(color_components.flatten(1), self.color_processing_weights)
        brightness_signal = torch.matmul(brightness_components.flatten(1), self.brightness_processing_weights)
        
        color_signal = self.activation(color_signal)
        brightness_signal = self.activation(brightness_signal)
        
        # Learn which signals to pass forward
        output_weights = self.output_selector(inputs.flatten(1))
        
        # Weighted combination - could emphasize color, brightness, or both
        output = output_weights[:, 0] * color_signal + output_weights[:, 1] * brightness_signal
        
        if self.use_bias:
            output = output + self.bias
        
        return output


class AttentionMultiWeightNeuron(BaseMultiWeightNeuron):
    """Neuron with attention-based cross-modal processing."""
    
    def __init__(self, input_size: int, attention_dim: int = 32, 
                 activation: str = 'relu', bias: bool = True):
        super().__init__(input_size, bias)
        
        # For feature vector inputs, each pathway gets half the input features
        feature_size = input_size // 2
        
        # Direct processing weights
        self.color_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        
        # Cross-modal attention weights
        self.color_to_brightness_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        self.brightness_to_color_weights = nn.Parameter(torch.randn(feature_size) * 0.01)
        
        if self.use_bias:
            self.color_bias = nn.Parameter(torch.zeros(1))
            self.brightness_bias = nn.Parameter(torch.zeros(1))
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention mechanism."""
        color_components, brightness_components = self.extract_features(inputs)
        
        # Direct processing paths
        color_direct = torch.matmul(color_components.flatten(1), self.color_weights)
        brightness_direct = torch.matmul(brightness_components.flatten(1), self.brightness_weights)
        
        # Cross-modal attention computation  
        brightness_to_color_attention = torch.matmul(brightness_components.flatten(1), self.brightness_to_color_weights)
        color_to_brightness_attention = torch.matmul(color_components.flatten(1), self.color_to_brightness_weights)
        
        # Apply cross-modal influence
        color_output = color_direct + brightness_to_color_attention
        brightness_output = brightness_direct + color_to_brightness_attention
        
        if self.use_bias:
            color_output = color_output + self.color_bias
            brightness_output = brightness_output + self.brightness_bias
        
        # Apply activation
        color_output = self.activation(color_output)
        brightness_output = self.activation(brightness_output)
        
        return color_output, brightness_output
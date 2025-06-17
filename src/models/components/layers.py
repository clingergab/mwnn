"""Shared layer components for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiWeightLinear(nn.Module):
    """Base class for multi-weight linear layers."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias


class MultiChannelLinear(MultiWeightLinear):
    """Linear layer using multi-channel processing."""
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = 'relu', bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # Weight matrices - expects separated inputs where each pathway has in_features
        self.color_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        if self.use_bias:
            self.color_bias = nn.Parameter(torch.zeros(out_features))
            self.brightness_bias = nn.Parameter(torch.zeros(out_features))
        
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
        """Forward pass through multi-channel linear layer."""
        # Process color
        color_output = F.linear(color_input, self.color_weights, 
                               self.color_bias if self.use_bias else None)
        color_output = self.activation(color_output)
        
        # Process brightness
        brightness_output = F.linear(brightness_input, self.brightness_weights,
                                    self.brightness_bias if self.use_bias else None)
        brightness_output = self.activation(brightness_output)
        
        return color_output, brightness_output


class PracticalMultiWeightLinear(MultiWeightLinear):
    """Single-output linear layer with specialized weights."""
    
    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu', bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # Specialized weights - input features are not pre-separated
        self.color_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
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
    
    def extract_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract color and brightness features separately."""
        # Process same input with different weights to get separate features
        color_features = F.linear(inputs, self.color_weights)
        brightness_features = F.linear(inputs, self.brightness_weights)
        
        # Apply activation
        color_features = self.activation(color_features)
        brightness_features = self.activation(brightness_features)
        
        return color_features, brightness_features
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass producing single output."""
        # Process same input with different weights
        color_contribution = F.linear(inputs, self.color_weights)
        brightness_contribution = F.linear(inputs, self.brightness_weights)
        
        # Single output combining both contributions
        output = color_contribution + brightness_contribution
        
        if self.use_bias:
            output = output + self.bias
        
        output = self.activation(output)
        
        return output


class AdaptiveMultiWeightLinear(MultiWeightLinear):
    """Linear layer with adaptive output selection."""
    
    def __init__(self, in_features: int, out_features: int,
                 activation: str = 'relu', bias: bool = True):
        super().__init__(in_features, out_features, bias)
        
        # Processing weights - process full input with different weights
        self.color_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
        # Output selector network
        self.output_selector = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_features * 2)
        )
        
        if self.use_bias:
            self.color_bias = nn.Parameter(torch.zeros(out_features))
            self.brightness_bias = nn.Parameter(torch.zeros(out_features))
        
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
        # Process full input with separate weights
        color_signal = F.linear(inputs, self.color_weights,
                               self.color_bias if self.use_bias else None)
        brightness_signal = F.linear(inputs, self.brightness_weights,
                                    self.brightness_bias if self.use_bias else None)
        
        color_signal = self.activation(color_signal)
        brightness_signal = self.activation(brightness_signal)
        
        # Learn which signals to pass forward
        selector_output = self.output_selector(inputs)
        selector_output = selector_output.view(-1, self.out_features, 2)
        output_weights = F.softmax(selector_output, dim=-1)
        
        # Weighted combination
        output = (output_weights[..., 0] * color_signal + 
                 output_weights[..., 1] * brightness_signal)
        
        return output
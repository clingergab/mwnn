#!/usr/bin/env python3
"""Simple test for MultiChannelNeuron to debug the hanging issue."""

import torch
import torch.nn as nn
from typing import Tuple

class SimpleMultiChannelNeuron(nn.Module):
    """Simplified version of MultiChannelNeuron for testing."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        
        # Separate weights for color and brightness
        self.color_weights = nn.Parameter(torch.randn(input_size) * 0.01)
        self.brightness_weights = nn.Parameter(torch.randn(input_size) * 0.01)
        self.color_bias = nn.Parameter(torch.zeros(1))
        self.brightness_bias = nn.Parameter(torch.zeros(1))
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, color_inputs: torch.Tensor, brightness_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing separate color and brightness outputs."""
        print("Starting forward pass...")
        
        # Color processing
        print("Processing color...")
        color_output = torch.matmul(color_inputs, self.color_weights)
        color_output = color_output + self.color_bias
        color_output = self.activation(color_output)
        print(f"Color output shape: {color_output.shape}")
        
        # Brightness processing
        print("Processing brightness...")
        brightness_output = torch.matmul(brightness_inputs, self.brightness_weights)
        brightness_output = brightness_output + self.brightness_bias
        brightness_output = self.activation(brightness_output)
        print(f"Brightness output shape: {brightness_output.shape}")
        
        print("Forward pass complete")
        return color_output, brightness_output

def main():
    print("=== Testing Simple MultiChannelNeuron ===")
    
    # Create neuron
    print("Creating neuron...")
    neuron = SimpleMultiChannelNeuron(input_size=64)
    print("✓ Neuron created")
    
    # Create inputs
    print("Creating inputs...")
    color_input = torch.randn(8, 64)
    brightness_input = torch.randn(8, 64)
    print("✓ Inputs created")
    
    # Test forward pass
    print("Testing forward pass...")
    result = neuron(color_input, brightness_input)
    print(f"✓ Result: {[r.shape for r in result]}")
    
    print("=== Test successful! ===")

if __name__ == "__main__":
    main()

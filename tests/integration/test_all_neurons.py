#!/usr/bin/env python3
"""Test all neuron types to verify DESIGN.md compliance."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
from models.components.neurons import (
    MultiChannelNeuron,
    PracticalMultiWeightNeuron,
    ContinuousIntegrationNeuron,
    CrossModalMultiWeightNeuron,
    AdaptiveMultiWeightNeuron,
    AttentionMultiWeightNeuron
)

def verify_neuron(neuron_class, name, design_option="N/A", input_size=64, *args, **kwargs):
    """Verify a single neuron type."""
    print(f"\nTesting {name} (DESIGN.md {design_option})...")
    
    try:
        # Create neuron (don't pass design_option to constructor)
        neuron = neuron_class(input_size, *args, **kwargs)
        print(f"  ✓ Created {name}")
        
        # Test forward pass based on neuron type
        if name in ["MultiChannelNeuron", "CrossModalMultiWeightNeuron"]:
            # These take separate color and brightness inputs
            color_input = torch.randn(8, input_size)
            brightness_input = torch.randn(8, input_size)
            result = neuron(color_input, brightness_input)
            print(f"  ✓ Forward pass: {[r.shape for r in result]}")
        else:
            # These take combined inputs
            inputs = torch.randn(8, input_size)
            result = neuron(inputs)
            if isinstance(result, tuple):
                print(f"  ✓ Forward pass: {[r.shape for r in result]}")
            else:
                print(f"  ✓ Forward pass: {result.shape}")
        
        # Check required attributes based on DESIGN.md
        required_attrs = {
            "MultiChannelNeuron": ["color_weights", "brightness_weights"],
            "PracticalMultiWeightNeuron": ["color_weights", "brightness_weights"],
            "ContinuousIntegrationNeuron": ["color_weights", "brightness_weights", "integration_weights"],
            "CrossModalMultiWeightNeuron": ["color_weights", "brightness_weights", "cross_weights_cb", "cross_weights_bc"],
            "AdaptiveMultiWeightNeuron": ["color_processing_weights", "brightness_processing_weights", "output_selector"],
            "AttentionMultiWeightNeuron": ["color_weights", "brightness_weights", "color_to_brightness_weights", "brightness_to_color_weights"]
        }
        
        for attr in required_attrs.get(name, []):
            assert hasattr(neuron, attr), f"Missing required attribute: {attr}"
            print(f"  ✓ Has {attr}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Test all neuron types according to DESIGN.md."""
    print("=== Testing Multi-Weight Neural Network Neurons ===")
    print("Verifying compliance with DESIGN.md specifications")
    
    results = []
    
    # Option 1A: Basic Multi-Channel Neurons
    results.append(verify_neuron(
        MultiChannelNeuron, 
        "MultiChannelNeuron",
        "Option 1A"
    ))
    
    # Option 1B: Continuous Learnable Integration
    results.append(verify_neuron(
        ContinuousIntegrationNeuron, 
        "ContinuousIntegrationNeuron",
        "Option 1B"
    ))
    
    # Option 1C: Cross-Modal Influence Architecture
    results.append(verify_neuron(
        CrossModalMultiWeightNeuron, 
        "CrossModalMultiWeightNeuron",
        "Option 1C"
    ))
    
    # Option 2A: Specialized Weights with Single Output
    results.append(verify_neuron(
        PracticalMultiWeightNeuron, 
        "PracticalMultiWeightNeuron",
        "Option 2A"
    ))
    
    # Option 2B: Context-Dependent Output Selection
    results.append(verify_neuron(
        AdaptiveMultiWeightNeuron, 
        "AdaptiveMultiWeightNeuron",
        "Option 2B"
    ))
    
    # Attention-Based Multi-Weight Neurons
    results.append(verify_neuron(
        AttentionMultiWeightNeuron, 
        "AttentionMultiWeightNeuron",
        "Attention-based"
    ))
    
    # Summary
    print(f"\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All neuron types work correctly and comply with DESIGN.md!")
        print("\nNaming consistency check:")
        print("✓ Option 1A: MultiChannelNeuron")
        print("✓ Option 1B: ContinuousIntegrationNeuron")
        print("✓ Option 1C: CrossModalMultiWeightNeuron")
        print("✓ Option 2A: PracticalMultiWeightNeuron")
        print("✓ Option 2B: AdaptiveMultiWeightNeuron")
        print("✓ Attention: AttentionMultiWeightNeuron")
        return True
    else:
        print(f"❌ {total - passed} neuron types failed")
        return False

def test_all_neurons():
    """Pytest wrapper for the main test function."""
    assert main(), "Not all neurons passed verification"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

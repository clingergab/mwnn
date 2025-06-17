# Tests Directory Structure

This directory contains all tests for the Multi-Weight Neural Networks project, organized by category.

## Directory Structure

```
tests/
├── __init__.py                 # Test package initialization
├── components/                 # Unit tests for individual components
│   ├── test_neurons.py        # Tests for all neuron types
│   ├── test_layers.py         # Tests for multi-weight layers
│   └── test_blocks.py         # Tests for building blocks
├── models/                    # Unit tests for complete models
│   ├── test_multi_channel_model.py      # Multi-channel model tests
│   ├── test_continuous_integration_model.py
│   ├── test_cross_modal_model.py
│   ├── test_attention_based_model.py
│   └── test_single_output_model.py
├── preprocessing/             # Tests for data preprocessing
│   ├── test_data_loaders.py   # Data loading and batching tests
│   └── test_color_extractors.py # HSV feature extraction tests
├── training/                  # Tests for training components
│   ├── test_trainer.py        # Training loop tests
│   ├── test_losses.py         # Loss function tests
│   └── test_metrics.py        # Metric calculation tests
├── utils/                     # Tests for utility functions
│   └── test_visualization.py  # Visualization utility tests
├── integration/               # Integration and end-to-end tests
│   ├── test_multi_channel_comprehensive.py  # Full model integration
│   ├── test_neuron_integration.py          # Neuron-model integration
│   ├── test_all_neurons.py                 # All neurons compliance
│   └── test_multi_channel_focused.py       # Focused testing
└── verification/              # Verification and compliance scripts
    ├── verify_option_1a.py    # DESIGN.md Option 1A verification
    ├── verify_final.py        # Final implementation verification
    └── debug_neuron.py         # Debug utilities
```

## Test Categories

### 1. Unit Tests
- **Components**: Test individual neurons, layers, and blocks in isolation
- **Models**: Test complete model architectures with mocked data
- **Preprocessing**: Test data loading and feature extraction
- **Training**: Test loss functions, metrics, and training procedures
- **Utils**: Test utility and helper functions

### 2. Integration Tests
- **Comprehensive**: End-to-end testing of complete workflows
- **Neuron Integration**: Test neuron integration with models
- **All Neurons**: Verify all neuron types meet DESIGN.md specifications
- **Focused**: Targeted tests for specific functionality

### 3. Verification Scripts
- **Option 1A Verification**: Verify DESIGN.md Option 1A compliance
- **Final Verification**: Final implementation status check
- **Debug Utilities**: Tools for debugging and troubleshooting

## Running Tests

### Using the organized test runner:
```bash
# Run all tests
python3 run_organized_tests.py

# Run specific categories
python3 run_organized_tests.py --category unit
python3 run_organized_tests.py --category integration
python3 run_organized_tests.py --category verification

# Run with verbose output
python3 run_organized_tests.py --verbose

# Run specific test types
python3 run_organized_tests.py --category components
python3 run_organized_tests.py --category models
```

### Using pytest directly:
```bash
# Run all unit tests
python3 -m pytest tests/components/ tests/models/ tests/preprocessing/ tests/training/ -v

# Run specific test files
python3 -m pytest tests/components/test_neurons.py -v
python3 -m pytest tests/models/test_multi_channel_model.py -v

# Run integration tests (as scripts)
python3 tests/integration/test_multi_channel_comprehensive.py
python3 tests/integration/test_neuron_integration.py

# Run verification scripts
python3 tests/verification/verify_option_1a.py
python3 tests/verification/verify_final.py
```

## Test Standards

All tests follow these standards:

1. **Naming**: Test files start with `test_` and test functions start with `test_`
2. **Documentation**: Each test has a clear docstring explaining what it tests
3. **Isolation**: Unit tests don't depend on external resources
4. **Coverage**: Tests cover both success and failure cases
5. **Performance**: Integration tests verify computational efficiency
6. **Compliance**: Verification tests ensure DESIGN.md adherence

## Key Test Files

### Critical for Option 1A (MultiChannelNeuron):
- `tests/components/test_neurons.py` - Core neuron functionality
- `tests/models/test_multi_channel_model.py` - Model integration
- `tests/integration/test_neuron_integration.py` - End-to-end integration
- `tests/verification/verify_option_1a.py` - DESIGN.md compliance

### Important for Overall Project:
- `tests/integration/test_multi_channel_comprehensive.py` - Complete workflow
- `tests/integration/test_all_neurons.py` - All neuron types
- `tests/components/test_*` - All component tests

## Success Criteria

The project passes all tests when:
- ✅ All unit tests pass (51+ tests)
- ✅ All integration tests pass
- ✅ All verification scripts confirm compliance
- ✅ Option 1A MultiChannelNeuron fully verified
- ✅ All neuron types implement DESIGN.md specifications

Current Status: **ALL TESTS PASSING** ✅

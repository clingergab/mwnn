# End-to-End Tests for MWNN

This directory contains comprehensive end-to-end tests that validate the complete MWNN pipeline from data loading to model training, evaluation, and saving/loading.

## Test Files

### 🎭 `test_end_to_end_demo.py`
**Quick synthetic data demo** - Fastest way to validate the API
- Uses synthetic data (no external datasets required)
- Tests basic API functionality
- Demonstrates different model configurations
- Shows error handling
- **Runtime**: ~30 seconds

```bash
python test_end_to_end_demo.py
```

### 🔢 `test_end_to_end_mnist.py`
**MNIST dataset validation** - Real data, quick training
- Requires MNIST CSV data in `data/MNIST/`
- Tests dual pathway (RGB + brightness) processing
- Multiple model configurations (shallow, medium, deep)
- Save/load functionality
- **Runtime**: ~2-5 minutes

```bash
python test_end_to_end_mnist.py
```

### 🖼️ `test_end_to_end_imagenet.py`
**ImageNet dataset validation** - Full-scale realistic testing
- Requires ImageNet data in `data/ImageNet-1K/`
- Tests with small subsets for efficiency
- Validates production-ready pipeline
- Cross-dataset compatibility tests
- **Runtime**: ~5-10 minutes (with subsets)

```bash
python test_end_to_end_imagenet.py
```

### 🎯 `test_end_to_end_comprehensive.py`
**Complete test suite runner** - Runs all tests with reporting
- Automatically detects available datasets
- Runs appropriate tests based on data availability
- Comprehensive reporting and timing
- Command-line options for selective testing

```bash
# Run all available tests
python test_end_to_end_comprehensive.py

# Run only MNIST tests
python test_end_to_end_comprehensive.py --mnist-only

# Run only ImageNet tests
python test_end_to_end_comprehensive.py --imagenet-only

# Quick validation mode
python test_end_to_end_comprehensive.py --quick
```

## What These Tests Validate

### 🔧 **API Functionality**
- ✅ Model creation with different configurations
- ✅ Data loading for multiple datasets
- ✅ Training pipeline with progress tracking
- ✅ Evaluation and metrics calculation
- ✅ Model saving and loading
- ✅ Prediction generation
- ✅ Error handling and validation

### 📊 **Data Processing**
- ✅ Dual pathway processing (RGB + brightness)
- ✅ Different input sizes and formats
- ✅ Batch processing and data loaders
- ✅ Multiple dataset compatibility
- ✅ Subset handling for efficient testing

### 🧠 **Model Training**
- ✅ Multiple model depths (shallow, medium, deep)
- ✅ Different channel configurations
- ✅ Various class counts (2, 5, 10, 1000)
- ✅ Learning rate and optimization
- ✅ Checkpoint saving during training

### 💾 **Persistence**
- ✅ Model state saving and loading
- ✅ Configuration preservation
- ✅ Training state restoration
- ✅ Cross-platform compatibility

## Dataset Requirements

### MNIST (Required for MNIST tests)
```
data/MNIST/
├── mnist_train.csv
└── mnist_test.csv
```

### ImageNet-1K (Optional for ImageNet tests)
```
data/ImageNet-1K/
├── val_images/
│   ├── n01440764/
│   └── ...
└── ILSVRC2013_devkit/
    └── ...
```

## Usage Examples

### Quick API Validation
```bash
# Fastest test - no external data needed
python test_end_to_end_demo.py
```

### Development Testing
```bash
# Quick MNIST validation during development
python test_end_to_end_mnist.py
```

### Full Validation
```bash
# Complete test suite
python test_end_to_end_comprehensive.py
```

### CI/CD Integration
```bash
# Automated testing with timeout and reporting
python test_end_to_end_comprehensive.py --quick 2>&1 | tee test_results.log
```

## Test Output

Each test provides detailed output including:
- 📊 Dataset loading confirmation
- 🧠 Model architecture summary
- 🎯 Training progress and metrics
- 📈 Evaluation results
- ⏱️ Timing information
- ✅/❌ Pass/fail status for each component

### Example Output
```
🧪 Running MNIST End-to-End Tests
==================================================

=== Testing MNIST Training (Basic) ===
Loading MNIST data from /path/to/data/MNIST
Subset size: 500, Batch size: 16, Epochs: 2
Train batches: 32, Test batches: 7

Creating MWNN model...
MWNN Model Summary:
  Classes: 10
  Depth: shallow
  Base Channels: 32
  Device: cuda:0
  Total Parameters: 45,678
  Trainable Parameters: 45,678

Starting training...
Epoch 1/2: 100%|██████████| 32/32 [00:15<00:00, 2.1it/s]
Training completed!
Final validation accuracy: 87.32%

✅ MNIST basic training test passed!
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the right directory
cd /path/to/mwnn/tests/integration/
python test_end_to_end_demo.py
```

**Missing Data**
```bash
# MNIST tests need CSV files
ls ../../data/MNIST/
# Should show: mnist_train.csv, mnist_test.csv

# ImageNet tests need image directory
ls ../../data/ImageNet-1K/
# Should show: val_images/, ILSVRC2013_devkit/
```

**Memory Issues**
- Use smaller batch sizes in the test files
- Reduce subset_size parameters
- Run tests individually instead of comprehensive suite

**Timeout Issues**
- Reduce epochs in test configurations
- Use smaller datasets/subsets
- Run on GPU for faster processing

## Contributing

When adding new end-to-end tests:

1. **Follow naming convention**: `test_end_to_end_*.py`
2. **Include comprehensive docstrings**
3. **Add timing and progress indicators**
4. **Test both success and failure cases**
5. **Provide clear error messages**
6. **Update this README with new test descriptions**

### Test Template
```python
def test_new_feature():
    """Test description and purpose."""
    print("\n=== Testing New Feature ===")
    
    try:
        # Setup
        # ... test implementation ...
        
        print("✅ New feature test passed!")
        return True
        
    except Exception as e:
        print(f"❌ New feature test failed: {e}")
        return False
```

## Performance Benchmarks

Typical runtimes on different hardware:

| Test Suite | CPU (M1) | GPU (RTX 3080) | CPU (Intel i7) |
|-----------|----------|----------------|----------------|
| Demo      | 30s      | 15s            | 45s            |
| MNIST     | 3m       | 1m             | 5m             |
| ImageNet  | 8m       | 3m             | 15m            |
| Full      | 12m      | 5m             | 20m            |

*Note: Times include small subsets for validation, not full training.*

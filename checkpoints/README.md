# Checkpoints Directory

This directory contains model checkpoints and experimental results.

## Structure

- `*.pth` files - Trained model weights (ignored in git due to size)
- `*.json` files - Experimental results and metrics (tracked in git)

## Files

- `batch_size_optimization_results.json` - Results from batch size optimization experiments
- `imagenet_continuous_integration_results.json` - ImageNet CI test results
- `mnist_csv_mwnn_results.json` - MNIST CSV format training results

## Usage

Model checkpoints are automatically saved here during training.
Results files contain metrics and can be used for analysis and reporting.

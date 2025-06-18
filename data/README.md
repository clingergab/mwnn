# Data Directory

This directory contains datasets used for training and testing MWNN models.

## Structure

- `MNIST/` - MNIST dataset files (CSV format)
- `ImageNet-1K/` - ImageNet-1K dataset (training and validation images)
  - `train_images_*/` - Training image directories (ignored in git)
  - `val_images/` - Validation images (ignored in git)
  - `ILSVRC2013_devkit/` - Development kit and metadata

## Usage

The MWNN data loaders automatically detect and load data from these directories.
Large image files are ignored in git - download datasets separately.

## Dataset Sources

- **MNIST**: Available via torchvision or as CSV from Kaggle
- **ImageNet-1K**: Download from official ImageNet website or Kaggle

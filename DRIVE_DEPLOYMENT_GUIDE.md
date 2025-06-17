# MWNN Google Drive Deployment Guide

## 🎯 Overview

This guide shows how to deploy and run the Multi-Weight Neural Network (MWNN) project on Google Colab using Google Drive for persistent storage. All data, models, checkpoints, and results are stored directly in your Google Drive.

## 🏗️ Setup Process

### 1. Prepare Your Google Drive

1. **Upload MWNN Project**:
   - Compress your local MWNN project folder
   - Upload to Google Drive (or use the provided `mwnn_colab_package.tar.gz`)
   - Extract in Drive to create a project folder

2. **Recommended Drive Structure**:
   ```
   /MyDrive/
   └── mwnn-project/              # Your project folder (any name)
       ├── src/                   # Source code
       ├── configs/               # Configuration files  
       ├── train_deep_colab.py    # Training scripts
       ├── setup_imagenet_colab.py
       ├── optimize_batch_sizes.py
       ├── test_*.py              # Test scripts
       ├── requirements_colab.txt # Dependencies
       └── MWNN_Colab_Training.ipynb  # Main notebook
   ```

### 2. Launch Google Colab

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload Notebook**: Upload `MWNN_Colab_Training.ipynb` from your Drive folder
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)

### 3. Configure Project Path

In the notebook's navigation cell, update the project path:

```python
# Update this to match your Drive folder
PROJECT_PATH = "/content/drive/MyDrive/your-folder-name"  # 👈 CHANGE THIS
```

## 🚀 Running the Training

### Step-by-Step Execution

1. **Mount Google Drive** (Cell 1):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Navigate to Project** (Cell 2):
   - Update `PROJECT_PATH` to your actual folder
   - Verify project structure

3. **Download ImageNet** (Cell 3):
   - Data downloads directly to your Drive
   - Persists between sessions

4. **Run Experiments** (Cells 4+):
   - MNIST validation
   - Ablation studies  
   - Robustness testing
   - ImageNet debugging
   - Batch size optimization
   - Deep model training

### Storage Locations

All outputs are stored in your Google Drive:

- **📊 Results**: `{PROJECT_PATH}/checkpoints/*.json`
- **🤖 Models**: `{PROJECT_PATH}/checkpoints/*.pth`
- **📈 Logs**: `{PROJECT_PATH}/logs/`
- **📦 Data**: `{PROJECT_PATH}/data/ImageNet-1K/`
- **📸 Plots**: Displayed in notebook + saved to Drive

## 🔧 Advanced Configuration

### Custom Training Settings

Modify these variables in the training cells:

```python
# Training Configuration
DATASET = 'CIFAR100'        # 'CIFAR10', 'CIFAR100', 'ImageNet'
COMPLEXITY = 'deep'         # 'shallow', 'medium', 'deep'
EPOCHS = 30                 # Number of training epochs
LEARNING_RATE = 0.0001      # Learning rate (None for auto)
BATCH_SIZE = None           # Batch size (None for auto)
```

### GPU Optimization

The notebook automatically:
- Detects available GPU memory
- Optimizes batch sizes for your hardware
- Uses mixed precision training when beneficial
- Implements gradient clipping and accumulation

## 📊 Monitoring and Results

### Real-Time Monitoring

- **Training Progress**: Live plots in notebook cells
- **GPU Usage**: Monitor in Colab's resource panel
- **File Growth**: Watch Drive storage usage

### Results Analysis

All experiments generate JSON result files:

```python
# Load any result file
import json
with open(f'{PROJECT_PATH}/checkpoints/experiment_results.json', 'r') as f:
    results = json.load(f)
```

### Key Result Files

- `mnist_validation_results.json` - MNIST baseline performance
- `ablation_study_results.json` - Architecture analysis
- `robustness_test_results.json` - Stability testing
- `imagenet_debug_results.json` - ImageNet pipeline analysis
- `batch_size_optimization_results.json` - GPU optimization
- `deep_mwnn_*.json` - Deep model training results

## 🛠️ Troubleshooting

### Common Issues

**1. Project Not Found**
- Verify `PROJECT_PATH` matches your Drive folder
- Check folder permissions and structure
- Ensure all required files were uploaded

**2. GPU Memory Errors**
- Run batch size optimization first
- Reduce model complexity (`COMPLEXITY = 'medium'`)
- Lower batch size manually

**3. Dataset Download Fails**
- Upload Kaggle credentials (`kaggle.json`)
- Or manually upload ImageNet files to `{PROJECT_PATH}/data/ImageNet-1K/`

**4. Permission Errors**
- Re-mount Google Drive
- Check folder sharing settings
- Verify Colab has Drive access

### Performance Tips

**GPU Utilization**:
- Use Colab Pro for better GPUs (V100, A100)
- Enable high-RAM runtime if available
- Monitor GPU memory usage during training

**Data Loading**:
- ImageNet data loads from Drive (may be slower initially)
- Consider smaller datasets (CIFAR-100) for testing
- Use `num_workers=2` for data loading optimization

**Training Speed**:
- Start with shallow models, progress to deep
- Use automated batch size detection
- Enable mixed precision training

## 📱 Access from Multiple Devices

### Collaborative Workflow

1. **Share Drive Folder**: Right-click → Share → Add collaborators
2. **Cloud Access**: Results viewable from any device with Google account
3. **Download Results**: Export specific files or entire folders
4. **Continue Training**: Load checkpoints to resume training

### Mobile Monitoring

- View training plots on mobile via Drive
- Check result files using Google Drive app
- Monitor training progress remotely

## 🔄 Session Management

### Handling Disconnections

Google Colab sessions can disconnect. The Drive-based approach ensures:

- ✅ **No Data Loss**: All results saved to Drive automatically
- ✅ **Resume Training**: Load from last checkpoint
- ✅ **Persistent Storage**: Data survives session restarts

### Reconnection Steps

1. Re-run mount and navigation cells
2. Skip data download (already in Drive)
3. Load previous results to continue analysis
4. Resume training from checkpoints if needed

## 📈 Scaling to Production

### From Prototype to Production

1. **Validate on CIFAR**: Start with CIFAR-10/100 for testing
2. **Optimize Hyperparameters**: Use automated optimization
3. **Scale to ImageNet**: Apply optimized settings to full dataset
4. **Deploy Model**: Export trained models for inference

### Resource Planning

- **Storage**: ~50GB for full ImageNet + results
- **Training Time**: 2-6 hours depending on model complexity
- **GPU Requirements**: T4 minimum, V100+ recommended

## 🎯 Next Steps

After successful training:

1. **Model Analysis**: Compare different architectures
2. **Hyperparameter Tuning**: Optimize learning rates, batch sizes
3. **Transfer Learning**: Apply to other datasets
4. **Production Deployment**: Export models for inference
5. **Research**: Publish results and methodologies

---

## 📞 Support

For issues or questions:

1. Check notebook cell outputs for specific error messages
2. Verify all file paths use `PROJECT_PATH` variable
3. Ensure sufficient Drive storage space
4. Monitor Colab resource usage

**Remember**: Everything is saved to your Google Drive, so you can always restart and continue where you left off!

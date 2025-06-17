# Multi-Weight Neural Networks (MWNN) - Google Colab Setup

This notebook sets up and runs the MWNN project on Google Colab with GPU acceleration.

## 🚀 Quick Setup

```python
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install matplotlib seaborn pandas numpy scipy
!pip install tensorboard

# Clone the repository (if needed)
# !git clone <your-repo-url>
# %cd multi-weight-neural-networks

# Or upload your project files to Colab
print("✅ Setup complete! Upload your project files to the Colab environment.")
```

## 📁 File Upload Instructions

1. **Compress your project locally**:
   ```bash
   # On your local machine
   cd /Users/gclinger/Documents/projects/mwnn
   tar -czf mwnn-project.tar.gz multi-weight-neural-networks/
   ```

2. **Upload to Colab**:
   - Use the file upload button in Colab
   - Or mount Google Drive and upload there

3. **Extract in Colab**:
   ```python
   # Extract the uploaded tar file
   !tar -xzf mwnn-project.tar.gz
   %cd multi-weight-neural-networks
   ```

## 🖥️ GPU Setup Verification

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 📊 Quick Test Runs

### 1. MNIST Validation (Quick)
```python
!python test_mnist_csv.py
```

### 2. Ablation Study (Medium)
```python
!python test_ablation_study.py
```

### 3. Robustness Testing (Medium)
```python
!python test_robustness.py
```

### 4. ImageNet Debugging (Long)
```python
!python debug_imagenet_pipeline.py
```

## 📈 Monitor Training with TensorBoard

```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard (run after training starts)
%tensorboard --logdir logs/
```

## 💾 Save Results to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r checkpoints/ /content/drive/MyDrive/mwnn_results/
!cp -r logs/ /content/drive/MyDrive/mwnn_results/
```

## 🎯 Recommended Colab Workflow

1. **Start with Quick Tests**: MNIST validation
2. **Run Ablation Study**: Identify best configurations  
3. **Progressive ImageNet Testing**: Start simple, increase complexity
4. **Save All Results**: To Google Drive for persistence

## ⚡ Colab-Specific Optimizations

- **Runtime**: Use GPU runtime (T4, V100, or A100 if available)
- **Memory**: Monitor RAM usage, restart if needed
- **Persistence**: Save frequently to Google Drive
- **Batch Sizes**: Can use larger batch sizes with better GPUs

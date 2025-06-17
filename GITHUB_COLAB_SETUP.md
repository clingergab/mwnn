# GitHub to Colab Setup Guide

## 🚀 Quick Start: From GitHub to Colab

### Option 1: Direct Colab Link (Recommended)
1. **Open Notebook**: Click the "Open in Colab" badge in the GitHub README
2. **Clone Project**: Run the GitHub clone cell in the notebook
3. **Upload Data**: Place ImageNet-1K in your Google Drive
4. **Start Training**: Run the training cells

### Option 2: Manual Clone in Colab
```python
# In a new Colab notebook
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME

# Open the main notebook
from google.colab import files
files.view('MWNN_Colab_Training.ipynb')
```

## 📁 Data Setup with GitHub Clone

When using GitHub clone, you have two options for data:

### Option A: Data in Drive (Recommended)
```
Google Drive Structure:
/MyDrive/
├── mwnn/data/ImageNet-1K/          # Data only
│   ├── train/
│   ├── val/ 
│   └── ILSVRC2013_devkit/
└── (GitHub clone in Colab session)
```

Update data paths in training script:
```python
data_dir='/content/drive/MyDrive/mwnn/data/ImageNet-1K'
```

### Option B: Data in Colab Session
```python
# Upload data directly to Colab (slower)
from google.colab import files
# Upload ImageNet files...
```

## 🔄 Workflow

1. **Fork the Repository** on GitHub
2. **Clone in Colab** using your fork
3. **Make Changes** in Colab
4. **Save to GitHub**:
   ```python
   # In Colab
   !git add .
   !git commit -m "Updated training results"
   !git push origin main
   ```

## 🎯 Benefits of GitHub Workflow

- ✅ **Version Control**: Track all changes
- ✅ **Collaboration**: Easy sharing and contributions
- ✅ **Backup**: Code safely stored on GitHub
- ✅ **Updates**: Easy to pull latest changes
- ✅ **Reproducibility**: Clear version history

## 📝 Notes

- **Data Not in Repo**: Data directory excluded via `.gitignore`
- **Checkpoints**: Training outputs saved to Drive for persistence
- **Logs**: Excluded from repo, save important results manually
- **Large Files**: Use Git LFS if needed for large model files

---

**Ready to start? Click the Colab badge in the main README!**

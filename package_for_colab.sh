#!/bin/bash
# Package MWNN project for Google Colab deployment with Drive integration
# This script creates a complete package that can be uploaded to Google Drive

echo "📦 Packaging MWNN for Google Colab with Drive Integration..."

# Create package directory
PKG_DIR="mwnn_colab_package"
rm -rf $PKG_DIR
mkdir $PKG_DIR

echo "📁 Copying source files..."

# Copy core source code
cp -r src/ $PKG_DIR/
cp -r configs/ $PKG_DIR/

# Copy training and setup scripts
cp train_deep_colab.py $PKG_DIR/
cp setup_imagenet_colab.py $PKG_DIR/
cp optimize_batch_sizes.py $PKG_DIR/

# Copy test and analysis scripts  
cp test_*.py $PKG_DIR/
cp debug_imagenet_pipeline.py $PKG_DIR/
cp run_*.py $PKG_DIR/

# Copy configuration and setup files
cp requirements_colab.txt $PKG_DIR/
cp setup.py $PKG_DIR/

# Copy documentation
cp README.md $PKG_DIR/
cp COLAB_*.md $PKG_DIR/
cp DRIVE_DEPLOYMENT_GUIDE.md $PKG_DIR/  # New Drive guide
cp IMAGENET_TRAINING_GUIDE.md $PKG_DIR/
cp *_SUMMARY.md $PKG_DIR/ 2>/dev/null || true

# Copy the main Colab notebook (updated for Drive)
cp MWNN_Colab_Training.ipynb $PKG_DIR/

# Copy utility scripts
cp setup_colab.py $PKG_DIR/
cp package_for_colab.sh $PKG_DIR/

# Copy example data if it exists (small files only)
if [ -d "data/MNIST" ]; then
    mkdir -p $PKG_DIR/data/MNIST
    # Only copy if files are reasonably sized
    find data/MNIST -name "*.csv" -size -50M -exec cp {} $PKG_DIR/data/MNIST/ \; 2>/dev/null || true
fi

# Create placeholder directories
mkdir -p $PKG_DIR/checkpoints
mkdir -p $PKG_DIR/logs  
mkdir -p $PKG_DIR/data/ImageNet-1K
mkdir -p $PKG_DIR/results

# Create Drive setup instructions
cat > $PKG_DIR/DRIVE_SETUP_INSTRUCTIONS.txt << 'EOF'
🚀 MWNN Google Drive Setup Instructions

1. UPLOAD TO DRIVE:
   - Upload this entire folder to your Google Drive
   - Recommended path: /MyDrive/mwnn-project/

2. OPEN IN COLAB:
   - Open Google Colab (colab.research.google.com)
   - Upload MWNN_Colab_Training.ipynb from your Drive folder
   - Or: File → Open notebook → Google Drive → Select the notebook

3. UPDATE PROJECT PATH:
   - In the notebook, find this line:
     PROJECT_PATH = "/content/drive/MyDrive/mwnn-project"
   - Change it to match your actual Drive folder path

4. RUN THE NOTEBOOK:
   - Enable GPU runtime (Runtime → Change runtime type → GPU)
   - Run all cells sequentially
   - All data and results will be saved to your Drive automatically

5. FEATURES:
   ✅ Persistent storage in Google Drive
   ✅ No data loss on disconnection
   ✅ Access from any device
   ✅ Easy collaboration and sharing
   ✅ Automatic batch size optimization
   ✅ Deep model training with ImageNet

📖 See DRIVE_DEPLOYMENT_GUIDE.md for detailed instructions!
EOF

# Create requirements summary
cat > $PKG_DIR/REQUIREMENTS_SUMMARY.txt << 'EOF'
📋 MWNN Colab Requirements

GOOGLE COLAB:
- GPU runtime (T4, V100, or A100 recommended)
- High-RAM runtime (optional, for large models)
- Google Drive with ~50GB free space

PYTHON PACKAGES (installed automatically):
- torch >= 1.9.0
- torchvision >= 0.10.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- pandas >= 1.1.0
- seaborn >= 0.11.0
- tensorboard >= 2.7.0

DATASETS:
- ImageNet-1K (downloaded via Kaggle in notebook)
- MNIST (CSV format, included in package)
- CIFAR-10/100 (downloaded automatically by PyTorch)

OPTIONAL:
- Kaggle account + API key for ImageNet download
- Colab Pro for longer runtimes and better GPUs
EOF

echo "📄 Creating package archive..."

# Create tar archive
tar -czf mwnn_colab_package.tar.gz $PKG_DIR/

# Cleanup
rm -rf $PKG_DIR

echo "✅ Package created: mwnn_colab_package.tar.gz"
echo ""
echo "📤 To deploy:"
echo "1. Upload mwnn_colab_package.tar.gz to Google Drive"
echo "2. Extract in Drive to create your project folder"  
echo "3. Open MWNN_Colab_Training.ipynb in Google Colab"
echo "4. Update PROJECT_PATH to match your Drive folder"
echo "5. Run all cells!"
echo ""
echo "📖 See DRIVE_DEPLOYMENT_GUIDE.md for detailed instructions"

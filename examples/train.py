#!/usr/bin/env python3
"""
Simple MWNN Training Script
Clean, easy-to-use training for ImageNet with MWNN

Usage:
    python train.py --data_path /path/to/imagenet --epochs 30 --batch_size 64
"""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from mwnn import MWNN


def main():
    """Main training function with clean API."""
    parser = argparse.ArgumentParser(description='Train MWNN on ImageNet')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet data directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--depth', type=str, default='deep',
                        choices=['shallow', 'medium', 'deep'],
                        help='Model depth')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Use subset of data for testing')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    print("🚀 MWNN Training - Clean & Simple")
    print(f"Data: {args.data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model Depth: {args.depth}")
    print()
    
    # Step 1: Load data
    print("📊 Loading ImageNet data...")
    train_loader, val_loader = MWNN.load_imagenet_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        subset_size=args.subset_size
    )
    print(f"✅ Data loaded - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Step 2: Create model
    print("🧠 Creating MWNN model...")
    model = MWNN(
        num_classes=1000,
        depth=args.depth,
        device='auto'
    )
    model.summary()
    print()
    
    # Step 3: Train
    print("🏋️ Starting training...")
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Step 4: Evaluate
    print("\n📈 Final evaluation...")
    results = model.evaluate(val_loader)
    print(f"Final Validation Accuracy: {results['accuracy']:.2f}%")
    print(f"Final Validation Loss: {results['loss']:.4f}")
    
    # Step 5: Save model
    save_path = Path(args.checkpoint_dir) / 'final_mwnn_model.pth'
    model.save(save_path)
    print(f"🎉 Training completed! Model saved to {save_path}")
    
    return history, results


if __name__ == "__main__":
    main()

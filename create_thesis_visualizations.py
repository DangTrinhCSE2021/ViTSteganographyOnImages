#!/usr/bin/env python3
"""
Comprehensive visualization script for ViT Steganography training results.
This script generates various plots and charts for thesis documentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse
from pathlib import Path
import pickle
import glob
from PIL import Image
import torch


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_training_data(run_folder):
    """Load training and validation CSV files."""
    train_csv = os.path.join(run_folder, 'train.csv')
    val_csv = os.path.join(run_folder, 'validation.csv')
    
    train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else None
    val_df = pd.read_csv(val_csv) if os.path.exists(val_csv) else None
    
    return train_df, val_df

def plot_loss_curves(train_df, val_df, save_path):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    
    # Key metrics to plot
    metrics = ['loss', 'encoder_mse', 'dec_mse', 'bitwise-error', 'img_recovery', 'perceptual_enc']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if metric in train_df.columns:
            ax.plot(train_df['epoch'], train_df[metric], label='Training', linewidth=2, alpha=0.8)
        
        if val_df is not None and metric in val_df.columns:
            ax.plot(val_df['epoch'], val_df[metric], label='Validation', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'loss_curves.pdf'), bbox_inches='tight')
    plt.close()

def plot_bit_accuracy(train_df, val_df, save_path):
    """Plot bit accuracy over training."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if 'bitwise-error' in train_df.columns:
        train_accuracy = 1 - train_df['bitwise-error']
        ax.plot(train_df['epoch'], train_accuracy * 100, label='Training Accuracy', linewidth=2)
    
    if val_df is not None and 'bitwise-error' in val_df.columns:
        val_accuracy = 1 - val_df['bitwise-error']
        ax.plot(val_df['epoch'], val_accuracy * 100, label='Validation Accuracy', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title('Watermark Bit Accuracy Over Training', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'bit_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'bit_accuracy.pdf'), bbox_inches='tight')
    plt.close()

def plot_image_quality_metrics(train_df, val_df, save_path):
    """Plot image quality metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE plot
    ax1 = axes[0]
    if 'encoder_mse' in train_df.columns:
        ax1.plot(train_df['epoch'], train_df['encoder_mse'], label='Training MSE', linewidth=2)
    if val_df is not None and 'encoder_mse' in val_df.columns:
        ax1.plot(val_df['epoch'], val_df['encoder_mse'], label='Validation MSE', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Image Quality (MSE)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Perceptual loss plot
    ax2 = axes[1]
    if 'perceptual_enc' in train_df.columns:
        ax2.plot(train_df['epoch'], train_df['perceptual_enc'], label='Training Perceptual', linewidth=2)
    if val_df is not None and 'perceptual_enc' in val_df.columns:
        ax2.plot(val_df['epoch'], val_df['perceptual_enc'], label='Validation Perceptual', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perceptual Loss')
    ax2.set_title('Perceptual Quality Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'image_quality_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'image_quality_metrics.pdf'), bbox_inches='tight')
    plt.close()

def create_training_summary_table(train_df, val_df, save_path):
    """Create a summary table of final training results."""
    summary_data = []
    
    # Get final epoch data
    if train_df is not None and len(train_df) > 0:
        final_train = train_df.iloc[-1]
        summary_data.append({
            'Dataset': 'Training',
            'Final Epoch': int(final_train['epoch']),
            'Total Loss': f"{final_train.get('loss', 0):.4f}",
            'Bit Accuracy (%)': f"{(1 - final_train.get('bitwise-error', 1)) * 100:.2f}",
            'Image MSE': f"{final_train.get('encoder_mse', 0):.6f}",
            'Perceptual Loss': f"{final_train.get('perceptual_enc', 0):.4f}"
        })
    
    if val_df is not None and len(val_df) > 0:
        final_val = val_df.iloc[-1]
        summary_data.append({
            'Dataset': 'Validation',
            'Final Epoch': int(final_val['epoch']),
            'Total Loss': f"{final_val.get('loss', 0):.4f}",
            'Bit Accuracy (%)': f"{(1 - final_val.get('bitwise-error', 1)) * 100:.2f}",
            'Image MSE': f"{final_val.get('encoder_mse', 0):.6f}",
            'Perceptual Loss': f"{final_val.get('perceptual_enc', 0):.4f}"
        })
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    summary_df = pd.DataFrame(summary_data)
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Training Results Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_path, 'training_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'training_summary_table.pdf'), bbox_inches='tight')
    plt.close()
    
    return summary_df

def plot_convergence_analysis(train_df, save_path):
    """Analyze training convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Loss smoothing
    if len(train_df) > 5:
        window = min(10, len(train_df) // 5)
        smoothed_loss = train_df['loss'].rolling(window=window).mean()
        
        axes[0, 0].plot(train_df['epoch'], train_df['loss'], alpha=0.3, label='Raw Loss')
        axes[0, 0].plot(train_df['epoch'], smoothed_loss, linewidth=2, label='Smoothed Loss')
        axes[0, 0].set_title('Loss Convergence')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient analysis (loss derivative)
    if len(train_df) > 2:
        loss_gradient = np.gradient(train_df['loss'])
        axes[0, 1].plot(train_df['epoch'], loss_gradient, linewidth=2)
        axes[0, 1].set_title('Loss Gradient (Rate of Change)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Gradient')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Bit error convergence
    if 'bitwise-error' in train_df.columns:
        axes[1, 0].plot(train_df['epoch'], train_df['bitwise-error'], linewidth=2)
        axes[1, 0].set_title('Bit Error Convergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Bit Error Rate')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Training stability (loss variance)
    if len(train_df) > 10:
        window = 10
        loss_var = train_df['loss'].rolling(window=window).var()
        axes[1, 1].plot(train_df['epoch'][window-1:], loss_var[window-1:], linewidth=2)
        axes[1, 1].set_title('Training Stability (Loss Variance)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Variance')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'convergence_analysis.pdf'), bbox_inches='tight')
    plt.close()

def create_architecture_diagram(save_path):
    """Create a simplified architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define components
    components = [
        {'name': 'Input Image\n(128x128x3)', 'pos': (1, 4), 'color': '#FF9999'},
        {'name': 'Message\n(30 bits)', 'pos': (1, 2), 'color': '#99FF99'},
        {'name': 'ViT Encoder\n(Vision Transformer)', 'pos': (3, 4), 'color': '#9999FF'},
        {'name': 'Message Embedding', 'pos': (3, 2), 'color': '#FFFF99'},
        {'name': 'Watermarked Image\n(128x128x3)', 'pos': (5, 3), 'color': '#FF99FF'},
        {'name': 'Noise Layers\n(Crop, JPEG, etc.)', 'pos': (7, 3), 'color': '#99FFFF'},
        {'name': 'Decoder\n(CNN)', 'pos': (9, 4), 'color': '#FFB366'},
        {'name': 'Discriminator\n(GAN)', 'pos': (9, 2), 'color': '#B366FF'},
        {'name': 'Recovered Message\n(30 bits)', 'pos': (11, 4), 'color': '#99FF99'},
        {'name': 'Authenticity Score', 'pos': (11, 2), 'color': '#66B2FF'}
    ]
    
    # Draw components
    for comp in components:
        rect = plt.Rectangle((comp['pos'][0]-0.4, comp['pos'][1]-0.3), 0.8, 0.6, 
                           facecolor=comp['color'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(comp['pos'][0], comp['pos'][1], comp['name'], 
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((1.4, 4), (2.6, 4)),    # Input -> ViT
        ((1.4, 2), (2.6, 2)),    # Message -> Embedding
        ((3.4, 4), (4.6, 3.2)),  # ViT -> Watermarked
        ((3.4, 2), (4.6, 2.8)),  # Embedding -> Watermarked
        ((5.4, 3), (6.6, 3)),    # Watermarked -> Noise
        ((7.4, 3), (8.6, 3.5)),  # Noise -> Decoder
        ((7.4, 3), (8.6, 2.5)),  # Noise -> Discriminator
        ((9.4, 4), (10.6, 4)),   # Decoder -> Recovered
        ((9.4, 2), (10.6, 2))    # Discriminator -> Score
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(1, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('ViT Steganography Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'architecture_diagram.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'architecture_diagram.pdf'), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for thesis')
    parser.add_argument('--run-folder', '-r', required=True, type=str,
                       help='Path to the training run folder')
    parser.add_argument('--output-dir', '-o', default='thesis_visualizations', type=str,
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading training data from {args.run_folder}...")
    train_df, val_df = load_training_data(args.run_folder)
    
    if train_df is None:
        print("Error: Could not load training data")
        return
    
    print("Generating visualizations...")
    
    # Generate all plots
    plot_loss_curves(train_df, val_df, args.output_dir)
    print("✓ Loss curves generated")
    
    plot_bit_accuracy(train_df, val_df, args.output_dir)
    print("✓ Bit accuracy plot generated")
    
    plot_image_quality_metrics(train_df, val_df, args.output_dir)
    print("✓ Image quality metrics generated")
    
    create_training_summary_table(train_df, val_df, args.output_dir)
    print("✓ Training summary table generated")
    
    plot_convergence_analysis(train_df, args.output_dir)
    print("✓ Convergence analysis generated")
    
    create_architecture_diagram(args.output_dir)
    print("✓ Architecture diagram generated")
    
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("Files generated:")
    for file in os.listdir(args.output_dir):
        print(f"  - {file}")

if __name__ == '__main__':
    main()

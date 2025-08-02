# ViT Steganography on Images

## Overview
This project implements an advanced **Vision Transformer (ViT) based steganography system** for embedding and recovering secret messages in images. The system combines **latent space embedding** with **message-to-image conversion** techniques to achieve robust steganography that can withstand various attacks and distortions.

### Key Achievements
- **High Recovery Performance**: Targets **32dB+ recovery PSNR** while maintaining imperceptible watermarks
- **Robust Message Extraction**: Achieves **95%+ bit accuracy** under various noise conditions
- **Advanced ViT Architecture**: Utilizes Vision Transformers for semantic-level message embedding
- **Latent Space Modification**: True latent space manipulation for enhanced robustness

## Architecture Overview

### Core Components
1. **MessageToImageLatentEncoder**: Combines message-to-image conversion with ViT-based latent space embedding
2. **AdvancedRecoveryDecoder**: Multi-scale decoder with skip connections and residual blocks for high-quality recovery
3. **LatentMessageEmbedder**: Embeds messages directly into ViT feature space
4. **Enhanced Noise Layers**: Simulate real-world attacks (JPEG, crop, resize, etc.)

### Key Features
- **Vision Transformer Backbone**: Leverages ViT for semantic message embedding
- **Latent Space Steganography**: Modifies ViT features for improved robustness
- **Multi-Phase Training**: Progressive training strategy with adaptive optimization
- **Attack Robustness**: Handles JPEG compression, cropping, resizing, and geometric transformations
- **High Recovery Quality**: Advanced decoder architecture targeting 32dB recovery PSNR

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/DangTrinhCSE2021/ViTSteganographyOnImages.git
   cd ViTSteganographyOnImages
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python setup_check.py
   ```

### Dependencies
```
torch>=1.8.0
torchvision>=0.9.0
vit-pytorch>=0.19.0
pillow>=8.0.0
numpy>=1.19.0
scikit-image>=0.17.0
opencv-python>=4.5.0
plotly>=4.14.0
tensorboardX>=2.1
```

## Data Preparation

### Dataset Structure
Organize your training data as follows:
```
data/
├── train/
│   └── images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    └── images/
        ├── val_image1.jpg
        ├── val_image2.jpg
        └── ...
```

### Supported Formats
- JPEG, PNG, BMP, TIFF
- Recommended resolution: 128x128 or higher
- Color images (RGB)

## Training

### High Recovery Latent Training (Recommended)
For the best performance with 32dB+ recovery PSNR:

```bash
python train_high_recovery_latent.py \
    --data-dir data \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0001 \
    --message-length 30 \
    --encoder-channels 32 \
    --decoder-channels 32 \
    --clean-train-epochs 30 \
    --device cuda
```

### Training Parameters
- `--data-dir`: Path to dataset directory
- `--batch-size`: Training batch size (16 recommended for GPU)
- `--epochs`: Total training epochs (100-200 recommended)
- `--lr`: Learning rate (0.0001 default)
- `--message-length`: Secret message length in bits (30 default)
- `--encoder-channels`: Encoder channel depth (32 default)
- `--decoder-channels`: Decoder channel depth (32 default)
- `--clean-train-epochs`: Epochs of clean training before attacks (30 default)
- `--device`: Device to use ('cuda' or 'cpu')

### Alternative Training Methods

#### Basic ViT Training
```bash
python main.py new \
    --data-dir data \
    --batch-size 16 \
    --epochs 100 \
    --name basic_vit_experiment \
    --encoder-mode vit
```



### Training Phases
1. **Clean Training (30% of epochs)**: Focus on message embedding and recovery without attacks
2. **Balanced Training (40% of epochs)**: Introduce light attacks while maintaining recovery quality
3. **Robust Training (30% of epochs)**: Full attack simulation for robustness

## Testing and Validation

### Model Validation
Validate trained models on the validation dataset:

```bash
python validate-trained-models.py \
    --data-dir data \
    --runs_root latent_runs/
```

### Single Image Testing
Test steganography on individual images:

```bash
python visual_test_model.py \
    --model-path "latent_runs/high_recovery_latent_YYYYMMDD_HHMMSS/checkpoints/best_model.pth" \
    --test-image "path/to/test_image.jpg" \
    --message "Your secret message here"
```

### Visual Comparison Testing
Generate visual comparisons of original, watermarked, and recovered images:

```bash
python visual_comparison_recovery.py \
    --model-path "latent_runs/high_recovery_latent_YYYYMMDD_HHMMSS/checkpoints/best_model.pth" \
    --test-dir "path/to/test_images/"
```

### Advanced Testing
For comprehensive model evaluation with attacks:

```bash
python test_corrected_recovery_model.py \
    --model-path "latent_runs/high_recovery_latent_YYYYMMDD_HHMMSS/checkpoints/best_model.pth" \
    --test-data-dir data/val \
    --enable-attacks
```

## Model Architecture Details

### Encoder Architecture
- **ViT Backbone**: Vision Transformer with 6 layers, 12 heads
- **Message Converter**: Converts binary messages to robust spatial patterns
- **Latent Embedder**: Modifies ViT feature space for message embedding
- **Spatial Processor**: Upsampling and spatial refinement layers

### Decoder Architecture  
- **Multi-Scale Processing**: Multiple downsampling/upsampling stages
- **Skip Connections**: U-Net style connections for information preservation
- **Residual Blocks**: 4 residual blocks for recovery refinement
- **Message Extractor**: Dedicated branch for message recovery

### Loss Functions
- **Message Loss**: Binary cross-entropy for message accuracy
- **Image Quality Loss**: MSE + L1 for watermark imperceptibility
- **Recovery Loss**: MSE + L1 + perceptual loss for recovery quality
- **Adaptive Weighting**: Phase-dependent loss balancing

## Noise Layers and Attacks

### Supported Attacks
- **JPEG Compression**: Various quality levels (10-95)
- **Geometric Transforms**: Crop, resize, rotation
- **Pixel Attacks**: Dropout, Gaussian noise, salt-and-pepper
- **Combined Attacks**: Multiple simultaneous distortions

### Attack Configuration
```python
# Example noise configuration
noise_layers = [
    "jpeg_compression(quality_lower=70, quality_upper=95)",
    "cropout((0.55, 0.6), (0.55, 0.6))",
    "resize((0.7, 0.8), (0.7, 0.8))",
    "dropout((0.55, 0.6), (0.55, 0.6))"
]
```

## Performance Metrics

### Key Performance Indicators
- **Recovery PSNR**: Target ≥32dB (excellent: ≥35dB)
- **Watermark PSNR**: Target 35-50dB (imperceptible)
- **Bit Accuracy**: Target ≥95% (excellent: ≥98%)
- **SSIM**: Target ≥0.95 for both watermark and recovery

### Results Tracking
Training metrics are automatically saved to:
- `latent_runs/[experiment_name]/high_recovery_training_metrics.csv`
- Tensorboard logs (if enabled)
- Model checkpoints and visualizations

## File Structure
```
ViTSteganographyOnImages/
├── model/                          # Core model architectures
│   ├── latent_space.py            # Latent space embedding
│   ├── message_to_image.py        # Message-to-image conversion
│   ├── enhanced_vit_encoder.py    # Enhanced ViT encoder
│   └── enhanced_decoder.py        # Advanced recovery decoder
├── noise_layers/                   # Attack simulation
├── train_high_recovery_latent.py   # Main training script
├── visual_test_model.py           # Single image testing
├── validate-trained-models.py     # Model validation
├── setup_check.py                 # Environment verification
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Advanced Usage

### Custom Model Configuration
```python
from options import HiDDenConfiguration

config = HiDDenConfiguration(
    H=128, W=128,
    message_length=30,
    encoder_channels=32,
    decoder_channels=32,
    encoder_blocks=4,
    decoder_blocks=7
)
```

### Integration Example
```python
from train_high_recovery_latent import HighRecoveryLatentSteganographySystem

# Load trained model
model = HighRecoveryLatentSteganographySystem(config, device)
model.load_state_dict(torch.load('path/to/model.pth'))

# Embed message
watermarked = model.encoder(cover_image, message)

# Extract message
decoded_message, recovered_image = model.decoder(watermarked)
```

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size to 8 or 4
2. **Poor Recovery Quality**: Ensure clean training phase is sufficient (30+ epochs)
3. **Low Bit Accuracy**: Check message length and model capacity
4. **Training Instability**: Reduce learning rate to 0.00005

### Performance Optimization
- Use mixed precision training: `--mixed-precision`
- Enable gradient clipping: `--grad-clip 1.0`
- Use data parallelism: `--multi-gpu`



## Acknowledgments
- Original HiDDeN paper implementation
- Vision Transformer (ViT) architecture
- RosteALS paper for latent space concepts

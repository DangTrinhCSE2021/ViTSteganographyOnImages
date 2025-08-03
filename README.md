# VitTransformerWatermarkingScheme

## Overview
This project implements a robust, deep learning-based image watermarking scheme using Vision Transformers (ViT) and other transformer-based models. The main objective is to embed a secret binary message into an image such that the message can later be extracted, even after the image has undergone various distortions (e.g., cropping, compression, resizing).

## Key Features
- **Vision Transformer (ViT) Encoder:** Leverages ViT to embed messages in the semantic space of images for improved robustness.
- **Robust to Noise:** Handles various noise types (crop, cropout, dropout, resize, JPEG compression, quantization, etc.).
- **Flexible Architecture:** Easily configurable encoder/decoder and noise layers.
- **Experiment Management:** Supports new runs, checkpointing, and resuming training.
- **Validation and Testing:** Scripts for validating on datasets and testing on single images.

## How It Works
1. **Encoder:** Embeds a binary message into an image using a ViT-based or DINO-based architecture.
2. **Noiser:** Applies random noise layers to simulate real-world distortions.
3. **Decoder:** Attempts to extract the original message from the (possibly noised) image.
4. **Training:** The model is trained to minimize both the visual difference between the original and watermarked images, and the error in the extracted message.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd VitTransformerWatermarkingScheme
   ```
2. Check your environment and install dependencies:
   ```bash
   python setup_check.py
   ```
   Or install manually:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify installation:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

## Data Preparation
Organize your dataset as follows:
```
data/
  train/
    train_class/
      image1.JPEG
      ...
  val/
    val_class/
      image1.JPEG
      ...
```

## Training a Model
Run the following command to start training:
```bash
python train_high_recovery_latent.py --data-dir data --batch-size 32 --epochs 5 --name my_experiment 
```
- `--data-dir`: Path to your data folder (should contain `train` and `val` subfolders)
- `--batch-size`: Training batch size
- `--epochs`: Number of epochs
- `--name`: Name for this experiment

- Optional: `--noise` to specify noise layers, `--tensorboard` to enable logging


### Example Commands
```bash
# Train with ViT encoder
python train_high_recovery_latent.py --epochs 15 --batch_size 32 --lr 3e-4 --clean_train_epochs 5

# Train with DINO output features


## Validating a Model
To validate your trained model on the validation set:
```bash
python validate-trained-models.py --data-dir data --runs_root runs/
```

## Testing on a Single Image
To embed and extract a message from a single image:
```bash
python test_corrected_recovery_model.py --model_path "..." --test_image "" --num_tests 5 --save_results

## Embedding and Extracting Messages
- **Embedding:** The encoder embeds a random binary message into a cropped region of the input image.
- **Extraction:** The decoder attempts to recover the message from the watermarked (and possibly noised) image.
- The process is robust to common image distortions.

## Using Vision Transformer (ViT)
- The encoder uses ViT to generate semantic features, making the watermark more robust and less visible.
- You can select different encoder modes (e.g., `vit`, `dino-output`, `dino-attention`) in the code/configuration.

## Requirements
See `requirements.txt` for all dependencies. Main requirements:
- Python 3.8+
- torch >= 1.8.0
- torchvision >= 0.9.0
- transformers >= 4.5.0
- pillow, numpy, scikit-image, plotly, tensorboardX, opencv-python

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
If you use this code for research, please cite the original HiDDeN paper and any relevant ViT/DINO papers.

#!/usr/bin/env python3
"""
Simple validation script for a specific trained model.
"""

import torch
import torch.nn
import argparse
import os
import numpy as np
from collections import defaultdict
import utils
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob


def validate_model(model_folder, data_dir, batch_size=4):
    """Validate a specific model on the validation dataset."""
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model configuration
    options_file = os.path.join(model_folder, 'options-and-config.pickle')
    if not os.path.exists(options_file):
        print(f"Error: Options file not found at {options_file}")
        return
    
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    
    # Load the latest checkpoint
    checkpoint_dir = os.path.join(model_folder, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return
    
    checkpoint, chpt_file_name = utils.load_last_checkpoint(checkpoint_dir)
    print(f'Loaded checkpoint from file {chpt_file_name}')
    print(f'Checkpoint epoch: {checkpoint["epoch"]}')
    
    # Create model
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None)
    utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()
    
    # Setup data loading
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(val_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} validation images")
    
    if len(image_files) == 0:
        print("No validation images found!")
        return
    
    # Validation metrics
    losses_accu = defaultdict(AverageMeter)
    
    # Transform for images
    transform = transforms.Compose([
        transforms.Resize((hidden_config.H, hidden_config.W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Starting validation...")
    total_images = min(len(image_files), 100)  # Limit to 100 images for faster testing
    
    with torch.no_grad():
        for i, image_path in enumerate(image_files[:total_images]):
            if i % 20 == 0:
                print(f"Processing image {i+1}/{total_images}")
            
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Generate random message
                message = torch.Tensor(np.random.choice([0, 1], (1, hidden_config.message_length))).to(device)
                
                # Forward pass
                losses, _ = model.validate_on_batch([image_tensor, message])
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    losses_accu[loss_name].update(loss_value)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    # Print results
    print("\nValidation Results:")
    print("=" * 50)
    for loss_name, loss_avg in losses_accu.items():
        print(f"{loss_name:<20}: {loss_avg.avg:.4f}")
    print("=" * 50)
    
    # Key metrics
    if 'bitwise-error' in losses_accu:
        bit_accuracy = 1.0 - losses_accu['bitwise-error'].avg
        print(f"Bit Accuracy: {bit_accuracy:.2%}")
    
    if 'encoder_mse' in losses_accu:
        print(f"Image Quality (lower=better): {losses_accu['encoder_mse'].avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Validate a specific trained model')
    parser.add_argument('--model-folder', '-m', required=True, type=str,
                        help='Path to the specific model folder (containing options-and-config.pickle)')
    parser.add_argument('--data-dir', '-d', required=True, type=str,
                        help='The directory where the validation data is stored.')
    parser.add_argument('--batch-size', '-b', default=4, type=int, help='Validation batch size.')
    
    args = parser.parse_args()
    
    validate_model(args.model_folder, args.data_dir, args.batch_size)


if __name__ == '__main__':
    main()

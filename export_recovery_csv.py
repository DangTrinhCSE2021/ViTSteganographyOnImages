"""
Export Recovery Results to CSV
Simple CSV export of recovery PSNR and SSIM results.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import csv
from datetime import datetime

from options import HiDDenConfiguration
from integration_guide import create_enhanced_model
from train_latent_space import compute_psnr, compute_ssim, create_clean_noiser


def load_enhanced_model(model_path, device):
    """Load the enhanced recovery-optimized model."""
    config = HiDDenConfiguration(
        H=128, W=128, message_length=30,
        encoder_channels=32, encoder_blocks=4,
        decoder_channels=32, decoder_blocks=7,
        use_discriminator=True, use_vgg=True,
        discriminator_blocks=3, discriminator_channels=64,
        encoder_loss=1.0, decoder_loss=1.0, adversarial_loss=1e-3
    )
    
    noiser = create_clean_noiser(device)
    model = create_enhanced_model(config, device, noiser, None, use_latent_space=True)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def export_recovery_csv():
    """Export recovery results to CSV format."""
    
    print("Exporting Recovery Results to CSV")
    print("=" * 40)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "latent_runs/recovery_optimized_20250729_114830/best_model.pth"
    
    # Load model
    model = load_enhanced_model(model_path, device)
    
    # Test images
    test_images = [
        "data/val/val_class/ILSVRC2012_val_00007255.JPEG",
        "data/val/val_class/ILSVRC2012_val_00007257.JPEG", 
        "data/val/val_class/ILSVRC2012_val_00007259.JPEG",
        "data/val/val_class/ILSVRC2012_val_00007261.JPEG",
        "data/val/val_class/ILSVRC2012_val_00007263.JPEG",
        "data/val/val_class/ILSVRC2012_val_00007267.JPEG"
    ]
    
    available_images = [img for img in test_images if os.path.exists(img)]
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"recovery_results_{timestamp}.csv"
    
    results = []
    
    print(f"Testing {len(available_images)} images...")
    
    for img_idx, image_path in enumerate(available_images):
        print(f"Processing image {img_idx + 1}: {os.path.basename(image_path)}")
        
        # Load image
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Test with 5 different messages
        for msg_idx in range(5):
            np.random.seed(42 + msg_idx)
            message_bits = np.random.randint(0, 2, 30).astype(np.float32)
            test_message = torch.from_numpy(message_bits).unsqueeze(0).to(device)
            
            with torch.no_grad():
                model.noiser = create_clean_noiser(device)
                losses, outputs = model.train_on_batch([image_tensor, test_message])
                watermarked, noised, decoded, recovered = outputs
                
                # Calculate recovery metrics
                recovery_psnr = compute_psnr(image_tensor, recovered)
                recovery_ssim = compute_ssim(image_tensor, recovered)
                
                results.append({
                    'image_name': os.path.basename(image_path),
                    'message_id': msg_idx + 1,
                    'recovery_psnr': round(recovery_psnr, 2),
                    'recovery_ssim': round(recovery_ssim, 4)
                })
    
    # Write to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'message_id', 'recovery_psnr', 'recovery_ssim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results exported to: {csv_filename}")
    
    # Print summary statistics
    all_psnr = [r['recovery_psnr'] for r in results]
    all_ssim = [r['recovery_ssim'] for r in results]
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total measurements: {len(results)}")
    print(f"Average Recovery PSNR: {np.mean(all_psnr):.2f}dB")
    print(f"Best Recovery PSNR: {np.max(all_psnr):.2f}dB")
    print(f"Worst Recovery PSNR: {np.min(all_psnr):.2f}dB")
    print(f"Average Recovery SSIM: {np.mean(all_ssim):.4f}")
    print(f"Best Recovery SSIM: {np.max(all_ssim):.4f}")
    print(f"Worst Recovery SSIM: {np.min(all_ssim):.4f}")
    
    return csv_filename


if __name__ == "__main__":
    export_recovery_csv()

"""
Enhanced training script with recovery quality optimization and MESSAGE-TO-IMAGE conversion.
This adds recovery PSNR/SSIM tracking and optimization with robust message-image embedding.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
from datetime import datetime
import math
import csv
from skimage.metrics import structural_similarity as ssim

from options import HiDDenConfiguration
from model.enhanced_integration import create_enhanced_steganography_model
from noise_layers.enhanced_noiser import create_enhanced_noiser, create_clean_enhanced_noiser
import utils

# Import/define local functions
def compute_psnr(img1, img2):
    """Compute PSNR between two image tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0  # Assuming images are in [-1, 1] range
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr.item()

def compute_ssim(img1, img2):
    """Compute SSIM between two image tensors."""
    # Convert to numpy and compute SSIM
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(img1_np.shape) == 4:
        ssim_values = []
        for i in range(img1_np.shape[0]):
            # Transpose from CHW to HWC for skimage
            img1_hwc = img1_np[i].transpose(1, 2, 0)
            img2_hwc = img2_np[i].transpose(1, 2, 0)
            # Normalize to [0, 1] for SSIM
            img1_norm = (img1_hwc + 1) / 2
            img2_norm = (img2_hwc + 1) / 2
            ssim_val = ssim(img1_norm, img2_norm, multichannel=True, channel_axis=2, data_range=1.0)
            ssim_values.append(ssim_val)
        return sum(ssim_values) / len(ssim_values)
    else:
        img1_norm = (img1_np + 1) / 2
        img2_norm = (img2_np + 1) / 2
        return ssim(img1_norm, img2_norm, multichannel=True, channel_axis=2, data_range=1.0)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced ViT Steganography Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')  # Realistic learning rate
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--message_length', type=int, default=30, help='Message length')
    parser.add_argument('--encoder_channels', type=int, default=64, help='Encoder channels')
    parser.add_argument('--decoder_channels', type=int, default=64, help='Decoder channels')
    parser.add_argument('--encoder_loss', type=float, default=0.5, help='Encoder loss weight')  # REDUCED for realistic PSNR
    parser.add_argument('--decoder_loss', type=float, default=2.0, help='Decoder loss weight')  # Increased for message priority
    parser.add_argument('--adversarial_loss', type=float, default=0.05, help='Adversarial loss weight')  # Reduced
    parser.add_argument('--clean_train_epochs', type=int, default=15, help='Clean training epochs')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Training data directory')
    return parser.parse_args()

class RealDataset(torch.utils.data.Dataset):
    """Dataset for loading real images from directory."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Find actual image files
        import glob
        self.image_paths = []
        
        # Look for images in subdirectories (ImageNet structure)
        for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            pattern = os.path.join(data_dir, '**', ext)
            self.image_paths.extend(glob.glob(pattern, recursive=True))
        
        # If no images found in subdirs, try main directory
        if len(self.image_paths) == 0:
            for ext in ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                pattern = os.path.join(data_dir, ext)
                self.image_paths.extend(glob.glob(pattern))
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}")
            print(f"Using dummy dataset with 100 random images")
            self.use_dummy = True
            self.length = 100
        else:
            print(f"Loaded real dataset from {data_dir} with {len(self.image_paths)} images")
            self.use_dummy = False
            self.length = len(self.image_paths)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self.use_dummy:
            # Fallback to random tensor if no images found
            image = torch.randn(3, 128, 128)
        else:
            # Load actual image
            from PIL import Image
            image_path = self.image_paths[idx % len(self.image_paths)]
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Resize to 128x128
                image = image.resize((128, 128), Image.Resampling.LANCZOS)
                
                # Convert to tensor
                image = transforms.ToTensor()(image)
                
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Fallback to random tensor
                image = torch.randn(3, 128, 128)
        
        if self.transform:
            image = self.transform(image)
        return image, 0

def create_data_loader(data_dir, batch_size, is_training=True):
    """Create data loader."""
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)),
        transforms.Lambda(lambda x: (x * 2.0) - 1.0),  # Normalize to [-1, 1]
    ])
    
    dataset = RealDataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=0)
    return loader

def save_checkpoint(model, experiment_name, epoch, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{experiment_name}_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'experiment_name': experiment_name
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def enhanced_train_epoch(model, dataloader, optimizer, device, epoch, total_epochs, enable_attacks=False):
    """Enhanced training with recovery quality tracking and MESSAGE-IMAGE conversion."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_recovery_psnr = 0.0
    total_recovery_ssim = 0.0
    num_batches = 0
    
    # Update training mode based on epoch (PROGRESSIVE TRAINING)
    training_mode = model.update_training_mode(epoch, total_epochs)
    
    # Log training mode
    if enable_attacks:
        print(f"  Training mode: {training_mode} with enhanced attacks")
    else:
        print(f"  Training mode: {training_mode} (clean phase)")
    
    for batch_idx, (images, _) in enumerate(dataloader):
        # Prepare data
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Generate random messages
        messages = torch.randint(0, 2, (batch_size, 30)).float().to(device)
        
        # Forward pass with MESSAGE-TO-IMAGE conversion
        losses, outputs = model.train_on_batch([images, messages], training_mode=training_mode)
        watermarked_images, noised_images, decoded_messages, recovered_images = outputs
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate image quality metrics
        with torch.no_grad():
            # Watermark quality (original vs watermarked)
            psnr = compute_psnr(images, watermarked_images)
            ssim_val = compute_ssim(images, watermarked_images)
            
            # Recovery quality (original vs recovered) - NEW!
            recovery_psnr = 0.0
            recovery_ssim = 0.0
            if recovered_images is not None:
                recovery_psnr = compute_psnr(images, recovered_images)
                recovery_ssim = compute_ssim(images, recovered_images)
            
            total_psnr += psnr
            total_ssim += ssim_val
            total_recovery_psnr += recovery_psnr
            total_recovery_ssim += recovery_ssim
        
        # Track metrics
        total_loss += losses['total_loss'].item()
        num_batches += 1
        
        # Enhanced progress logging with MESSAGE-IMAGE info
        if batch_idx % 50 == 0:
            attack_status = "with enhanced attacks" if enable_attacks else "clean"
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)} ({attack_status})')
            print(f'  Training Mode: {training_mode}')
            print(f'  Loss: {losses["total_loss"].item():.4f}')
            print(f'  Watermark Quality - PSNR: {psnr:.2f}dB, SSIM: {ssim_val:.4f}')
            if recovered_images is not None:
                print(f'  Recovery Quality  - PSNR: {recovery_psnr:.2f}dB, SSIM: {recovery_ssim:.4f}')
            
            # Log detailed losses for message-image system
            print(f'  Message Loss: {losses["message_loss"].item():.4f}, '
                  f'Image Loss: {losses["image_mse"].item():.4f}')
            if 'recovery_mse' in losses:
                print(f'  Recovery MSE: {losses["recovery_mse"].item():.4f}, '
                      f'Recovery L1: {losses["recovery_l1"].item():.4f}')
            if 'vgg_recovery' in losses and losses['vgg_recovery'].item() > 0:
                print(f'  VGG Recovery: {losses["vgg_recovery"].item():.4f}')
                
            # Show message-to-image conversion info
            try:
                message_image = model.get_message_image_for_visualization(messages[:1])
                print(f'  Message Image Range: [{message_image.min():.3f}, {message_image.max():.3f}]')
            except:
                pass  # Skip if visualization fails
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_recovery_psnr = total_recovery_psnr / num_batches
    avg_recovery_ssim = total_recovery_ssim / num_batches
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'recovery_psnr': avg_recovery_psnr,
        'recovery_ssim': avg_recovery_ssim
    }


def enhanced_evaluate_model(model, dataloader, device, total_epochs, enable_attacks=False):
    """Enhanced evaluation with recovery quality tracking and MESSAGE-IMAGE robustness."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_recovery_psnr = 0.0
    total_recovery_ssim = 0.0
    correct_bits = 0
    total_bits = 0
    num_batches = 0
    
    # Use standard mode for evaluation
    training_mode = 'standard'
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            messages = torch.randint(0, 2, (batch_size, 30)).float().to(device)
            
            # Forward pass with message-to-image conversion
            losses, outputs = model.validate_on_batch([images, messages])
            watermarked, noised, decoded, recovered = outputs
            
            # Calculate image quality metrics
            psnr = compute_psnr(images, watermarked)
            ssim_val = compute_ssim(images, watermarked)
            
            # Recovery quality metrics
            recovery_psnr = 0.0
            recovery_ssim = 0.0
            if recovered is not None:
                recovery_psnr = compute_psnr(images, recovered)
                recovery_ssim = compute_ssim(images, recovered)
            
            total_psnr += psnr
            total_ssim += ssim_val
            total_recovery_psnr += recovery_psnr
            total_recovery_ssim += recovery_ssim
            
            # Calculate bit accuracy
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            
            decoded_sigmoid = torch.sigmoid(decoded)
            decoded_bits = (decoded_sigmoid > 0.5).float()
            correct_bits += (decoded_bits == messages).sum().item()
            total_bits += messages.numel()
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_recovery_psnr = total_recovery_psnr / num_batches
    avg_recovery_ssim = total_recovery_ssim / num_batches
    bit_accuracy = correct_bits / total_bits
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'recovery_psnr': avg_recovery_psnr,
        'recovery_ssim': avg_recovery_ssim,
        'bit_accuracy': bit_accuracy
    }


def main():
    """Enhanced training with MESSAGE-TO-IMAGE conversion and recovery quality optimization."""
    args = parse_arguments()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration with enhanced loss weights for recovery
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=args.message_length,
        encoder_channels=args.encoder_channels,
        encoder_blocks=4,
        decoder_channels=args.decoder_channels,
        decoder_blocks=7,
        use_discriminator=True,
        use_vgg=True,  # Enable VGG for better perceptual quality
        discriminator_blocks=3,
        discriminator_channels=64,
        encoder_loss=args.encoder_loss,
        decoder_loss=args.decoder_loss,
        adversarial_loss=args.adversarial_loss
    )
    
    # Create enhanced steganography model with MESSAGE-TO-IMAGE conversion
    print(f"Creating MESSAGE-TO-IMAGE enhanced ViT steganography system...")
    model = create_enhanced_steganography_model(
        config, device, use_robust_training=True
    )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer with REALISTIC learning rate for proper steganography
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Use specified learning rate
    
    # Create data loaders
    train_dataloader = create_data_loader(args.data_dir, args.batch_size, is_training=True)
    val_dataloader = create_data_loader(args.data_dir, args.batch_size, is_training=False)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"latent_runs/recovery_optimized_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Starting enhanced training with recovery optimization for {args.epochs} epochs...")
    
    # Create enhanced CSV with recovery metrics
    csv_path = os.path.join(output_dir, "enhanced_training_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_psnr', 'train_ssim', 
                     'train_recovery_psnr', 'train_recovery_ssim',
                     'eval_loss', 'eval_psnr', 'eval_ssim', 
                     'eval_recovery_psnr', 'eval_recovery_ssim',
                     'bit_accuracy', 'attacks_enabled']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Training loop
    best_loss = float('inf')
    best_psnr = 0.0
    best_recovery_psnr = 0.0  # Track best recovery quality
    attack_transition_epoch = 15
    
    print(f"Enhanced training plan:")
    print(f"  Epochs 1-{attack_transition_epoch}: Clean training (focus on watermark quality)")
    print(f"  Epochs {attack_transition_epoch+1}-{args.epochs}: Attack training (focus on recovery robustness)")
    
    for epoch in range(1, args.epochs + 1):
        enable_attacks = epoch > attack_transition_epoch
        attack_status = "with attacks" if enable_attacks else "clean"
        
        print(f"\n=== Epoch {epoch}/{args.epochs} ({attack_status}) ===")
        
        # Enhanced training and evaluation
        train_metrics = enhanced_train_epoch(
            model, train_dataloader, optimizer, device, epoch, True, enable_attacks
        )
        
        eval_metrics = enhanced_evaluate_model(
            model, val_dataloader, device, True, enable_attacks
        )
        
        # Enhanced results display
        print(f"Epoch {epoch} Results:")
        print(f"  Training Mode: {attack_status}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Watermark Quality - PSNR: {train_metrics['psnr']:.2f}dB, SSIM: {train_metrics['ssim']:.4f}")
        print(f"  Recovery Quality  - PSNR: {train_metrics['recovery_psnr']:.2f}dB, SSIM: {train_metrics['recovery_ssim']:.4f}")
        print(f"  Eval Recovery Quality - PSNR: {eval_metrics['recovery_psnr']:.2f}dB, SSIM: {eval_metrics['recovery_ssim']:.4f}")
        print(f"  Bit Accuracy: {eval_metrics['bit_accuracy']:.1%}")
        
        # Enhanced CSV logging
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_psnr': train_metrics['psnr'],
                'train_ssim': train_metrics['ssim'],
                'train_recovery_psnr': train_metrics['recovery_psnr'],
                'train_recovery_ssim': train_metrics['recovery_ssim'],
                'eval_loss': eval_metrics['loss'],
                'eval_psnr': eval_metrics['psnr'],
                'eval_ssim': eval_metrics['ssim'],
                'eval_recovery_psnr': eval_metrics['recovery_psnr'],
                'eval_recovery_ssim': eval_metrics['recovery_ssim'],
                'bit_accuracy': eval_metrics['bit_accuracy'],
                'attacks_enabled': enable_attacks
            })
        
        # Track best metrics including recovery
        if eval_metrics['recovery_psnr'] > best_recovery_psnr:
            best_recovery_psnr = eval_metrics['recovery_psnr']
        
        # Save checkpoint based on recovery quality
        if (eval_metrics['loss'] < best_loss or 
            eval_metrics['recovery_psnr'] > best_recovery_psnr - 1.0):  # Save if recovery is close to best
            if eval_metrics['loss'] < best_loss:
                best_loss = eval_metrics['loss']
                save_path = os.path.join(output_dir, f"best_model.pth")
                print(f"  New best model! Saving...")
            else:
                save_path = os.path.join(output_dir, f"good_recovery_epoch_{epoch}.pth")
                print(f"  Good recovery quality! Saving...")
            
            save_checkpoint(model, f"recovery_optimized", epoch, output_dir)
    
    print(f"\nEnhanced training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best recovery PSNR: {best_recovery_psnr:.2f}dB")
    print(f"Enhanced metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()

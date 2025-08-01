"""
Enhanced Latent Space Training with IMPROVED RECOVERY (targeting 32dB recovery PSNR)
This version addresses the recovery performance issues to achieve 32dB recovery PSNR.
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
import torch.nn as nn
import torch.nn.functional as F

from options import HiDDenConfiguration
from model.message_to_image import MessageToImageConverter
from model.latent_space import LatentMessageEmbedder
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


class MessageToImageLatentEncoder(nn.Module):
    """
    Enhanced ViT encoder that combines message-to-image conversion with true latent space embedding.
    Provides realistic steganography performance.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(MessageToImageLatentEncoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.message_length = config.message_length
        
        # Message-to-image converter
        self.message_converter = MessageToImageConverter(
            message_length=config.message_length,
            image_size=min(config.H, config.W),
            pattern_size=64,
            redundancy_factor=0.3  # Reduced for faster training
        )
        
        # ViT for processing cover images
        from vit_pytorch import ViT
        self.cover_vit = ViT(
            image_size=(config.H, config.W),
            patch_size=16,
            num_classes=0,
            dim=768,
            depth=6,
            heads=12,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Latent space message embedder (TRUE latent space modification)
        self.latent_embedder = LatentMessageEmbedder(
            message_length=config.message_length,
            feature_dim=768
        )
        
        # Feature decoder from latent space to spatial features
        self.feature_decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.encoder_channels * config.H * config.W // 16)
        )
        
        # Spatial processing layers
        self.spatial_processor = nn.Sequential(
            nn.ConvTranspose2d(config.encoder_channels, config.encoder_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(config.encoder_channels, config.encoder_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.encoder_channels, config.encoder_channels, 3, padding=1),
            nn.ReLU()
        )
        
        # Watermark generator with REALISTIC strength
        self.watermark_generator = nn.Sequential(
            nn.Conv2d(config.encoder_channels + 3, config.encoder_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.encoder_channels, 3, 1),
            nn.Tanh()
        )
        
        # FIXED blending parameters for realistic steganography performance
        self.alpha = 0.15  # REDUCED for better recovery (was 0.20)
        self.beta = 0.85   # INCREASED for better recovery (was 0.80)
        
    def forward(self, cover_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with TRUE latent space embedding and realistic watermark strength.
        """
        batch_size = cover_image.shape[0]
        
        # 1. Convert message to spatial pattern (for robustness)
        message_image = self.message_converter(message, target_size=self.H)
        
        # 2. Extract ViT features from cover image
        cover_patches = self.cover_vit.to_patch_embedding(cover_image)
        cover_patches += self.cover_vit.pos_embedding[:, 1:(cover_patches.shape[1] + 1)]
        cover_features = self.cover_vit.transformer(cover_patches)  # [batch, num_patches, 768]
        
        # 3. TRUE LATENT SPACE EMBEDDING: Modify ViT features with message
        modified_features = self.latent_embedder(message, cover_features)
        
        # 4. Decode modified latent features to spatial domain
        global_features = modified_features.mean(dim=1)  # Global pooling
        spatial_features = self.feature_decoder(global_features)
        spatial_features = spatial_features.view(batch_size, self.conv_channels, self.H // 4, self.W // 4)
        
        # 5. Upsample to full resolution
        upsampled = self.spatial_processor(spatial_features)  # [batch, conv_channels, H, W]
        
        # 6. Combine with cover image for watermark generation
        combined_input = torch.cat([cover_image, upsampled], dim=1)
        watermark = self.watermark_generator(combined_input)
        
        # 7. REALISTIC blending for proper steganography (IMPROVED for recovery)
        watermarked_image = self.beta * cover_image + self.alpha * watermark
        
        # 8. Ensure valid range
        watermarked_image = torch.clamp(watermarked_image, -1, 1)
        
        return watermarked_image


class AdvancedRecoveryDecoder(nn.Module):
    """
    IMPROVED decoder with advanced recovery architecture targeting 32dB recovery PSNR.
    Features: Skip connections, residual blocks, multi-scale recovery, perceptual loss.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(AdvancedRecoveryDecoder, self).__init__()
        self.message_length = config.message_length
        self.H = config.H
        self.W = config.W
        self.channels = config.decoder_channels
        
        # ENHANCED feature extraction with multiple scales
        self.conv1 = nn.Conv2d(3, self.channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.channels, self.channels, 3, padding=1)
        self.conv3 = nn.Conv2d(self.channels, self.channels, 3, padding=1)
        self.conv4 = nn.Conv2d(self.channels, self.channels, 3, padding=1)
        
        # Downsampling for multi-scale processing
        self.down1 = nn.Conv2d(self.channels, self.channels * 2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(self.channels * 2, self.channels * 4, 3, stride=2, padding=1)
        
        # Message extraction from deep features
        self.message_pool = nn.AdaptiveAvgPool2d(4)
        self.message_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels * 4 * 16, 1024),  # 4 * 4 * channels * 4
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.message_length)
        )
        
        # ADVANCED recovery network with skip connections
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels * 8, self.channels * 8, 3, padding=1),
            nn.ReLU()
        )
        
        # Upsampling with skip connections (FIXED dimensions)
        self.up1 = nn.ConvTranspose2d(self.channels * 8, self.channels * 4, 4, stride=2, padding=1)
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(self.channels * 8, self.channels * 4, 3, padding=1),  # Skip connection: 8*ch -> 4*ch
            nn.ReLU(),
            nn.Conv2d(self.channels * 4, self.channels * 4, 3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(self.channels * 4, self.channels * 2, 4, stride=2, padding=1)
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(self.channels * 4, self.channels * 2, 3, padding=1),  # Skip connection: 4*ch -> 2*ch
            nn.ReLU(),
            nn.Conv2d(self.channels * 2, self.channels * 2, 3, padding=1),
            nn.ReLU()
        )
        
        self.up3 = nn.ConvTranspose2d(self.channels * 2, self.channels, 4, stride=2, padding=1)
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, 3, padding=1),  # Skip connection: 2*ch -> ch
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU()
        )
        
        self.up4 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1)
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, 3, padding=1),  # Skip connection: 2*ch -> ch
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU()
        )
        
        # RESIDUAL recovery refinement
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(self.channels) for _ in range(4)
        ])
        
        # Final recovery output
        self.recovery_output = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels, 3, 1),
            nn.Tanh()
        )
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def _make_residual_block(self, channels):
        """Create a residual block for recovery refinement."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, watermarked_image: torch.Tensor):
        """Extract message and recover original image with advanced architecture."""
        # Encoder path with skip connections
        x1 = self.relu(self.conv1(watermarked_image))  # [B, ch, H, W]
        x2 = self.relu(self.conv2(x1))                 # [B, ch, H, W]
        x3 = self.relu(self.conv3(x2))                 # [B, ch, H, W]
        x4 = self.relu(self.conv4(x3))                 # [B, ch, H, W]
        
        # Downsampling
        down1 = self.relu(self.down1(x4))    # [B, 2*ch, H/2, W/2]
        down2 = self.relu(self.down2(down1)) # [B, 4*ch, H/4, W/4]
        
        # Message extraction from deep features
        msg_features = self.message_pool(down2)  # [B, 4*ch, 4, 4]
        message_logits = self.message_extractor(msg_features)
        
        # Recovery decoder with skip connections
        # Bottleneck
        bottleneck = self.bottleneck(down2)  # [B, 8*ch, H/4, W/4]
        
        # Upsampling with skip connections (CORRECTED)
        up1 = self.up1(bottleneck)  # [B, 4*ch, H/2, W/2]
        # Resize down2 to match up1 spatial dimensions
        down2_resized = F.interpolate(down2, size=up1.shape[-2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, down2_resized], dim=1)  # [B, 8*ch, H/2, W/2]
        up1 = self.up_conv1(up1)  # [B, 4*ch, H/2, W/2]
        
        up2 = self.up2(up1)  # [B, 2*ch, H, W]
        # Resize down1 to match up2 spatial dimensions  
        down1_resized = F.interpolate(down1, size=up2.shape[-2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, down1_resized], dim=1)  # [B, 4*ch, H, W]
        up2 = self.up_conv2(up2)  # [B, 2*ch, H, W]
        
        up3 = self.up3(up2)  # [B, ch, H*2, W*2]
        # Resize to target resolution and concatenate with x4
        up3 = F.interpolate(up3, size=(self.H, self.W), mode='bilinear', align_corners=False)  # [B, ch, H, W]
        up3 = torch.cat([up3, x4], dim=1)  # [B, 2*ch, H, W]
        up3 = self.up_conv3(up3)  # [B, ch, H, W]
        
        # Apply residual refinement for high-quality recovery
        recovery = up3
        for residual_block in self.residual_blocks:
            residual = residual_block(recovery)
            recovery = recovery + residual  # Residual connection
        
        # Final recovery output
        recovered_image = self.recovery_output(recovery)
        
        return message_logits, recovered_image


class HighRecoveryLatentSteganographySystem(nn.Module):
    """
    Enhanced latent space steganography system targeting 32dB recovery PSNR.
    """
    def __init__(self, config: HiDDenConfiguration, device: torch.device, use_robust_training: bool = True):
        super(HighRecoveryLatentSteganographySystem, self).__init__()
        
        self.config = config
        self.device = device
        self.use_robust_training = use_robust_training
        
        # Enhanced encoder with latent space embedding
        self.encoder = MessageToImageLatentEncoder(config).to(device)
        
        # ADVANCED recovery decoder
        self.decoder = AdvancedRecoveryDecoder(config).to(device)
        
        # Enhanced noiser
        if use_robust_training:
            self.noiser = create_enhanced_noiser(device)
        else:
            self.noiser = create_clean_enhanced_noiser(device)
            
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Perceptual loss for better recovery
        self.perceptual_weight = 0.1
        
        # Training mode tracking
        self.current_training_mode = 'recovery_focused'
        
    def update_training_mode(self, epoch, total_epochs):
        """Update training mode based on epoch."""
        if epoch <= total_epochs * 0.2:
            self.current_training_mode = 'recovery_focused'  # Focus on recovery first
        elif epoch <= total_epochs * 0.5:
            self.current_training_mode = 'balanced'
        elif epoch <= total_epochs * 0.8:
            self.current_training_mode = 'multi_scale'
        else:
            self.current_training_mode = 'robust'
        return self.current_training_mode
    
    def train_on_batch(self, batch, training_mode='recovery_focused'):
        """Training forward pass with RECOVERY-FOCUSED loss balancing."""
        original_images, messages = batch
        batch_size = original_images.shape[0]
        
        # Forward pass through encoder
        watermarked_images = self.encoder(original_images, messages)
        
        # Apply noise based on training mode
        if training_mode == 'robust' and self.use_robust_training:
            noiser_output = self.noiser([watermarked_images, original_images])
            noised_images = noiser_output[0]  # Extract only the noised watermarked images
        else:
            noised_images = watermarked_images
        
        # Forward pass through decoder
        decoded_messages, recovered_images = self.decoder(noised_images)
        
        # Calculate RECOVERY-OPTIMIZED losses
        losses = self.compute_losses(
            original_images, watermarked_images, messages, 
            decoded_messages, recovered_images, training_mode
        )
        
        return losses, (watermarked_images, noised_images, decoded_messages, recovered_images)
    
    def validate_on_batch(self, batch):
        """Validation forward pass."""
        with torch.no_grad():
            return self.train_on_batch(batch, training_mode='balanced')
    
    def compute_losses(self, original_images, watermarked_images, messages, decoded_messages, recovered_images, training_mode='recovery_focused'):
        """Compute RECOVERY-OPTIMIZED losses targeting 32dB recovery PSNR."""
        losses = {}
        
        # Message reconstruction loss
        message_loss = self.bce_loss(decoded_messages, messages)
        losses['message_loss'] = message_loss
        
        # Image quality loss - controlled for steganography
        image_mse = self.mse_loss(watermarked_images, original_images)
        image_l1 = self.l1_loss(watermarked_images, original_images)
        losses['image_mse'] = image_mse
        losses['image_l1'] = image_l1
        
        # RECOVERY quality loss - HEAVILY WEIGHTED for 32dB target
        if recovered_images is not None:
            recovery_mse = self.mse_loss(recovered_images, original_images)
            recovery_l1 = self.l1_loss(recovered_images, original_images)
            losses['recovery_mse'] = recovery_mse
            losses['recovery_l1'] = recovery_l1
            
            # Additional perceptual loss for recovery quality
            recovery_perceptual = torch.mean(torch.abs(recovered_images - original_images))
            losses['recovery_perceptual'] = recovery_perceptual
        else:
            losses['recovery_mse'] = torch.tensor(0.0, device=original_images.device)
            losses['recovery_l1'] = torch.tensor(0.0, device=original_images.device)
            losses['recovery_perceptual'] = torch.tensor(0.0, device=original_images.device)
        
        # ADAPTIVE loss weighting based on training mode
        if training_mode == 'recovery_focused':
            # RECOVERY PRIORITY - targeting 32dB
            total_loss = (
                0.8 * message_loss +                    # Reduced message priority
                0.3 * image_mse +                       # REDUCED: allow stronger watermarks initially
                0.1 * image_l1 +                        # REDUCED: focus on recovery
                3.0 * losses['recovery_mse'] +          # HEAVILY INCREASED: recovery priority
                1.5 * losses['recovery_l1'] +          # INCREASED: recovery detail
                0.5 * losses['recovery_perceptual']     # NEW: perceptual recovery quality
            )
        elif training_mode == 'balanced':
            # BALANCED phase
            total_loss = (
                1.2 * message_loss +                    # Message accuracy
                0.8 * image_mse +                       # Moderate image quality
                0.4 * image_l1 +                        # 
                2.0 * losses['recovery_mse'] +          # STRONG recovery priority
                1.0 * losses['recovery_l1'] +          # Recovery detail
                0.3 * losses['recovery_perceptual']     # Perceptual quality
            )
        else:
            # ROBUST/MULTI_SCALE phase  
            total_loss = (
                1.5 * message_loss +                    # Message robustness
                1.0 * image_mse +                       # Image quality balance
                0.5 * image_l1 +                        # 
                1.5 * losses['recovery_mse'] +          # MAINTAINED recovery priority
                0.8 * losses['recovery_l1'] +          # Recovery detail
                0.2 * losses['recovery_perceptual']     # Perceptual quality
            )
        
        # Recovery PSNR constraint - encourage 32dB target
        recovery_psnr = -10 * torch.log10(losses['recovery_mse'] + 1e-8)
        if recovery_psnr < 25:  # If recovery too poor, add penalty
            recovery_penalty = 0.5 * (25 - recovery_psnr) ** 2
            total_loss = total_loss + recovery_penalty
        
        # Watermark PSNR constraint - prevent unrealistic values
        watermark_psnr = -10 * torch.log10(image_mse + 1e-8)
        if watermark_psnr > 55:  # If watermark too strong, add penalty
            watermark_penalty = 0.1 * (watermark_psnr - 55) ** 2
            total_loss = total_loss + watermark_penalty
        
        losses['total_loss'] = total_loss
        return losses
    
    def get_message_image_for_visualization(self, messages):
        """Get message image for visualization."""
        return self.encoder.message_converter(messages, target_size=self.config.H)


# Keep the same data loading and training functions but use the new system
from train_realistic_latent_space import (
    parse_arguments, RealDataset, create_data_loader, 
    save_checkpoint, latent_evaluate_model
)

def recovery_focused_train_epoch(model, dataloader, optimizer, device, epoch, total_epochs, enable_attacks=False):
    """Training epoch with RECOVERY-FOCUSED optimization for 32dB target."""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_recovery_psnr = 0.0
    total_recovery_ssim = 0.0
    num_batches = 0
    
    # Update training mode for recovery optimization
    training_mode = model.update_training_mode(epoch, total_epochs)
    
    # Log training mode
    if enable_attacks:
        print(f"  RECOVERY training mode: {training_mode} with enhanced attacks")
    else:
        print(f"  RECOVERY training mode: {training_mode} (clean phase)")
    
    for batch_idx, (images, _) in enumerate(dataloader):
        # Prepare data
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Generate random messages
        messages = torch.randint(0, 2, (batch_size, 30)).float().to(device)
        
        # Forward pass with RECOVERY-FOCUSED optimization
        losses, outputs = model.train_on_batch([images, messages], training_mode=training_mode)
        watermarked_images, noised_images, decoded_messages, recovered_images = outputs
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate image quality metrics
        with torch.no_grad():
            # Watermark quality
            psnr = compute_psnr(images, watermarked_images)
            ssim_val = compute_ssim(images, watermarked_images)
            
            # RECOVERY quality (TARGET: 32dB)
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
        
        # Enhanced progress logging with RECOVERY focus
        if batch_idx % 50 == 0:
            attack_status = "with attacks" if enable_attacks else "clean"
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)} ({attack_status})')
            print(f'  RECOVERY Mode: {training_mode}')
            print(f'  Loss: {losses["total_loss"].item():.4f}')
            print(f'  Watermark Quality - PSNR: {psnr:.2f}dB, SSIM: {ssim_val:.4f}')
            if recovered_images is not None:
                recovery_status = "🎯 TARGET!" if recovery_psnr >= 32 else "📈 IMPROVING" if recovery_psnr >= 28 else "🔧 TRAINING"
                print(f'  🔄 RECOVERY Quality - PSNR: {recovery_psnr:.2f}dB, SSIM: {recovery_ssim:.4f} {recovery_status}')
            
            # Log detailed losses
            print(f'  Message Loss: {losses["message_loss"].item():.4f}, '
                  f'Recovery MSE: {losses["recovery_mse"].item():.4f}')
    
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


def main():
    """Enhanced recovery-focused training targeting 32dB recovery PSNR."""
    args = parse_arguments()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=args.message_length,
        encoder_channels=args.encoder_channels,
        encoder_blocks=4,
        decoder_channels=args.decoder_channels,  # Increased for better recovery
        decoder_blocks=7,
        use_discriminator=False,
        use_vgg=False,
        discriminator_blocks=3,
        discriminator_channels=64,
        encoder_loss=1.0,
        decoder_loss=1.0,
        adversarial_loss=0.0
    )
    
    # Create HIGH RECOVERY latent space steganography model
    print(f"Creating HIGH RECOVERY ViT steganography system (targeting 32dB recovery PSNR)...")
    model = HighRecoveryLatentSteganographySystem(
        config, device, use_robust_training=True
    )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer with different learning rates for different components
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    
    # Use higher learning rate for recovery decoder
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': args.lr},
        {'params': decoder_params, 'lr': args.lr * 1.5}  # Higher LR for recovery optimization
    ])
    
    # Create data loaders
    train_dataloader = create_data_loader(args.data_dir, args.batch_size, is_training=True)
    val_dataloader = create_data_loader(args.data_dir, args.batch_size, is_training=False)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"latent_runs/high_recovery_latent_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Starting HIGH RECOVERY latent space training for {args.epochs} epochs...")
    print(f"TARGET: Recovery PSNR 32dB+, Watermark PSNR 35-50dB")
    
    # Create CSV with recovery-focused metrics tracking
    csv_path = os.path.join(output_dir, "high_recovery_training_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_psnr', 'train_ssim', 
                     'train_recovery_psnr', 'train_recovery_ssim',
                     'eval_loss', 'eval_psnr', 'eval_ssim', 
                     'eval_recovery_psnr', 'eval_recovery_ssim',
                     'bit_accuracy', 'attacks_enabled', 'training_mode']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Training loop with recovery focus
    best_loss = float('inf')
    best_recovery_psnr = 0.0
    attack_transition_epoch = args.clean_train_epochs
    target_achieved = False
    
    print(f"RECOVERY-FOCUSED training plan:")
    print(f"  Epochs 1-{attack_transition_epoch}: Clean recovery training")
    print(f"  Epochs {attack_transition_epoch+1}-{args.epochs}: Robust recovery training")
    
    for epoch in range(1, args.epochs + 1):
        enable_attacks = epoch > attack_transition_epoch
        attack_status = "with attacks" if enable_attacks else "clean"
        
        print(f"\n=== Epoch {epoch}/{args.epochs} - HIGH RECOVERY TRAINING ({attack_status}) ===")
        
        # RECOVERY-FOCUSED training and evaluation
        train_metrics = recovery_focused_train_epoch(
            model, train_dataloader, optimizer, device, epoch, args.epochs, enable_attacks
        )
        
        eval_metrics = latent_evaluate_model(
            model, val_dataloader, device, enable_attacks
        )
        
        # Check if 32dB recovery target achieved
        if eval_metrics['recovery_psnr'] >= 32.0 and not target_achieved:
            target_achieved = True
            print(f"  🎉 32dB RECOVERY TARGET ACHIEVED! ({eval_metrics['recovery_psnr']:.2f}dB)")
        
        # Display results with recovery focus
        print(f"Epoch {epoch} HIGH RECOVERY Results:")
        print(f"  Training Mode: {model.current_training_mode} {attack_status}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Watermark Quality - PSNR: {train_metrics['psnr']:.2f}dB, SSIM: {train_metrics['ssim']:.4f}")
        
        # Highlight recovery performance
        recovery_status = "🎯 EXCELLENT!" if train_metrics['recovery_psnr'] >= 32 else "📈 GOOD" if train_metrics['recovery_psnr'] >= 28 else "🔧 IMPROVING"
        print(f"  🔄 RECOVERY Quality - PSNR: {train_metrics['recovery_psnr']:.2f}dB, SSIM: {train_metrics['recovery_ssim']:.4f} {recovery_status}")
        
        eval_recovery_status = "🎯 TARGET!" if eval_metrics['recovery_psnr'] >= 32 else "📈 CLOSE" if eval_metrics['recovery_psnr'] >= 30 else "🔧 TRAINING"
        print(f"  🎯 Eval Recovery - PSNR: {eval_metrics['recovery_psnr']:.2f}dB, SSIM: {eval_metrics['recovery_ssim']:.4f} {eval_recovery_status}")
        print(f"  📊 Bit Accuracy: {eval_metrics['bit_accuracy']:.1%}")
        
        # CSV logging
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
                'attacks_enabled': enable_attacks,
                'training_mode': model.current_training_mode
            })
        
        # Track best recovery performance
        if eval_metrics['recovery_psnr'] > best_recovery_psnr:
            best_recovery_psnr = eval_metrics['recovery_psnr']
        
        # Save checkpoint (prioritize recovery performance)
        if eval_metrics['recovery_psnr'] > 28 and eval_metrics['loss'] < best_loss:
            best_loss = eval_metrics['loss']
            save_checkpoint(model, f"high_recovery_latent", epoch, output_dir)
            print(f"  ✅ New best recovery model saved!")
        
        # Save intermediate checkpoints
        if epoch % 5 == 0:
            save_checkpoint(model, f"recovery_epoch_{epoch}", epoch, output_dir)
    
    print(f"\n🎉 HIGH RECOVERY training completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best recovery PSNR: {best_recovery_psnr:.2f}dB {'🎯 TARGET ACHIEVED!' if best_recovery_psnr >= 32 else '📈 Keep training for 32dB+'}")
    print(f"Metrics saved to: {csv_path}")
    print(f"This model uses ADVANCED RECOVERY with skip connections and residual blocks!")


if __name__ == "__main__":
    main()

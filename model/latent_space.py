"""
Latent Space Steganography Implementation for ViT Project
This module adds latent space embedding capabilities to your existing architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from vit_pytorch import ViT


class LatentMessageEmbedder(nn.Module):
    """
    Embeds secret messages into ViT feature space.
    This is inspired by RosteALS but adapted for your ViT architecture.
    """
    def __init__(self, message_length, feature_dim):
        super(LatentMessageEmbedder, self).__init__()
        self.message_length = message_length
        self.feature_dim = feature_dim
        
        # Message encoding network (lightweight)
        self.message_encoder = nn.Sequential(
            nn.Linear(message_length, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
            nn.Tanh()  # Bounded output for stable training
        )
        
        # Feature modulation network
        self.feature_modulator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()  # Modulation weights
        )
        
        # Initialize with small weights for stability
        for module in [self.message_encoder, self.feature_modulator]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, message, vit_features):
        """
        Args:
            message: [batch_size, message_length] - Secret message
            vit_features: [batch_size, num_patches, feature_dim] - ViT features
        Returns:
            modified_features: [batch_size, num_patches, feature_dim] - Modified features
        """
        batch_size, num_patches, feature_dim = vit_features.shape
        
        # Encode message to feature space
        message_embedding = self.message_encoder(message)  # [batch, feature_dim]
        
        # Generate modulation weights
        modulation = self.feature_modulator(message_embedding)  # [batch, feature_dim]
        
        # Expand for broadcasting
        message_embedding = message_embedding.unsqueeze(1)  # [batch, 1, feature_dim]
        modulation = modulation.unsqueeze(1)  # [batch, 1, feature_dim]
        
        # Apply latent space modification: f' = f + α * m
        # where α is the modulation weight and m is the message embedding
        modified_features = vit_features + modulation * message_embedding
        
        return modified_features


class LatentSpaceEncoder(nn.Module):
    """
    Enhanced encoder that embeds messages in ViT latent space.
    Integrates with your existing architecture.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(LatentSpaceEncoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.message_length = config.message_length
        
        # Your existing ViT architecture
        self.vit = ViT(
            image_size=(config.H, config.W),
            patch_size=32,
            num_classes=0,  # No classification head
            dim=1024,
            depth=config.decoder_blocks // 2,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Latent space message embedder
        self.message_embedder = LatentMessageEmbedder(
            message_length=config.message_length,
            feature_dim=1024  # ViT dimension
        )
        
        # Feature-to-image decoder (similar to your existing architecture)
        self.feature_decoder = nn.Sequential(
            nn.Linear(1024, self.conv_channels * 4 * 4),  # Start with 4x4 spatial
            nn.ReLU()
        )
        
        # Convolutional upsampling (matching your style)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.conv_channels, self.conv_channels, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(self.conv_channels, self.conv_channels, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(self.conv_channels, self.conv_channels, 4, 2, 1),  # 16x16 -> 32x32
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(self.conv_channels, self.conv_channels, 4, 2, 1),  # 32x32 -> 64x64
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(self.conv_channels, self.conv_channels, 4, 2, 1),  # 64x64 -> 128x128
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU()
        )
        
        # Final layer to produce watermarked image
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=3, padding=1)
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, image, message):
        """
        Latent space embedding forward pass.
        """
        batch_size = image.shape[0]
        
        # Extract ViT features
        vit_features = self.extract_vit_features(image)  # [batch, num_patches, 1024]
        
        # Embed message in latent space
        modified_features = self.message_embedder(message, vit_features)
        
        # Convert back to image space
        # Average pool features to get global representation
        global_features = modified_features.mean(dim=1)  # [batch, 1024]
        
        # Decode to spatial features
        spatial_features = self.feature_decoder(global_features)  # [batch, conv_channels * 16]
        spatial_features = spatial_features.view(batch_size, self.conv_channels, 4, 4)
        
        # Upsample to full resolution
        upsampled = self.conv_layers(spatial_features)  # [batch, conv_channels, 128, 128]
        
        # Generate watermark
        watermark = self.final_layer(upsampled)  # [batch, 3, 128, 128]
        
        # Apply residual connection: output = input + α * watermark
        watermarked_image = image + self.residual_weight * watermark
        
        # Ensure output is in valid range
        watermarked_image = torch.tanh(watermarked_image)
        
        return watermarked_image
    
    def extract_vit_features(self, image):
        """Extract ViT features maintaining patch structure."""
        # ViT patch embedding
        patches = self.vit.to_patch_embedding(image)  # [batch, num_patches, dim]
        
        # Add positional embedding
        patches += self.vit.pos_embedding[:, 1:(patches.shape[1] + 1)]
        patches = self.vit.dropout(patches)
        
        # Pass through transformer layers
        features = self.vit.transformer(patches)
        
        return features


class LatentSpaceDecoder(nn.Module):
    """
    Decoder that extracts messages from latent space modified features.
    Works with your existing decoder architecture.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(LatentSpaceDecoder, self).__init__()
        self.message_length = config.message_length
        
        # Feature extractor (similar to your decoder)
        self.feature_extractor = nn.Sequential(
            ConvBNRelu(3, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
        )
        
        # Global pooling and message extraction
        self.message_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.decoder_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.message_length)
        )
        
        # Image recovery branch (optional)
        self.image_recovery = nn.Sequential(
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            ConvBNRelu(config.decoder_channels, config.decoder_channels),
            nn.Conv2d(config.decoder_channels, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, watermarked_image):
        """
        Extract message from watermarked image.
        """
        # Extract features
        features = self.feature_extractor(watermarked_image)
        
        # Extract message
        message_logits = self.message_extractor(features)
        
        # Recover image (optional)
        recovered_image = self.image_recovery(features)
        
        return message_logits, recovered_image


def create_latent_space_model(config: HiDDenConfiguration):
    """
    Factory function to create a complete latent space steganography model.
    """
    encoder = LatentSpaceEncoder(config)
    decoder = LatentSpaceDecoder(config)
    
    return encoder, decoder


# Integration with your existing training loop
class LatentSpaceLoss(nn.Module):
    """
    Enhanced loss function for latent space steganography with recovery quality optimization.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(LatentSpaceLoss, self).__init__()
        self.message_loss_weight = config.decoder_loss
        self.image_loss_weight = config.encoder_loss
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, original_image, watermarked_image, original_message, 
                decoded_message, recovered_image=None):
        """
        Compute enhanced loss with recovery quality optimization.
        """
        # Message reconstruction loss (primary objective)
        message_loss = self.bce_loss(decoded_message, original_message)
        
        # Image quality loss (watermark imperceptibility)
        image_loss = self.mse_loss(watermarked_image, original_image)
        
        # Initialize total loss
        total_loss = (self.message_loss_weight * message_loss + 
                     self.image_loss_weight * image_loss)
        
        losses = {
            'message_loss': message_loss,
            'image_loss': image_loss,
            'total_loss': total_loss
        }
        
        # Recovery quality loss (NEW - key for your goal!)
        if recovered_image is not None:
            # MSE loss for pixel-level accuracy
            recovery_mse = self.mse_loss(recovered_image, original_image)
            
            # L1 loss for sharpness and detail preservation
            recovery_l1 = self.l1_loss(recovered_image, original_image)
            
            # SSIM-like loss for structural similarity
            recovery_structural = self.compute_structural_loss(recovered_image, original_image)
            
            # Add recovery losses with significant weight
            recovery_total = recovery_mse + 0.5 * recovery_l1 + 0.3 * recovery_structural
            
            losses['recovery_mse'] = recovery_mse
            losses['recovery_l1'] = recovery_l1
            losses['recovery_structural'] = recovery_structural
            losses['recovery_total'] = recovery_total
            
            # Add to total loss with high importance (2.0 weight)
            losses['total_loss'] += 2.0 * recovery_total
        
        return losses
    
    def compute_structural_loss(self, pred, target):
        """
        Compute a structural similarity loss (simplified SSIM-like).
        """
        # Gaussian blur for local structure
        pred_blur = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        target_blur = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        
        # Mean and variance
        mu_pred = pred_blur
        mu_target = target_blur
        
        # Structural similarity component
        structural_diff = torch.mean((mu_pred - mu_target) ** 2)
        
        return structural_diff
        
        # Optional image recovery loss
        recovery_loss = 0.0
        if recovered_image is not None:
            recovery_loss = self.mse_loss(recovered_image, original_image)
        
        # Combined loss
        total_loss = (self.message_loss_weight * message_loss + 
                     self.image_loss_weight * image_loss + 
                     0.1 * recovery_loss)
        
        return {
            'total_loss': total_loss,
            'message_loss': message_loss,
            'image_loss': image_loss,
            'recovery_loss': recovery_loss
        }


if __name__ == "__main__":
    # Test the latent space implementation
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=30,
        encoder_channels=32,
        encoder_blocks=4,
        decoder_channels=32,
        decoder_blocks=7
    )
    
    # Create models
    encoder, decoder = create_latent_space_model(config)
    
    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 3, 128, 128)
    message = torch.randint(0, 2, (batch_size, 30)).float()
    
    # Encode
    watermarked = encoder(image, message)
    print(f"Watermarked shape: {watermarked.shape}")
    
    # Decode
    decoded_message, recovered_image = decoder(watermarked)
    print(f"Decoded message shape: {decoded_message.shape}")
    print(f"Recovered image shape: {recovered_image.shape}")
    
    # Test loss
    loss_fn = LatentSpaceLoss(config)
    losses = loss_fn(image, watermarked, message, decoded_message, recovered_image)
    print(f"Total loss: {losses['total_loss'].item():.4f}")

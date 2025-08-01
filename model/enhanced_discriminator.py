"""
Enhanced Discriminator for Message-Image Based Steganography
Provides improved security analysis and adversarial training for the ViT-based system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from vit_pytorch import ViT


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that analyzes images at different resolutions
    to detect steganographic content more effectively.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_scales = 3
        self.channels = config.discriminator_channels
        
        # Discriminators for different scales
        self.discriminators = nn.ModuleList()
        
        for i in range(self.num_scales):
            discriminator = nn.Sequential(
                # Initial convolution
                nn.Conv2d(3, self.channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                
                # Progressive downsampling
                self._make_layer(self.channels, self.channels * 2, 4, 2, 1),
                self._make_layer(self.channels * 2, self.channels * 4, 4, 2, 1),
                self._make_layer(self.channels * 4, self.channels * 8, 4, 2, 1),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                # Classification head
                nn.Linear(self.channels * 8, self.channels * 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.5),
                nn.Linear(self.channels * 2, 1)
            )
            self.discriminators.append(discriminator)
            
        # Downsampling layers for multi-scale input
        self.downsamplers = nn.ModuleList([
            nn.Identity(),  # Original scale
            nn.AvgPool2d(2, 2),  # 1/2 scale
            nn.AvgPool2d(4, 4)   # 1/4 scale
        ])
        
        # Feature fusion for final decision
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.num_scales, self.num_scales * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.num_scales * 2, 1),
            nn.Sigmoid()
        )
        
    def _make_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """Create a discriminator layer with normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, image: torch.Tensor) -> tuple:
        """
        Multi-scale discrimination.
        
        Args:
            image: [batch_size, 3, H, W] - Input image
        Returns:
            tuple: (final_score, scale_scores)
                final_score: [batch_size, 1] - Final discrimination score
                scale_scores: List of [batch_size, 1] - Individual scale scores
        """
        scale_scores = []
        
        for i, (discriminator, downsampler) in enumerate(zip(self.discriminators, self.downsamplers)):
            # Downsample input
            scaled_input = downsampler(image)
            
            # Discriminate at this scale
            score = discriminator(scaled_input)
            scale_scores.append(score)
        
        # Combine scale scores
        combined_scores = torch.cat(scale_scores, dim=1)  # [batch, num_scales]
        final_score = self.fusion_layer(combined_scores)
        
        return final_score, scale_scores


class ViTDiscriminator(nn.Module):
    """
    Vision Transformer-based discriminator for detecting steganographic content.
    Leverages attention mechanisms to focus on subtle artifacts.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(ViTDiscriminator, self).__init__()
        
        # ViT backbone for feature extraction
        self.vit = ViT(
            image_size=(config.H, config.W),
            patch_size=16,
            num_classes=0,  # No classification head
            dim=384,        # Smaller than encoder for efficiency
            depth=6,
            heads=6,
            mlp_dim=1536,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Attention-based feature aggregation
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=384,
            num_heads=6,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Patch-level analysis for localization
        self.patch_classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image: torch.Tensor) -> tuple:
        """
        ViT-based discrimination with patch-level analysis.
        
        Args:
            image: [batch_size, 3, H, W] - Input image
        Returns:
            tuple: (global_score, patch_scores, attention_weights)
        """
        # Extract ViT features
        patches = self.vit.to_patch_embedding(image)
        patches += self.vit.pos_embedding[:, 1:(patches.shape[1] + 1)]
        vit_features = self.vit.transformer(patches)  # [batch, num_patches, 384]
        
        # Attention-based global feature
        attended_features, attention_weights = self.attention_pooling(
            vit_features, vit_features, vit_features
        )
        global_feature = attended_features.mean(dim=1)  # [batch, 384]
        
        # Global classification
        global_score = self.classifier(global_feature)
        
        # Patch-level classification
        patch_scores = self.patch_classifier(vit_features)  # [batch, num_patches, 1]
        
        return global_score, patch_scores, attention_weights


class HybridDiscriminator(nn.Module):
    """
    Hybrid discriminator combining CNN and ViT approaches for comprehensive analysis.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(HybridDiscriminator, self).__init__()
        
        # Multi-scale CNN discriminator
        self.cnn_discriminator = MultiScaleDiscriminator(config)
        
        # ViT discriminator
        self.vit_discriminator = ViTDiscriminator(config)
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(2, 8),  # CNN + ViT scores
            nn.LeakyReLU(0.2),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.2),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image: torch.Tensor) -> dict:
        """
        Hybrid discrimination with confidence estimation.
        
        Args:
            image: [batch_size, 3, H, W] - Input image
        Returns:
            dict: Comprehensive discrimination results
        """
        # CNN discrimination
        cnn_score, cnn_scale_scores = self.cnn_discriminator(image)
        
        # ViT discrimination
        vit_score, patch_scores, attention_weights = self.vit_discriminator(image)
        
        # Combine scores
        combined_input = torch.cat([cnn_score, vit_score], dim=1)
        final_score = self.fusion_net(combined_input)
        confidence = self.confidence_estimator(combined_input)
        
        return {
            'final_score': final_score,
            'cnn_score': cnn_score,
            'vit_score': vit_score,
            'cnn_scale_scores': cnn_scale_scores,
            'patch_scores': patch_scores,
            'attention_weights': attention_weights,
            'confidence': confidence
        }


class AdversarialLoss(nn.Module):
    """
    Enhanced adversarial loss with multiple discrimination strategies.
    """
    def __init__(self, config: HiDDenConfiguration, loss_type: str = 'hybrid'):
        super(AdversarialLoss, self).__init__()
        
        self.loss_type = loss_type
        
        if loss_type == 'hybrid':
            self.discriminator = HybridDiscriminator(config)
        elif loss_type == 'vit':
            self.discriminator = ViTDiscriminator(config)
        elif loss_type == 'multiscale':
            self.discriminator = MultiScaleDiscriminator(config)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def discriminator_loss(self, cover_images: torch.Tensor, 
                          watermarked_images: torch.Tensor) -> dict:
        """
        Calculate discriminator loss.
        
        Args:
            cover_images: [batch_size, 3, H, W] - Original images
            watermarked_images: [batch_size, 3, H, W] - Watermarked images
        Returns:
            dict: Discriminator losses
        """
        batch_size = cover_images.shape[0]
        device = cover_images.device
        
        # Real labels (cover images should be classified as real)
        real_labels = torch.ones(batch_size, 1, device=device)
        # Fake labels (watermarked images should be classified as fake)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        if self.loss_type == 'hybrid':
            # Discriminate cover images (real)
            real_results = self.discriminator(cover_images)
            real_loss = self.bce_loss(real_results['final_score'], real_labels)
            
            # Discriminate watermarked images (fake)
            fake_results = self.discriminator(watermarked_images.detach())
            fake_loss = self.bce_loss(fake_results['final_score'], fake_labels)
            
            # Additional losses
            confidence_loss = self.mse_loss(real_results['confidence'], real_labels) + \
                            self.mse_loss(fake_results['confidence'], torch.ones_like(fake_labels))
            
            total_loss = real_loss + fake_loss + 0.1 * confidence_loss
            
            return {
                'total_loss': total_loss,
                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'confidence_loss': confidence_loss
            }
            
        elif self.loss_type == 'multiscale':
            # Multi-scale discrimination
            real_score, real_scale_scores = self.discriminator(cover_images)
            fake_score, fake_scale_scores = self.discriminator(watermarked_images.detach())
            
            # Main loss
            real_loss = self.bce_loss(real_score, real_labels)
            fake_loss = self.bce_loss(fake_score, fake_labels)
            
            # Scale-specific losses
            scale_loss = 0
            for real_scale, fake_scale in zip(real_scale_scores, fake_scale_scores):
                scale_loss += self.bce_loss(torch.sigmoid(real_scale), real_labels)
                scale_loss += self.bce_loss(torch.sigmoid(fake_scale), fake_labels)
            
            total_loss = real_loss + fake_loss + 0.1 * scale_loss
            
            return {
                'total_loss': total_loss,
                'real_loss': real_loss,
                'fake_loss': fake_loss,
                'scale_loss': scale_loss
            }
            
        else:  # ViT discriminator
            real_score, _, _ = self.discriminator(cover_images)
            fake_score, _, _ = self.discriminator(watermarked_images.detach())
            
            real_loss = self.bce_loss(torch.sigmoid(real_score), real_labels)
            fake_loss = self.bce_loss(torch.sigmoid(fake_score), fake_labels)
            
            total_loss = real_loss + fake_loss
            
            return {
                'total_loss': total_loss,
                'real_loss': real_loss,
                'fake_loss': fake_loss
            }
    
    def generator_loss(self, watermarked_images: torch.Tensor) -> dict:
        """
        Calculate generator (encoder) loss.
        
        Args:
            watermarked_images: [batch_size, 3, H, W] - Watermarked images
        Returns:
            dict: Generator losses
        """
        batch_size = watermarked_images.shape[0]
        device = watermarked_images.device
        
        # Generator wants watermarked images to be classified as real
        real_labels = torch.ones(batch_size, 1, device=device)
        
        if self.loss_type == 'hybrid':
            results = self.discriminator(watermarked_images)
            generator_loss = self.bce_loss(results['final_score'], real_labels)
            
            # Encourage high confidence in "real" classification
            confidence_loss = self.mse_loss(results['confidence'], real_labels)
            
            total_loss = generator_loss + 0.1 * confidence_loss
            
            return {
                'total_loss': total_loss,
                'generator_loss': generator_loss,
                'confidence_loss': confidence_loss
            }
            
        elif self.loss_type == 'multiscale':
            score, scale_scores = self.discriminator(watermarked_images)
            generator_loss = self.bce_loss(score, real_labels)
            
            # Scale-specific generator losses
            scale_loss = 0
            for scale_score in scale_scores:
                scale_loss += self.bce_loss(torch.sigmoid(scale_score), real_labels)
            
            total_loss = generator_loss + 0.1 * scale_loss
            
            return {
                'total_loss': total_loss,
                'generator_loss': generator_loss,
                'scale_loss': scale_loss
            }
            
        else:  # ViT discriminator
            score, _, _ = self.discriminator(watermarked_images)
            generator_loss = self.bce_loss(torch.sigmoid(score), real_labels)
            
            return {
                'total_loss': generator_loss,
                'generator_loss': generator_loss
            }


def test_enhanced_discriminators():
    """Test the enhanced discriminator components."""
    from options import HiDDenConfiguration
    
    # Create test configuration
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=30,
        encoder_channels=64,
        encoder_blocks=4,
        decoder_channels=64,
        decoder_blocks=7,
        use_discriminator=True,
        use_vgg=True,
        discriminator_blocks=3,
        discriminator_channels=64,
        encoder_loss=1.0,
        decoder_loss=1.0,
        adversarial_loss=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size = 2
    cover_images = torch.randn(batch_size, 3, 128, 128).to(device)
    watermarked_images = torch.randn(batch_size, 3, 128, 128).to(device)
    
    print("Testing Enhanced Discriminators")
    print(f"Cover images shape: {cover_images.shape}")
    print(f"Watermarked images shape: {watermarked_images.shape}")
    
    # Test MultiScale Discriminator
    print("\n=== MultiScale Discriminator ===")
    multiscale_disc = MultiScaleDiscriminator(config).to(device)
    ms_score, ms_scale_scores = multiscale_disc(cover_images)
    print(f"Final score shape: {ms_score.shape}")
    print(f"Number of scale scores: {len(ms_scale_scores)}")
    for i, score in enumerate(ms_scale_scores):
        print(f"Scale {i} score shape: {score.shape}")
    
    # Test ViT Discriminator
    print("\n=== ViT Discriminator ===")
    vit_disc = ViTDiscriminator(config).to(device)
    vit_global, vit_patches, vit_attention = vit_disc(cover_images)
    print(f"Global score shape: {vit_global.shape}")
    print(f"Patch scores shape: {vit_patches.shape}")
    print(f"Attention weights shape: {vit_attention.shape}")
    
    # Test Hybrid Discriminator
    print("\n=== Hybrid Discriminator ===")
    hybrid_disc = HybridDiscriminator(config).to(device)
    hybrid_results = hybrid_disc(cover_images)
    print(f"Final score shape: {hybrid_results['final_score'].shape}")
    print(f"CNN score shape: {hybrid_results['cnn_score'].shape}")
    print(f"ViT score shape: {hybrid_results['vit_score'].shape}")
    print(f"Confidence shape: {hybrid_results['confidence'].shape}")
    
    # Test Adversarial Loss
    print("\n=== Adversarial Loss ===")
    adv_loss = AdversarialLoss(config, loss_type='hybrid').to(device)
    
    # Discriminator loss
    disc_losses = adv_loss.discriminator_loss(cover_images, watermarked_images)
    print("Discriminator losses:")
    for key, value in disc_losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Generator loss
    gen_losses = adv_loss.generator_loss(watermarked_images)
    print("Generator losses:")
    for key, value in gen_losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\nDiscriminator testing completed successfully!")
    return multiscale_disc, vit_disc, hybrid_disc, adv_loss


if __name__ == "__main__":
    discriminators = test_enhanced_discriminators()

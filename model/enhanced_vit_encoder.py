"""
Enhanced ViT Encoder with Message-Image Integration
Adapts the existing ViT steganography encoder to work with message images
for improved robustness against geometric attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.message_to_image import MessageToImageConverter, ResizeRobustMessageImage
from vit_pytorch import ViT
from transformers import ViTFeatureExtractor, ViTModel


class MessageImageViTEncoder(nn.Module):
    """
    Enhanced ViT encoder that uses message-to-image conversion
    for robust steganographic embedding.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(MessageImageViTEncoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        self.message_length = config.message_length
        
        # Message-to-image converter
        self.message_converter = MessageToImageConverter(
            message_length=config.message_length,
            image_size=min(config.H, config.W),
            pattern_size=64,
            redundancy_factor=0.5
        )
        
        # Resize-robust message converter for testing different scales
        self.robust_converter = ResizeRobustMessageImage(
            message_length=config.message_length,
            base_size=min(config.H, config.W)
        )
        
        # ViT for processing cover images
        self.cover_vit = ViT(
            image_size=(config.H, config.W),
            patch_size=16,  # Smaller patches for finer detail
            num_classes=0,  # No classification
            dim=768,       # Standard ViT dimension
            depth=6,       # Moderate depth
            heads=12,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # ViT for processing message images
        self.message_vit = ViT(
            image_size=(config.H, config.W),
            patch_size=16,
            num_classes=0,
            dim=768,
            depth=4,       # Lighter processing for message
            heads=8,
            mlp_dim=1536,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Fusion network to combine cover and message features
        self.feature_fusion = nn.Sequential(
            nn.Linear(768 * 2, 1024),  # Combine both ViT outputs
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, config.encoder_channels * config.H * config.W // 16)
        )
        
        # Convolutional layers for spatial processing
        conv_layers = []
        for i in range(config.encoder_blocks):
            if i == 0:
                conv_layers.append(ConvBNRelu(3 + config.encoder_channels, config.encoder_channels))
            else:
                conv_layers.append(ConvBNRelu(config.encoder_channels, config.encoder_channels))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Final watermark generation
        self.watermark_generator = nn.Sequential(
            nn.Conv2d(config.encoder_channels, config.encoder_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.encoder_channels // 2, 3, 1),
            nn.Tanh()
        )
        
        # REALISTIC blending weight for proper steganography (30-45dB PSNR)
        self.blend_weight = nn.Parameter(torch.tensor(0.25))  # Increased from 0.1 for visible watermark
        
        # Scale adaptation for robustness
        self.scale_adapter = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, cover_image: torch.Tensor, message: torch.Tensor, 
                training_mode: str = 'standard') -> torch.Tensor:
        """
        Forward pass with message-image conversion.
        
        Args:
            cover_image: [batch_size, 3, H, W] - Cover image
            message: [batch_size, message_length] - Secret message bits
            training_mode: 'standard', 'multi_scale', or 'robust'
        Returns:
            watermarked_image: [batch_size, 3, H, W] - Watermarked image
        """
        batch_size = cover_image.shape[0]
        
        if training_mode == 'multi_scale':
            return self.forward_multi_scale(cover_image, message)
        elif training_mode == 'robust':
            return self.forward_robust(cover_image, message)
        else:
            return self.forward_standard(cover_image, message)
    
    def forward_standard(self, cover_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        batch_size = cover_image.shape[0]
        current_H, current_W = cover_image.shape[2], cover_image.shape[3]
        
        # Convert message to image (adjust size to match cover image)
        message_image = self.message_converter(message, target_size=current_H)
        
        # Process cover image through ViT
        cover_patches = self.cover_vit.to_patch_embedding(cover_image)
        cover_patches += self.cover_vit.pos_embedding[:, 1:(cover_patches.shape[1] + 1)]
        cover_features = self.cover_vit.transformer(cover_patches)
        cover_global = cover_features.mean(dim=1)  # Global pooling
        
        # Process message image through ViT
        message_patches = self.message_vit.to_patch_embedding(message_image)
        message_patches += self.message_vit.pos_embedding[:, 1:(message_patches.shape[1] + 1)]
        message_features = self.message_vit.transformer(message_patches)
        message_global = message_features.mean(dim=1)  # Global pooling
        
        # Fuse features
        combined_features = torch.cat([cover_global, message_global], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Reshape to spatial format (adjust to current image size)
        spatial_features = fused_features.view(batch_size, self.conv_channels, current_H // 4, current_W // 4)
        spatial_features = F.interpolate(spatial_features, size=(current_H, current_W), 
                                       mode='bilinear', align_corners=True)
        
        # Combine with cover image
        combined_input = torch.cat([cover_image, spatial_features], dim=1)
        
        # Process through convolutional layers
        conv_output = self.conv_layers(combined_input)
        
        # Generate watermark
        watermark = self.watermark_generator(conv_output)
        
        # Adaptive blending with REALISTIC strength
        watermarked_image = cover_image + self.blend_weight * watermark
        
        # Apply scale adaptation for robustness with MODERATE strength
        watermarked_image = watermarked_image + 0.08 * self.scale_adapter(watermarked_image)  # Increased from 0.05
        
        # Ensure valid range
        watermarked_image = torch.clamp(watermarked_image, -1, 1)
        
        return watermarked_image
    
    def forward_multi_scale(self, cover_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Multi-scale training for robustness."""
        # Random scale during training
        if self.training:
            # Use scales that are compatible with patch size (16)
            scale_options = [64, 80, 96, 112, 128, 144, 160, 176, 192]  # All divisible by 16
            scaled_size = scale_options[torch.randint(0, len(scale_options), (1,)).item()]
            
            # Resize cover image
            scaled_cover = F.interpolate(cover_image, size=(scaled_size, scaled_size), 
                                       mode='bilinear', align_corners=True)
            
            # Get watermarked at this scale
            watermarked_scaled = self.forward_standard(scaled_cover, message)
            
            # Resize back
            watermarked_image = F.interpolate(watermarked_scaled, size=(self.H, self.W),
                                            mode='bilinear', align_corners=True)
        else:
            watermarked_image = self.forward_standard(cover_image, message)
            
        return watermarked_image
    
    def forward_robust(self, cover_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Robust training with multiple message scales."""
        # Generate message images at multiple scales
        multi_scale_messages = self.robust_converter(message)
        
        # Use different scales during training
        if self.training:
            scales = list(multi_scale_messages.keys())
            chosen_scale = torch.randint(0, len(scales), (1,)).item()
            scale_key = scales[chosen_scale]
            message_image = multi_scale_messages[scale_key]
            
            # Resize to match cover image if needed
            if message_image.shape[-1] != self.H:
                message_image = F.interpolate(message_image, size=(self.H, self.W),
                                            mode='bilinear', align_corners=True)
        else:
            message_image = multi_scale_messages[f'size_{self.H}']
        
        # Continue with standard processing using the selected message image
        batch_size = cover_image.shape[0]
        
        # Process through ViTs
        cover_patches = self.cover_vit.to_patch_embedding(cover_image)
        cover_patches += self.cover_vit.pos_embedding[:, 1:(cover_patches.shape[1] + 1)]
        cover_features = self.cover_vit.transformer(cover_patches)
        cover_global = cover_features.mean(dim=1)
        
        message_patches = self.message_vit.to_patch_embedding(message_image)
        message_patches += self.message_vit.pos_embedding[:, 1:(message_patches.shape[1] + 1)]
        message_features = self.message_vit.transformer(message_patches)
        message_global = message_features.mean(dim=1)
        
        # Fuse and generate watermark
        combined_features = torch.cat([cover_global, message_global], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        spatial_features = fused_features.view(batch_size, self.conv_channels, self.H // 4, self.W // 4)
        spatial_features = F.interpolate(spatial_features, size=(self.H, self.W), 
                                       mode='bilinear', align_corners=True)
        
        combined_input = torch.cat([cover_image, spatial_features], dim=1)
        conv_output = self.conv_layers(combined_input)
        watermark = self.watermark_generator(conv_output)
        
        watermarked_image = cover_image + self.blend_weight * watermark
        watermarked_image = torch.clamp(watermarked_image, -1, 1)
        
        return watermarked_image
    
    def get_message_image(self, message: torch.Tensor, robust: bool = False) -> torch.Tensor:
        """
        Get the message image for visualization or analysis.
        
        Args:
            message: [batch_size, message_length] - Secret message
            robust: Whether to use robust converter
        Returns:
            message_image: [batch_size, 3, H, W] - Message as image
        """
        if robust:
            multi_scale = self.robust_converter(message)
            return multi_scale[f'size_{self.H}']
        else:
            return self.message_converter(message, target_size=self.H)


class EnhancedViTEncoder(nn.Module):
    """
    Wrapper that provides backward compatibility with your existing system
    while adding message-to-image capabilities.
    """
    def __init__(self, config: HiDDenConfiguration, use_message_images: bool = True):
        super(EnhancedViTEncoder, self).__init__()
        
        self.use_message_images = use_message_images
        
        if use_message_images:
            self.encoder = MessageImageViTEncoder(config)
        else:
            # Fallback to your original encoder
            from model.encoder import Encoder
            self.encoder = Encoder(config)
            
        self.config = config
        
    def forward(self, image: torch.Tensor, message: torch.Tensor, 
                training_mode: str = 'standard') -> torch.Tensor:
        """
        Forward pass with optional message-to-image conversion.
        
        Args:
            image: Cover image
            message: Secret message
            training_mode: Training strategy ('standard', 'multi_scale', 'robust')
        """
        if self.use_message_images:
            return self.encoder(image, message, training_mode)
        else:
            return self.encoder(image, message)
    
    def switch_to_message_images(self):
        """Switch to message-image mode."""
        if not self.use_message_images:
            self.use_message_images = True
            self.encoder = MessageImageViTEncoder(self.config)
            
    def switch_to_standard(self):
        """Switch to standard mode."""
        if self.use_message_images:
            self.use_message_images = False
            from model.encoder import Encoder
            self.encoder = Encoder(self.config)


def test_enhanced_vit_encoder():
    """Test the enhanced ViT encoder with message images."""
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
    cover_image = torch.randn(batch_size, 3, 128, 128).to(device)
    message = torch.randint(0, 2, (batch_size, 30)).float().to(device)
    
    # Test enhanced encoder
    encoder = EnhancedViTEncoder(config, use_message_images=True).to(device)
    
    print("Testing Enhanced ViT Encoder with Message Images")
    print(f"Cover image shape: {cover_image.shape}")
    print(f"Message shape: {message.shape}")
    
    # Test different training modes
    modes = ['standard', 'multi_scale', 'robust']
    
    for mode in modes:
        print(f"\nTesting {mode} mode:")
        watermarked = encoder(cover_image, message, training_mode=mode)
        print(f"Watermarked image shape: {watermarked.shape}")
        print(f"Watermarked range: [{watermarked.min():.3f}, {watermarked.max():.3f}]")
        
        # Calculate PSNR
        mse = F.mse_loss(watermarked, cover_image)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Assuming range [-1, 1]
        print(f"PSNR: {psnr.item():.2f} dB")
    
    # Test message image visualization
    message_image = encoder.encoder.get_message_image(message, robust=True)
    print(f"\nMessage image shape: {message_image.shape}")
    print(f"Message image range: [{message_image.min():.3f}, {message_image.max():.3f}]")
    
    print("\nTesting completed successfully!")
    return encoder


if __name__ == "__main__":
    encoder = test_enhanced_vit_encoder()

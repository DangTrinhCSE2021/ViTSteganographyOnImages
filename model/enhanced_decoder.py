"""
Enhanced Decoder for Message-Image Based ViT Steganography
Handles extraction of messages from images that were embedded using message-to-image conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from vit_pytorch import ViT


class MessageImageDecoder(nn.Module):
    """
    Decoder specifically designed for message-image based embeddings.
    Uses ViT to extract both message and recover original image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(MessageImageDecoder, self).__init__()
        
        self.H = config.H
        self.W = config.W
        self.message_length = config.message_length
        self.decoder_channels = config.decoder_channels
        
        # ViT for feature extraction from watermarked images
        self.feature_vit = ViT(
            image_size=(config.H, config.W),
            patch_size=16,
            num_classes=0,  # No classification
            dim=768,
            depth=8,        # Deeper for better feature extraction
            heads=12,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # Message extraction branch
        self.message_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.message_length * 2),  # Extra bits for error correction
            nn.ReLU(),
            nn.Linear(self.message_length * 2, self.message_length)
            # No final activation - will use BCE with logits loss
        )
        
        # Image recovery branch
        self.image_recovery_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.decoder_channels * (config.H // 8) * (config.W // 8))
        )
        
        # Convolutional layers for image reconstruction
        self.recovery_convs = nn.Sequential(
            # Start from H//8 x W//8
            nn.ConvTranspose2d(config.decoder_channels, config.decoder_channels, 4, 2, 1),  # H//4 x W//4
            nn.BatchNorm2d(config.decoder_channels),
            nn.ReLU(),
            
            nn.ConvTranspose2d(config.decoder_channels, config.decoder_channels//2, 4, 2, 1),  # H//2 x W//2
            nn.BatchNorm2d(config.decoder_channels//2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(config.decoder_channels//2, config.decoder_channels//4, 4, 2, 1),  # H x W
            nn.BatchNorm2d(config.decoder_channels//4),
            nn.ReLU(),
            
            nn.Conv2d(config.decoder_channels//4, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Residual refinement network
        self.residual_refiner = nn.Sequential(
            nn.Conv2d(6, 32, 7, padding=3),  # 6 = watermarked + recovered
            nn.ReLU(),
            nn.Conv2d(32, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1),
            nn.Tanh()
        )
        
        # Attention mechanism for robust message extraction
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # Scale-aware processing
        self.scale_processor = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Identity()  # No activation for residual connection
        )
        
    def forward(self, watermarked_image: torch.Tensor) -> tuple:
        """
        Extract message and recover original image.
        
        Args:
            watermarked_image: [batch_size, 3, H, W] - Watermarked image
        Returns:
            tuple: (decoded_message, recovered_image)
                decoded_message: [batch_size, message_length] - Extracted message logits
                recovered_image: [batch_size, 3, H, W] - Recovered original image
        """
        batch_size = watermarked_image.shape[0]
        
        # Apply scale preprocessing
        scale_processed = watermarked_image + 0.1 * self.scale_processor(watermarked_image)
        
        # Extract features using ViT
        patches = self.feature_vit.to_patch_embedding(scale_processed)
        patches += self.feature_vit.pos_embedding[:, 1:(patches.shape[1] + 1)]
        patches = self.feature_vit.dropout(patches)
        
        # Process through transformer
        vit_features = self.feature_vit.transformer(patches)  # [batch, num_patches, 768]
        
        # Apply attention pooling for robust feature aggregation
        attended_features, _ = self.attention_pooling(vit_features, vit_features, vit_features)
        
        # Global feature for message extraction
        global_feature = attended_features.mean(dim=1)  # [batch, 768]
        
        # Extract message
        decoded_message = self.message_extractor(global_feature)  # [batch, message_length]
        
        # Recover image
        recovery_features = self.image_recovery_proj(global_feature)
        recovery_features = recovery_features.view(batch_size, self.decoder_channels, 
                                                 self.H // 8, self.W // 8)
        
        # Upsample to full resolution
        recovered_base = self.recovery_convs(recovery_features)
        
        # Residual refinement
        combined = torch.cat([watermarked_image, recovered_base], dim=1)
        residual = self.residual_refiner(combined)
        recovered_image = recovered_base + 0.1 * residual
        
        # Ensure proper range
        recovered_image = torch.clamp(recovered_image, -1, 1)
        
        return decoded_message, recovered_image
    
    def extract_message_only(self, watermarked_image: torch.Tensor) -> torch.Tensor:
        """
        Extract only the message (faster inference).
        
        Args:
            watermarked_image: [batch_size, 3, H, W] - Watermarked image
        Returns:
            decoded_message: [batch_size, message_length] - Extracted message logits
        """
        # Scale preprocessing
        scale_processed = watermarked_image + 0.1 * self.scale_processor(watermarked_image)
        
        # ViT processing
        patches = self.feature_vit.to_patch_embedding(scale_processed)
        patches += self.feature_vit.pos_embedding[:, 1:(patches.shape[1] + 1)]
        vit_features = self.feature_vit.transformer(patches)
        
        # Attention pooling and message extraction
        attended_features, _ = self.attention_pooling(vit_features, vit_features, vit_features)
        global_feature = attended_features.mean(dim=1)
        decoded_message = self.message_extractor(global_feature)
        
        return decoded_message


class RobustMessageDecoder(nn.Module):
    """
    Enhanced decoder with multi-scale robustness and error correction.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(RobustMessageDecoder, self).__init__()
        
        self.base_decoder = MessageImageDecoder(config)
        self.message_length = config.message_length
        
        # Multi-scale processing
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1),
                nn.Tanh()
            ) for _ in range(3)  # For different scales
        ])
        
        # Error correction decoder
        self.error_corrector = nn.Sequential(
            nn.Linear(config.message_length, config.message_length * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.message_length * 2, config.message_length),
            nn.Sigmoid()
        )
        
        # Consensus mechanism for robust extraction
        self.consensus_net = nn.Sequential(
            nn.Linear(config.message_length * 3, config.message_length * 2),
            nn.ReLU(),
            nn.Linear(config.message_length * 2, config.message_length)
        )
        
    def forward(self, watermarked_image: torch.Tensor, robust_mode: bool = True) -> tuple:
        """
        Robust message extraction and image recovery.
        
        Args:
            watermarked_image: [batch_size, 3, H, W] - Watermarked image
            robust_mode: Whether to use multi-scale robust extraction
        Returns:
            tuple: (decoded_message, recovered_image)
        """
        if not robust_mode:
            return self.base_decoder(watermarked_image)
        
        # Multi-scale extraction
        messages = []
        scales = [0.8, 1.0, 1.2]
        
        for i, scale in enumerate(scales):
            if scale != 1.0:
                # Resize image
                new_size = int(watermarked_image.shape[-1] * scale)
                scaled_image = F.interpolate(watermarked_image, size=(new_size, new_size),
                                           mode='bilinear', align_corners=True)
                
                # Apply scale-specific preprocessing
                processed_image = scaled_image + 0.05 * self.scale_branches[i](scaled_image)
                
                # Resize back to original size
                processed_image = F.interpolate(processed_image, 
                                              size=(watermarked_image.shape[-2], watermarked_image.shape[-1]),
                                              mode='bilinear', align_corners=True)
            else:
                processed_image = watermarked_image + 0.05 * self.scale_branches[i](watermarked_image)
            
            # Extract message
            message = self.base_decoder.extract_message_only(processed_image)
            messages.append(message)
        
        # Consensus-based message combination
        combined_messages = torch.cat(messages, dim=1)
        final_message = self.consensus_net(combined_messages)
        
        # Apply error correction
        corrected_message = self.error_corrector(torch.sigmoid(final_message))
        
        # Get recovered image from base scale
        _, recovered_image = self.base_decoder(watermarked_image)
        
        return corrected_message, recovered_image


class EnhancedMessageDecoder(nn.Module):
    """
    Wrapper that provides backward compatibility while adding robustness features.
    """
    def __init__(self, config: HiDDenConfiguration, use_robust_extraction: bool = True):
        super(EnhancedMessageDecoder, self).__init__()
        
        self.use_robust_extraction = use_robust_extraction
        
        if use_robust_extraction:
            self.decoder = RobustMessageDecoder(config)
        else:
            self.decoder = MessageImageDecoder(config)
            
        self.config = config
        
    def forward(self, watermarked_image: torch.Tensor) -> tuple:
        """
        Extract message and recover image.
        
        Args:
            watermarked_image: [batch_size, 3, H, W] - Watermarked image
        Returns:
            tuple: (decoded_message, recovered_image)
        """
        if self.use_robust_extraction:
            return self.decoder(watermarked_image, robust_mode=True)
        else:
            return self.decoder(watermarked_image)
    
    def extract_message_only(self, watermarked_image: torch.Tensor) -> torch.Tensor:
        """Extract only the message for faster inference."""
        if hasattr(self.decoder, 'base_decoder'):
            return self.decoder.base_decoder.extract_message_only(watermarked_image)
        else:
            return self.decoder.extract_message_only(watermarked_image)


def test_enhanced_decoder():
    """Test the enhanced decoder with message-image embeddings."""
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
    watermarked_image = torch.randn(batch_size, 3, 128, 128).to(device)
    original_message = torch.randint(0, 2, (batch_size, 30)).float().to(device)
    
    print("Testing Enhanced Message Decoder")
    print(f"Watermarked image shape: {watermarked_image.shape}")
    print(f"Original message shape: {original_message.shape}")
    
    # Test standard decoder
    decoder = EnhancedMessageDecoder(config, use_robust_extraction=False).to(device)
    decoded_message, recovered_image = decoder(watermarked_image)
    
    print(f"\nStandard Decoder Results:")
    print(f"Decoded message shape: {decoded_message.shape}")
    print(f"Recovered image shape: {recovered_image.shape}")
    print(f"Recovered image range: [{recovered_image.min():.3f}, {recovered_image.max():.3f}]")
    
    # Test robust decoder
    robust_decoder = EnhancedMessageDecoder(config, use_robust_extraction=True).to(device)
    robust_decoded, robust_recovered = robust_decoder(watermarked_image)
    
    print(f"\nRobust Decoder Results:")
    print(f"Robust decoded message shape: {robust_decoded.shape}")
    print(f"Robust recovered image shape: {robust_recovered.shape}")
    
    # Test message-only extraction
    message_only = robust_decoder.extract_message_only(watermarked_image)
    print(f"Message-only extraction shape: {message_only.shape}")
    
    # Test with different image sizes (robustness test)
    resized_image = F.interpolate(watermarked_image, size=(96, 96), mode='bilinear', align_corners=True)
    resized_image = F.interpolate(resized_image, size=(128, 128), mode='bilinear', align_corners=True)
    
    resize_decoded, resize_recovered = robust_decoder(resized_image)
    print(f"\nResize robustness test:")
    print(f"Resize decoded message shape: {resize_decoded.shape}")
    
    # Calculate bit accuracy (if we had ground truth)
    # bit_accuracy = ((torch.sigmoid(decoded_message) > 0.5).float() == original_message).float().mean()
    # print(f"Bit accuracy: {bit_accuracy.item():.3f}")
    
    print("\nDecoder testing completed successfully!")
    return decoder, robust_decoder


if __name__ == "__main__":
    decoder, robust_decoder = test_enhanced_decoder()

"""
Integration guide for adding latent space steganography to your existing ViT project.
This shows how to modify your existing code to support latent space embedding.
"""

import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.latent_space import LatentSpaceEncoder, LatentSpaceDecoder, LatentSpaceLoss
from options import HiDDenConfiguration


class HybridEncoder(nn.Module):
    """
    Hybrid encoder that can switch between your original encoder and latent space encoder.
    """
    def __init__(self, config: HiDDenConfiguration, use_latent_space=False):
        super(HybridEncoder, self).__init__()
        self.use_latent_space = use_latent_space
        
        if use_latent_space:
            self.encoder = LatentSpaceEncoder(config)
        else:
            self.encoder = Encoder(config)
    
    def forward(self, image, message):
        return self.encoder(image, message)


class HybridDecoder(nn.Module):
    """
    Hybrid decoder that can switch between your original decoder and latent space decoder.
    """
    def __init__(self, config: HiDDenConfiguration, use_latent_space=False):
        super(HybridDecoder, self).__init__()
        self.use_latent_space = use_latent_space
        
        if use_latent_space:
            self.decoder = LatentSpaceDecoder(config)
        else:
            self.decoder = Decoder(config)
    
    def forward(self, watermarked_image):
        if self.use_latent_space:
            return self.decoder(watermarked_image)
        else:
            # Your original decoder returns (decoded_message, None)
            decoded_message = self.decoder(watermarked_image)
            return decoded_message, None


class EnhancedHidden(nn.Module):
    """
    Enhanced version of your Hidden class with latent space support.
    Drop-in replacement for your existing Hidden class.
    """
    def __init__(self, config: HiDDenConfiguration, device, noiser, tb_logger, use_latent_space=False):
        super(EnhancedHidden, self).__init__()
        
        self.config = config
        self.device = device
        self.use_latent_space = use_latent_space
        
        # Create hybrid encoder and decoder
        self.encoder = HybridEncoder(config, use_latent_space).to(device)
        self.decoder = HybridDecoder(config, use_latent_space).to(device)
        
        # Your existing discriminator (unchanged)
        if config.use_discriminator:
            from model.discriminator import Discriminator
            self.discriminator = Discriminator(config).to(device)
        else:
            self.discriminator = None
        
        # Enhanced loss function
        if use_latent_space:
            self.criterion = LatentSpaceLoss(config)
        else:
            # Your existing loss functions
            self.mse_loss = nn.MSELoss()
            
        # Common loss functions (always available)
        self.mse_loss = nn.MSELoss()  # Always available for recovery loss
        self.bce_loss = nn.BCELoss()  # For discriminator
        self.bce_logits_loss = nn.BCEWithLogitsLoss()  # For other uses
        
        # VGG loss (if used)
        self.vgg_loss = None
        if config.use_vgg:
            try:
                from vgg_loss import VGGLoss
                self.vgg_loss = VGGLoss(3, 1, False)
                self.vgg_loss.to(device)
            except ImportError:
                print("Warning: VGGLoss not available, skipping")
        
        self.noiser = noiser
        self.tb_logger = tb_logger
    
    def train_on_batch(self, batch):
        """
        Enhanced training step that works with both original and latent space approaches.
        """
        images, messages = batch
        batch_size = images.shape[0]
        
        # Encode (embed message)
        watermarked_images = self.encoder(images, messages)
        
        # Add noise (your existing noiser)
        if self.noiser:
            # Pass both watermarked and cover images for attacks like dropout
            noised_images = self.noiser([watermarked_images, images])
        else:
            noised_images = [watermarked_images]
        
        # Decode (extract message)
        decoded_messages, recovered_images = self.decoder(noised_images[0])
        
        # Calculate losses
        if self.use_latent_space:
            losses = self.criterion(
                images, watermarked_images, messages, 
                decoded_messages, recovered_images
            )
            
            # Add recovery quality loss (NEW)
            if recovered_images is not None:
                recovery_mse = self.mse_loss(recovered_images, images)
                recovery_l1 = torch.mean(torch.abs(recovered_images - images))
                losses['recovery_mse'] = recovery_mse
                losses['recovery_l1'] = recovery_l1
                # Add to total loss with significant weight
                losses['total_loss'] += 2.0 * recovery_mse + 0.5 * recovery_l1
            
            # Add VGG loss for both watermarked and recovered images
            if self.config.use_vgg and self.vgg_loss is not None:
                vgg_watermark = self.mse_loss(self.vgg_loss(watermarked_images), self.vgg_loss(images))
                losses['vgg_watermark'] = vgg_watermark
                losses['total_loss'] += 0.1 * vgg_watermark
                
                # VGG loss for recovery quality (NEW)
                if recovered_images is not None:
                    vgg_recovery = self.mse_loss(self.vgg_loss(recovered_images), self.vgg_loss(images))
                    losses['vgg_recovery'] = vgg_recovery
                    losses['total_loss'] += 0.2 * vgg_recovery  # Higher weight for recovery
            
            # Add discriminator loss if enabled
            if self.discriminator is not None:
                d_loss = self.calculate_discriminator_loss(images, watermarked_images)
                losses['d_loss'] = d_loss
                losses['total_loss'] += self.config.adversarial_loss * d_loss
        
        else:
            # Your original loss calculation
            losses = self.calculate_original_losses(
                images, watermarked_images, messages, decoded_messages
            )
        
        return losses, (watermarked_images, noised_images[0], decoded_messages, recovered_images)
    
    def calculate_original_losses(self, images, watermarked_images, messages, decoded_messages):
        """Your existing loss calculation with improved image quality preservation."""
        # Main image reconstruction loss
        encoder_mse = self.mse_loss(watermarked_images, images)
        
        # Handle decoder output format - might be tuple (decoded_message, None) or just decoded_message
        if isinstance(decoded_messages, tuple):
            decoded_messages = decoded_messages[0]
        
        # Use BCEWithLogitsLoss for message reconstruction
        decoder_mse = self.bce_logits_loss(decoded_messages, messages)
        
        # Emphasize image quality preservation (reduce weights to match successful runs)
        total_loss = (self.config.encoder_loss * encoder_mse + 
                     self.config.decoder_loss * decoder_mse)
        
        losses = {
            'total_loss': total_loss,
            'encoder_mse': encoder_mse,
            'decoder_mse': decoder_mse
        }
        
        # Add perceptual loss if VGG is available (helps with image quality)
        if self.config.use_vgg and self.vgg_loss is not None:
            vgg_loss = self.vgg_loss(watermarked_images, images)
            losses['vgg_loss'] = vgg_loss
            # Use smaller VGG loss weight
            losses['total_loss'] += 0.01 * vgg_loss  # Much smaller weight
        
        # Add L1 loss for additional image quality preservation (with smaller weight)
        l1_loss = torch.mean(torch.abs(watermarked_images - images))
        losses['l1_loss'] = l1_loss
        losses['total_loss'] += 0.1 * l1_loss  # Much smaller weight
        
        if self.discriminator is not None:
            d_loss = self.calculate_discriminator_loss(images, watermarked_images)
            losses['d_loss'] = d_loss
            losses['total_loss'] += self.config.adversarial_loss * d_loss
        
        return losses
    
    def calculate_discriminator_loss(self, real_images, fake_images):
        """Calculate discriminator loss (unchanged from your implementation)."""
        if self.discriminator is None:
            return torch.tensor(0.0)
        
        real_pred = self.discriminator(real_images)
        fake_pred = self.discriminator(fake_images.detach())
        
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
        
        return (real_loss + fake_loss) / 2


def create_enhanced_model(config: HiDDenConfiguration, device, noiser, tb_logger, use_latent_space=False):
    """
    Factory function to create enhanced model with optional latent space support.
    
    Args:
        config: Your existing configuration
        device: CUDA device
        noiser: Your existing noiser
        tb_logger: TensorBoard logger
        use_latent_space: Whether to use latent space embedding
    
    Returns:
        Enhanced model that can use either original or latent space approach
    """
    return EnhancedHidden(config, device, noiser, tb_logger, use_latent_space)


# Usage examples:

def example_usage():
    """
    Examples of how to use the latent space implementation.
    """
    
    # Your existing configuration
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=30,
        encoder_channels=32,
        encoder_blocks=4,
        decoder_channels=32,
        decoder_blocks=7,
        use_discriminator=True,
        use_vgg=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example 1: Use your original model (no changes)
    print("=== Original Model ===")
    original_model = create_enhanced_model(config, device, None, None, use_latent_space=False)
    
    # Example 2: Use latent space model
    print("=== Latent Space Model ===")
    latent_model = create_enhanced_model(config, device, None, None, use_latent_space=True)
    
    # Example 3: Compare both approaches
    print("=== Comparison ===")
    batch_size = 4
    images = torch.randn(batch_size, 3, 128, 128).to(device)
    messages = torch.randint(0, 2, (batch_size, 30)).float().to(device)
    
    # Original approach
    original_model.eval()
    with torch.no_grad():
        orig_losses, orig_outputs = original_model.train_on_batch([images, messages])
        print(f"Original - Total Loss: {orig_losses['total_loss']:.4f}")
    
    # Latent space approach
    latent_model.eval()
    with torch.no_grad():
        latent_losses, latent_outputs = latent_model.train_on_batch([images, messages])
        print(f"Latent Space - Total Loss: {latent_losses['total_loss']:.4f}")
    
    # Compare outputs
    orig_watermarked, _, orig_decoded, _ = orig_outputs
    latent_watermarked, _, latent_decoded, latent_recovered = latent_outputs
    
    print(f"Original watermarked range: [{orig_watermarked.min():.3f}, {orig_watermarked.max():.3f}]")
    print(f"Latent watermarked range: [{latent_watermarked.min():.3f}, {latent_watermarked.max():.3f}]")
    
    if latent_recovered is not None:
        print(f"Latent recovered available: {latent_recovered.shape}")


if __name__ == "__main__":
    example_usage()

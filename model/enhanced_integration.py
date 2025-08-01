"""
Integration module for Message-to-Image Enhanced ViT Steganography
Combines all enhanced components and provides easy integration with existing training.
"""

import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.enhanced_vit_encoder import MessageImageViTEncoder
from model.enhanced_decoder import RobustMessageDecoder
from model.enhanced_discriminator import HybridDiscriminator
from noise_layers.enhanced_noiser import create_enhanced_noiser, create_clean_enhanced_noiser
from model.message_to_image import MessageToImageConverter, ResizeRobustMessageImage


class MessageImageSteganographySystem(nn.Module):
    """
    Complete steganography system with message-to-image conversion and enhanced robustness.
    """
    def __init__(self, config: HiDDenConfiguration, device: torch.device, 
                 use_robust_training: bool = True):
        super(MessageImageSteganographySystem, self).__init__()
        
        self.config = config
        self.device = device
        self.use_robust_training = use_robust_training
        
        # Enhanced encoder with message-to-image conversion
        self.encoder = MessageImageViTEncoder(config).to(device)
        
        # Enhanced decoder with robustness features
        self.decoder = RobustMessageDecoder(config).to(device)
        
        # Enhanced discriminator
        if config.use_discriminator:
            self.discriminator = HybridDiscriminator(config).to(device)
        else:
            self.discriminator = None
        
        # Enhanced noiser
        if use_robust_training:
            self.noiser = create_enhanced_noiser(device)
        else:
            self.noiser = create_clean_enhanced_noiser(device)
            
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # VGG loss (if available)
        self.vgg_loss = None
        if config.use_vgg:
            try:
                from vgg_loss import VGGLoss
                self.vgg_loss = VGGLoss(3, 1, False).to(device)
            except ImportError:
                print("Warning: VGGLoss not available")
                
    def forward(self, images: torch.Tensor, messages: torch.Tensor, 
                training_mode: str = 'standard') -> tuple:
        """
        Complete forward pass through the steganography system.
        
        Args:
            images: [batch_size, 3, H, W] - Cover images
            messages: [batch_size, message_length] - Secret messages
            training_mode: 'standard', 'multi_scale', or 'robust'
        Returns:
            tuple: (losses, outputs)
        """
        # Encode (embed message in image)
        watermarked_images = self.encoder(images, messages, training_mode)
        
        # Apply noise/attacks
        if self.noiser:
            noised_data = self.noiser([watermarked_images, images])
            noised_images = noised_data[0]
        else:
            noised_images = watermarked_images
            
        # Decode (extract message and recover image)
        decoded_messages, recovered_images = self.decoder(noised_images)
        
        # Calculate losses
        losses = self.calculate_losses(images, watermarked_images, messages, 
                                     decoded_messages, recovered_images, noised_images)
        
        outputs = (watermarked_images, noised_images, decoded_messages, recovered_images)
        
        return losses, outputs
    
    def calculate_losses(self, original_images: torch.Tensor, watermarked_images: torch.Tensor,
                        messages: torch.Tensor, decoded_messages: torch.Tensor,
                        recovered_images: torch.Tensor, noised_images: torch.Tensor) -> dict:
        """Calculate comprehensive loss function."""
        losses = {}
        
        # Image quality loss (watermark should be imperceptible)
        image_mse = self.mse_loss(watermarked_images, original_images)
        image_l1 = self.l1_loss(watermarked_images, original_images)
        losses['image_mse'] = image_mse
        losses['image_l1'] = image_l1
        
        # Message reconstruction loss
        message_loss = self.bce_loss(decoded_messages, messages)
        losses['message_loss'] = message_loss
        
        # Recovery quality loss (NEW - for enhanced image recovery)
        if recovered_images is not None:
            recovery_mse = self.mse_loss(recovered_images, original_images)
            recovery_l1 = self.l1_loss(recovered_images, original_images)
            losses['recovery_mse'] = recovery_mse
            losses['recovery_l1'] = recovery_l1
        else:
            losses['recovery_mse'] = torch.tensor(0.0, device=original_images.device)
            losses['recovery_l1'] = torch.tensor(0.0, device=original_images.device)
            
        # VGG perceptual loss
        if self.vgg_loss is not None:
            vgg_watermark = self.mse_loss(self.vgg_loss(watermarked_images), 
                                        self.vgg_loss(original_images))
            losses['vgg_watermark'] = vgg_watermark
            
            if recovered_images is not None:
                vgg_recovery = self.mse_loss(self.vgg_loss(recovered_images),
                                          self.vgg_loss(original_images))
                losses['vgg_recovery'] = vgg_recovery
            else:
                losses['vgg_recovery'] = torch.tensor(0.0, device=original_images.device)
        else:
            losses['vgg_watermark'] = torch.tensor(0.0, device=original_images.device)
            losses['vgg_recovery'] = torch.tensor(0.0, device=original_images.device)
            
        # Adversarial loss (if discriminator is used)
        if self.config.use_discriminator and hasattr(self.discriminator, 'generator_loss'):
            adv_losses = self.discriminator.generator_loss(watermarked_images)
            losses['adversarial'] = adv_losses['total_loss']
        else:
            losses['adversarial'] = torch.tensor(0.0, device=original_images.device)
            
        # Combined total loss with REALISTIC weights for proper steganography (30-45dB PSNR)
        total_loss = (
            1.5 * message_loss +                        # High priority: message accuracy
            0.3 * image_mse +                          # REDUCED: allow visible watermark (was 2.0)
            0.1 * image_l1 +                           # REDUCED: allow reasonable distortion
            0.8 * losses['recovery_mse'] +             # Recovery quality (reduced from 2.0)
            0.3 * losses['recovery_l1'] +              # Recovery detail preservation
            0.05 * losses['vgg_watermark'] +           # Perceptual quality (reduced)
            0.05 * losses['vgg_recovery'] +            # Recovery perceptual quality
            self.config.adversarial_loss * losses['adversarial']  # Adversarial component
        )
        
        losses['total_loss'] = total_loss
        
        return losses
        
    def train_on_batch(self, batch: list, training_mode: str = 'robust') -> tuple:
        """
        Training step with enhanced robustness.
        
        Args:
            batch: [images, messages]
            training_mode: Training strategy
        Returns:
            tuple: (losses, outputs)
        """
        images, messages = batch
        return self.forward(images, messages, training_mode)
        
    def validate_on_batch(self, batch: list) -> tuple:
        """Validation step."""
        with torch.no_grad():
            return self.train_on_batch(batch, training_mode='standard')
            
    def update_training_mode(self, epoch: int, total_epochs: int):
        """Update training strategy based on epoch."""
        # Progressive training strategy
        if epoch < total_epochs * 0.3:
            # Early training: focus on basic embedding
            self.noiser = create_clean_enhanced_noiser(self.device)
            training_mode = 'standard'
        elif epoch < total_epochs * 0.7:
            # Mid training: introduce moderate attacks
            attack_config = {
                'resize_prob': 0.4,
                'quantization_prob': 0.3,
                'spatial_dropout_prob': 0.2,
                'blur_prob': 0.3,
                'color_jitter_prob': 0.3,
                'adversarial_prob': 0.2,
                'sequential_attacks': True,
                'max_attacks': 2
            }
            self.noiser = create_enhanced_noiser(self.device, attack_config)
            training_mode = 'multi_scale'
        else:
            # Late training: full robustness training
            self.noiser = create_enhanced_noiser(self.device)
            training_mode = 'robust'
            
        return training_mode
        
    def get_message_image_for_visualization(self, message: torch.Tensor) -> torch.Tensor:
        """Get message image for visualization/analysis."""
        return self.encoder.get_message_image(message, robust=True)


def create_enhanced_steganography_model(config: HiDDenConfiguration, device: torch.device,
                                      use_robust_training: bool = True) -> MessageImageSteganographySystem:
    """
    Factory function to create the complete enhanced steganography system.
    
    Args:
        config: HiDDen configuration
        device: Target device
        use_robust_training: Whether to use robust training features
    Returns:
        MessageImageSteganographySystem: Complete system
    """
    return MessageImageSteganographySystem(config, device, use_robust_training)


def demo_message_to_image_system():
    """Demonstration of the complete message-to-image steganography system."""
    from options import HiDDenConfiguration
    
    # Create configuration
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=30,
        encoder_channels=64,
        encoder_blocks=4,
        decoder_channels=64,
        decoder_blocks=7,
        use_discriminator=True,
        use_vgg=False,  # Disable for demo
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
    secret_messages = torch.randint(0, 2, (batch_size, 30)).float().to(device)
    
    print("=== Message-to-Image Enhanced ViT Steganography Demo ===")
    print(f"Cover images shape: {cover_images.shape}")
    print(f"Secret messages shape: {secret_messages.shape}")
    print(f"Device: {device}")
    
    # Create enhanced steganography system
    system = create_enhanced_steganography_model(config, device, use_robust_training=True)
    
    # Test message-to-image conversion
    print("\n=== Message-to-Image Conversion ===")
    message_image = system.get_message_image_for_visualization(secret_messages)
    print(f"Message image shape: {message_image.shape}")
    print(f"Message image range: [{message_image.min():.3f}, {message_image.max():.3f}]")
    
    # Test different training modes
    training_modes = ['standard', 'multi_scale', 'robust']
    
    for mode in training_modes:
        print(f"\n=== Testing {mode.upper()} Mode ===")
        
        # Forward pass
        losses, outputs = system.train_on_batch([cover_images, secret_messages], training_mode=mode)
        watermarked, noised, decoded, recovered = outputs
        
        print(f"Outputs shapes:")
        print(f"  Watermarked: {watermarked.shape}")
        print(f"  Noised: {noised.shape}")
        print(f"  Decoded messages: {decoded.shape}")
        print(f"  Recovered images: {recovered.shape}")
        
        print(f"Loss values:")
        for loss_name, loss_value in losses.items():
            print(f"  {loss_name}: {loss_value.item():.6f}")
            
        # Calculate metrics
        with torch.no_grad():
            # PSNR calculation
            mse = torch.mean((watermarked - cover_images) ** 2)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Assuming [-1, 1] range
            
            # Bit accuracy
            decoded_bits = (torch.sigmoid(decoded) > 0.5).float()
            bit_accuracy = (decoded_bits == secret_messages).float().mean()
            
            # Recovery PSNR
            if recovered is not None:
                recovery_mse = torch.mean((recovered - cover_images) ** 2)
                recovery_psnr = 20 * torch.log10(2.0 / torch.sqrt(recovery_mse))
            else:
                recovery_psnr = torch.tensor(0.0)
            
            print(f"Quality metrics:")
            print(f"  Watermark PSNR: {psnr.item():.2f} dB")
            print(f"  Recovery PSNR: {recovery_psnr.item():.2f} dB")
            print(f"  Bit accuracy: {bit_accuracy.item():.3f}")
    
    # Test progressive training strategy
    print(f"\n=== Progressive Training Strategy ===")
    total_epochs = 50
    for epoch in [5, 15, 25, 40]:
        training_mode = system.update_training_mode(epoch, total_epochs)
        print(f"Epoch {epoch}/{total_epochs}: Training mode = {training_mode}")
    
    print("\n=== Demo completed successfully! ===")
    print("Your enhanced ViT steganography system with message-to-image conversion is ready!")
    
    return system


if __name__ == "__main__":
    system = demo_message_to_image_system()

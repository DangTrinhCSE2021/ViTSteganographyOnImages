"""
Message-to-Image Conversion Module for Robust ViT Steganography
Inspired by RoSteALS but adapted for Vision Transformer architecture.

This module converts secret messages into robust image patterns that can
withstand geometric attacks like resizing, rotation, and cropping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class ErrorCorrectionEncoder(nn.Module):
    """
    Reed-Solomon inspired error correction for message robustness.
    Adds redundancy to the original message bits.
    """
    def __init__(self, message_length: int, redundancy_factor: float = 0.5):
        super(ErrorCorrectionEncoder, self).__init__()
        self.message_length = message_length
        self.redundancy_bits = int(message_length * redundancy_factor)
        self.total_bits = message_length + self.redundancy_bits
        
        # SIMPLIFIED error correction encoding for better initial convergence
        self.ecc_encoder = nn.Sequential(
            nn.Linear(message_length, message_length),  # Simplified: same size
            nn.Tanh()  # Changed from ReLU → Sigmoid to Tanh for better gradients
        )
        
    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Args:
            message: [batch_size, message_length] - Original message bits
        Returns:
            encoded_message: [batch_size, total_bits] - Message with error correction
        """
        # SIMPLIFIED: Just pass through the encoder with basic redundancy
        encoded = self.ecc_encoder(message)
        
        # Add simple bit repetition for redundancy (instead of complex parity)
        redundant_bits = message[:, :self.redundancy_bits]  # Simple repetition
        
        # Combine learned encoding with redundancy
        combined = torch.cat([encoded, redundant_bits], dim=1)
        
        return combined


class SpatialMessagePattern(nn.Module):
    """
    Converts encoded message bits into spatially robust 2D patterns.
    Uses multiple encoding strategies for robustness.
    """
    def __init__(self, message_bits: int, pattern_size: int = 64):
        super(SpatialMessagePattern, self).__init__()
        self.message_bits = message_bits
        self.pattern_size = pattern_size
        self.grid_size = int(math.sqrt(message_bits))
        
        # Ensure we have enough spatial locations
        if self.grid_size * self.grid_size < message_bits:
            self.grid_size += 1
            
        # Learnable spatial embedding
        self.spatial_embedder = nn.Sequential(
            nn.Linear(message_bits, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, pattern_size * pattern_size),
            nn.Tanh()
        )
        
        # Frequency domain embedding for resize robustness
        self.freq_embedder = nn.Sequential(
            nn.Linear(message_bits, 128),
            nn.ReLU(),
            nn.Linear(128, (pattern_size // 4) * (pattern_size // 4)),
            nn.Tanh()
        )
        
    def create_spatial_redundancy(self, message: torch.Tensor) -> torch.Tensor:
        """Create spatially redundant pattern"""
        batch_size = message.shape[0]
        
        # Create base spatial pattern
        spatial_pattern = self.spatial_embedder(message)
        spatial_pattern = spatial_pattern.view(batch_size, 1, self.pattern_size, self.pattern_size)
        
        return spatial_pattern
        
    def create_frequency_pattern(self, message: torch.Tensor) -> torch.Tensor:
        """Create frequency domain pattern for resize robustness"""
        batch_size = message.shape[0]
        freq_size = self.pattern_size // 4
        
        # Low-frequency components (robust to resizing)
        freq_pattern = self.freq_embedder(message)
        freq_pattern = freq_pattern.view(batch_size, 1, freq_size, freq_size)
        
        # Upsample to full size using bilinear interpolation (smooth)
        freq_pattern = F.interpolate(freq_pattern, size=(self.pattern_size, self.pattern_size), 
                                   mode='bilinear', align_corners=True)
        
        return freq_pattern
        
    def create_block_pattern(self, message: torch.Tensor) -> torch.Tensor:
        """Create block-based pattern for crop robustness"""
        batch_size = message.shape[0]
        block_size = self.pattern_size // 8
        
        # Repeat message in blocks
        message_blocks = message.unsqueeze(-1).unsqueeze(-1)  # [B, bits, 1, 1]
        message_blocks = message_blocks.expand(-1, -1, block_size, block_size)  # [B, bits, 8, 8]
        
        # Arrange blocks in grid
        blocks_per_row = 8
        pattern = torch.zeros(batch_size, 1, self.pattern_size, self.pattern_size, device=message.device)
        
        for i in range(min(self.message_bits, blocks_per_row * blocks_per_row)):
            row = i // blocks_per_row
            col = i % blocks_per_row
            start_r, end_r = row * block_size, (row + 1) * block_size
            start_c, end_c = col * block_size, (col + 1) * block_size
            
            if end_r <= self.pattern_size and end_c <= self.pattern_size:
                pattern[:, 0, start_r:end_r, start_c:end_c] = message_blocks[:, i]
                
        return pattern
        
    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Convert message to robust spatial pattern.
        
        Args:
            message: [batch_size, message_bits] - Encoded message
        Returns:
            pattern: [batch_size, 3, pattern_size, pattern_size] - RGB message pattern
        """
        # Create different types of patterns
        spatial_pattern = self.create_spatial_redundancy(message)
        freq_pattern = self.create_frequency_pattern(message) 
        block_pattern = self.create_block_pattern(message)
        
        # Combine patterns into RGB channels
        # R: Spatial pattern, G: Frequency pattern, B: Block pattern
        rgb_pattern = torch.cat([
            spatial_pattern,     # Red channel
            freq_pattern,        # Green channel  
            block_pattern        # Blue channel
        ], dim=1)
        
        # Apply smoothing filter for resize robustness
        rgb_pattern = self.apply_smoothing_filter(rgb_pattern)
        
        return rgb_pattern
        
    def apply_smoothing_filter(self, pattern: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing for resize robustness"""
        # Create Gaussian kernel
        kernel_size = 5
        sigma = 1.0
        kernel = self.create_gaussian_kernel(kernel_size, sigma, pattern.device)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        
        # Apply smoothing
        smoothed = F.conv2d(pattern, kernel, padding=kernel_size//2, groups=3)
        
        return smoothed
        
    def create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create Gaussian smoothing kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g.outer(g)
        kernel /= kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)


class MessageToImageConverter(nn.Module):
    """
    Main module to convert secret messages into robust image patterns.
    Combines error correction and spatial encoding for maximum robustness.
    """
    def __init__(self, message_length: int, image_size: int = 128, 
                 pattern_size: int = 64, redundancy_factor: float = 0.5):
        super(MessageToImageConverter, self).__init__()
        
        self.message_length = message_length
        self.image_size = image_size
        self.pattern_size = pattern_size
        
        # Error correction encoder
        self.ecc_encoder = ErrorCorrectionEncoder(message_length, redundancy_factor)
        
        # Spatial pattern generator (simplified total bits calculation)
        total_bits = message_length + self.ecc_encoder.redundancy_bits  # Simplified: base + redundancy
        self.pattern_generator = SpatialMessagePattern(total_bits, pattern_size)
        
        # Adaptive scaling network for different image sizes
        self.size_adapter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=1),
            nn.Tanh()
        )
        
    def forward(self, message: torch.Tensor, target_size: Optional[int] = None) -> torch.Tensor:
        """
        Convert message to robust image pattern.
        
        Args:
            message: [batch_size, message_length] - Secret message bits (0 or 1)
            target_size: Optional target image size (default: self.image_size)
        Returns:
            message_image: [batch_size, 3, target_size, target_size] - Message as image
        """
        if target_size is None:
            target_size = self.image_size
            
        # Step 1: Add error correction
        encoded_message = self.ecc_encoder(message)
        
        # Step 2: Convert to spatial pattern
        pattern = self.pattern_generator(encoded_message)
        
        # Step 3: Resize to target size if needed
        if pattern.shape[-1] != target_size:
            pattern = F.interpolate(pattern, size=(target_size, target_size), 
                                  mode='bilinear', align_corners=True)
        
        # Step 4: Apply adaptive refinement
        message_image = self.size_adapter(pattern)
        
        # Step 5: Normalize to reasonable range for ViT
        message_image = torch.clamp(message_image, -1, 1)
        
        return message_image
        
    def extract_message_from_image(self, message_image: torch.Tensor) -> torch.Tensor:
        """
        Extract message from image pattern (for validation/testing).
        This is a simplified extraction - in practice, the decoder handles this.
        """
        # This is a placeholder for message extraction
        # In your full system, the ViT decoder will handle message extraction
        batch_size = message_image.shape[0]
        
        # Simple extraction by averaging channels and spatial locations
        avg_pattern = torch.mean(message_image, dim=1, keepdim=True)  # Average RGB
        avg_pattern = F.adaptive_avg_pool2d(avg_pattern, (8, 8))  # Spatial average
        
        # Convert back to bits (simplified)
        flattened = avg_pattern.view(batch_size, -1)
        message_bits = torch.sigmoid(flattened[:, :self.message_length])
        
        return message_bits


class ResizeRobustMessageImage(nn.Module):
    """
    Enhanced version with specific resize robustness techniques.
    """
    def __init__(self, message_length: int, base_size: int = 128):
        super(ResizeRobustMessageImage, self).__init__()
        
        self.message_length = message_length
        self.base_size = base_size
        
        # Multi-scale message converter
        self.base_converter = MessageToImageConverter(message_length, base_size)
        
        # Scale-invariant feature network
        self.scale_invariant_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, message: torch.Tensor, test_sizes: list = None) -> dict:
        """
        Generate message images at multiple scales for robustness testing.
        
        Args:
            message: [batch_size, message_length] - Secret message
            test_sizes: List of sizes to test (default: [64, 96, 128, 160, 192])
        Returns:
            dict: Message images at different scales
        """
        if test_sizes is None:
            test_sizes = [64, 96, 128, 160, 192]
            
        results = {}
        
        # Generate base message image
        base_image = self.base_converter(message, self.base_size)
        
        # Apply scale-invariant processing
        enhanced_image = self.scale_invariant_net(base_image)
        
        # Generate at multiple scales
        for size in test_sizes:
            if size == self.base_size:
                results[f'size_{size}'] = enhanced_image
            else:
                # Resize with anti-aliasing for robustness
                resized = F.interpolate(enhanced_image, size=(size, size), 
                                      mode='bilinear', align_corners=True)
                results[f'size_{size}'] = resized
                
        return results


def test_message_to_image_conversion():
    """Test the message-to-image conversion system."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    batch_size = 4
    message_length = 30
    image_size = 128
    
    # Create test message
    message = torch.randint(0, 2, (batch_size, message_length)).float().to(device)
    print(f"Original message shape: {message.shape}")
    
    # Test basic conversion
    converter = MessageToImageConverter(message_length, image_size).to(device)
    message_image = converter(message)
    print(f"Message image shape: {message_image.shape}")
    print(f"Message image range: [{message_image.min():.3f}, {message_image.max():.3f}]")
    
    # Test resize robustness
    robust_converter = ResizeRobustMessageImage(message_length, image_size).to(device)
    multi_scale_images = robust_converter(message)
    
    print("\nMulti-scale message images:")
    for size_key, img in multi_scale_images.items():
        print(f"{size_key}: {img.shape}")
        
    # Test resize consistency
    base_img = multi_scale_images['size_128']
    resized_img = F.interpolate(base_img, size=(96, 96), mode='bilinear', align_corners=True)
    target_img = multi_scale_images['size_96']
    
    mse_error = F.mse_loss(resized_img, target_img)
    print(f"\nResize consistency MSE: {mse_error.item():.6f}")
    
    return converter, robust_converter


if __name__ == "__main__":
    # Run tests
    converter, robust_converter = test_message_to_image_conversion()
    print("Message-to-image conversion test completed successfully!")

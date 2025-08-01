"""
Enhanced Noise Layers for Robust Steganography Testing
Provides comprehensive attack simulation including geometric, compression,
and adversarial attacks specifically designed for message-image robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
import random


class ResizeAttack(nn.Module):
    """
    Enhanced resize attack with various interpolation methods and scales.
    Tests robustness of message-image conversion against geometric distortions.
    """
    def __init__(self, scale_range: tuple = (0.5, 1.5), 
                 interpolation_modes: List[str] = None):
        super(ResizeAttack, self).__init__()
        
        self.scale_min, self.scale_max = scale_range
        self.interpolation_modes = interpolation_modes or ['bilinear', 'nearest', 'bicubic']
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply resize attack with random scale and interpolation.
        
        Args:
            noised_and_cover: [watermarked_image, cover_image]
        Returns:
            List[torch.Tensor]: [attacked_image, cover_image]
        """
        watermarked_image = noised_and_cover[0]
        original_size = watermarked_image.shape[-1]
        
        # Random scale factor
        scale_factor = torch.rand(1).item() * (self.scale_max - self.scale_min) + self.scale_min
        new_size = int(original_size * scale_factor)
        new_size = max(32, min(512, new_size))  # Reasonable bounds
        
        # Random interpolation mode
        interpolation = random.choice(self.interpolation_modes)
        
        # Resize down then up (simulates real-world scenario)
        resized_down = F.interpolate(watermarked_image, size=(new_size, new_size),
                                   mode=interpolation, align_corners=True if interpolation != 'nearest' else None)
        
        resized_back = F.interpolate(resized_down, size=(original_size, original_size),
                                   mode=interpolation, align_corners=True if interpolation != 'nearest' else None)
        
        return [resized_back] + noised_and_cover[1:]


class AdaptiveQuantization(nn.Module):
    """
    Adaptive quantization attack that simulates various bit-depth reductions
    and color space changes.
    """
    def __init__(self, bit_depths: List[int] = None, probability: float = 0.8):
        super(AdaptiveQuantization, self).__init__()
        
        self.bit_depths = bit_depths or [4, 5, 6, 7]  # Bits per channel
        self.probability = probability
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply adaptive quantization."""
        if torch.rand(1).item() > self.probability:
            return noised_and_cover
            
        watermarked_image = noised_and_cover[0]
        
        # Random bit depth
        bit_depth = random.choice(self.bit_depths)
        levels = 2 ** bit_depth
        
        # Quantize
        # Assume input is in [-1, 1] range
        normalized = (watermarked_image + 1) / 2  # Convert to [0, 1]
        quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
        quantized = quantized * 2 - 1  # Convert back to [-1, 1]
        
        return [quantized] + noised_and_cover[1:]


class SpatialDropout(nn.Module):
    """
    Spatial dropout that removes rectangular regions from the image,
    simulating occlusion or cropping attacks.
    """
    def __init__(self, drop_prob: float = 0.3, max_drop_ratio: float = 0.3):
        super(SpatialDropout, self).__init__()
        
        self.drop_prob = drop_prob
        self.max_drop_ratio = max_drop_ratio
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply spatial dropout."""
        watermarked_image = noised_and_cover[0]
        
        if torch.rand(1).item() > self.drop_prob:
            return noised_and_cover
            
        batch_size, channels, height, width = watermarked_image.shape
        
        # Random dropout region
        drop_height = int(height * torch.rand(1).item() * self.max_drop_ratio)
        drop_width = int(width * torch.rand(1).item() * self.max_drop_ratio)
        
        if drop_height == 0 or drop_width == 0:
            return noised_and_cover
        
        # Random position
        start_h = torch.randint(0, height - drop_height + 1, (1,)).item()
        start_w = torch.randint(0, width - drop_width + 1, (1,)).item()
        
        # Create dropout mask
        attacked_image = watermarked_image.clone()
        attacked_image[:, :, start_h:start_h+drop_height, start_w:start_w+drop_width] = 0
        
        return [attacked_image] + noised_and_cover[1:]


class GaussianBlur(nn.Module):
    """
    Gaussian blur attack with varying kernel sizes and standard deviations.
    """
    def __init__(self, kernel_size_range: tuple = (3, 9), 
                 sigma_range: tuple = (0.5, 2.0), probability: float = 0.7):
        super(GaussianBlur, self).__init__()
        
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.probability = probability
        
    def create_gaussian_kernel(self, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create 2D Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel = g.outer(g)
        kernel /= kernel.sum()
        
        return kernel.expand(3, 1, kernel_size, kernel_size)
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply Gaussian blur."""
        if torch.rand(1).item() > self.probability:
            return noised_and_cover
            
        watermarked_image = noised_and_cover[0]
        
        # Random kernel size (must be odd)
        kernel_size = torch.randint(self.kernel_size_range[0], self.kernel_size_range[1] + 1, (1,)).item()
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Random sigma
        sigma = torch.rand(1).item() * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        
        # Create and apply kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma, watermarked_image.device)
        blurred = F.conv2d(watermarked_image, kernel, padding=kernel_size//2, groups=3)
        
        return [blurred] + noised_and_cover[1:]


class ColorJitter(nn.Module):
    """
    Color space jittering including brightness, contrast, saturation changes.
    """
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, 
                 saturation: float = 0.2, probability: float = 0.6):
        super(ColorJitter, self).__init__()
        
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.probability = probability
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply color jittering."""
        if torch.rand(1).item() > self.probability:
            return noised_and_cover
            
        watermarked_image = noised_and_cover[0]
        
        # Brightness adjustment
        if self.brightness > 0:
            brightness_factor = 1 + (torch.rand(1).item() - 0.5) * 2 * self.brightness
            watermarked_image = watermarked_image * brightness_factor
        
        # Contrast adjustment
        if self.contrast > 0:
            contrast_factor = 1 + (torch.rand(1).item() - 0.5) * 2 * self.contrast
            mean = watermarked_image.mean(dim=(2, 3), keepdim=True)
            watermarked_image = (watermarked_image - mean) * contrast_factor + mean
        
        # Saturation adjustment (simplified RGB version)
        if self.saturation > 0:
            saturation_factor = 1 + (torch.rand(1).item() - 0.5) * 2 * self.saturation
            # Convert to grayscale weights
            gray = 0.299 * watermarked_image[:, 0:1] + 0.587 * watermarked_image[:, 1:2] + 0.114 * watermarked_image[:, 2:3]
            gray = gray.expand_as(watermarked_image)
            watermarked_image = gray + (watermarked_image - gray) * saturation_factor
        
        # Clamp to valid range
        watermarked_image = torch.clamp(watermarked_image, -1, 1)
        
        return [watermarked_image] + noised_and_cover[1:]


class AdversarialNoise(nn.Module):
    """
    Adversarial noise generation to test robustness against targeted attacks.
    """
    def __init__(self, epsilon: float = 0.1, steps: int = 5, alpha: float = 0.02):
        super(AdversarialNoise, self).__init__()
        
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply adversarial noise (simplified version)."""
        watermarked_image = noised_and_cover[0]
        
        # Add random adversarial-like noise
        noise = torch.randn_like(watermarked_image) * self.epsilon
        noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        
        adversarial_image = watermarked_image + noise
        adversarial_image = torch.clamp(adversarial_image, -1, 1)
        
        return [adversarial_image] + noised_and_cover[1:]


class EnhancedNoiser(nn.Module):
    """
    Enhanced noise layer combining multiple attack types for comprehensive robustness testing.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super(EnhancedNoiser, self).__init__()
        
        # Default configuration
        self.config = config or {
            'resize_prob': 0.6,
            'quantization_prob': 0.5,
            'spatial_dropout_prob': 0.4,
            'blur_prob': 0.5,
            'color_jitter_prob': 0.4,
            'adversarial_prob': 0.3,
            'sequential_attacks': True,
            'max_attacks': 3
        }
        
        # Initialize attack modules
        self.resize_attack = ResizeAttack(scale_range=(0.6, 1.4))
        self.quantization_attack = AdaptiveQuantization(probability=self.config['quantization_prob'])
        self.spatial_dropout = SpatialDropout(drop_prob=self.config['spatial_dropout_prob'])
        self.gaussian_blur = GaussianBlur(probability=self.config['blur_prob'])
        self.color_jitter = ColorJitter(probability=self.config['color_jitter_prob'])
        self.adversarial_noise = AdversarialNoise()
        
        # Attack registry
        self.attacks = {
            'resize': (self.resize_attack, self.config['resize_prob']),
            'quantization': (self.quantization_attack, self.config['quantization_prob']),
            'spatial_dropout': (self.spatial_dropout, self.config['spatial_dropout_prob']),
            'blur': (self.gaussian_blur, self.config['blur_prob']),
            'color_jitter': (self.color_jitter, self.config['color_jitter_prob']),
            'adversarial': (self.adversarial_noise, self.config['adversarial_prob'])
        }
        
    def forward(self, noised_and_cover: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply random combination of attacks.
        
        Args:
            noised_and_cover: [watermarked_image, cover_image]
        Returns:
            List[torch.Tensor]: [attacked_image, cover_image]
        """
        if not self.training:
            return noised_and_cover  # No attacks during evaluation
            
        current_images = noised_and_cover
        applied_attacks = []
        
        if self.config['sequential_attacks']:
            # Apply multiple attacks sequentially
            available_attacks = list(self.attacks.keys())
            num_attacks = torch.randint(1, min(self.config['max_attacks'] + 1, len(available_attacks) + 1), (1,)).item()
            
            selected_attacks = random.sample(available_attacks, num_attacks)
            
            for attack_name in selected_attacks:
                attack_module, probability = self.attacks[attack_name]
                
                if torch.rand(1).item() < probability:
                    current_images = attack_module(current_images)
                    applied_attacks.append(attack_name)
        else:
            # Apply single random attack
            attack_name = random.choice(list(self.attacks.keys()))
            attack_module, probability = self.attacks[attack_name]
            
            if torch.rand(1).item() < probability:
                current_images = attack_module(current_images)
                applied_attacks.append(attack_name)
        
        # Store applied attacks for debugging/analysis
        if hasattr(self, '_applied_attacks'):
            self._applied_attacks.append(applied_attacks)
        
        return current_images
    
    def set_attack_probabilities(self, attack_probs: Dict[str, float]):
        """Update attack probabilities dynamically."""
        for attack_name, prob in attack_probs.items():
            if attack_name in self.attacks:
                attack_module, _ = self.attacks[attack_name]
                self.attacks[attack_name] = (attack_module, prob)
                
    def get_attack_statistics(self) -> Dict[str, int]:
        """Get statistics of applied attacks (for analysis)."""
        if not hasattr(self, '_applied_attacks'):
            return {}
            
        stats = {}
        for attacks_list in self._applied_attacks:
            for attack in attacks_list:
                stats[attack] = stats.get(attack, 0) + 1
                
        return stats
    
    def reset_statistics(self):
        """Reset attack statistics."""
        self._applied_attacks = []


def create_enhanced_noiser(device: torch.device, attack_config: Dict[str, Any] = None) -> EnhancedNoiser:
    """
    Factory function to create enhanced noiser with specific configuration.
    
    Args:
        device: Target device
        attack_config: Configuration for attacks
    Returns:
        EnhancedNoiser: Configured noiser
    """
    # Default robust configuration
    if attack_config is None:
        attack_config = {
            'resize_prob': 0.7,
            'quantization_prob': 0.6,
            'spatial_dropout_prob': 0.5,
            'blur_prob': 0.6,
            'color_jitter_prob': 0.5,
            'adversarial_prob': 0.4,
            'sequential_attacks': True,
            'max_attacks': 2
        }
    
    noiser = EnhancedNoiser(attack_config).to(device)
    return noiser


def create_clean_enhanced_noiser(device: torch.device) -> EnhancedNoiser:
    """Create noiser with minimal attacks for clean training."""
    clean_config = {
        'resize_prob': 0.1,
        'quantization_prob': 0.0,
        'spatial_dropout_prob': 0.0,
        'blur_prob': 0.1,
        'color_jitter_prob': 0.1,
        'adversarial_prob': 0.0,
        'sequential_attacks': False,
        'max_attacks': 1
    }
    
    return EnhancedNoiser(clean_config).to(device)


def test_enhanced_noiser():
    """Test the enhanced noiser with various attacks."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    batch_size = 2
    watermarked_image = torch.randn(batch_size, 3, 128, 128).to(device)
    cover_image = torch.randn(batch_size, 3, 128, 128).to(device)
    
    print("Testing Enhanced Noiser")
    print(f"Input image shape: {watermarked_image.shape}")
    
    # Test individual attacks
    attacks = {
        'Resize': ResizeAttack(),
        'Quantization': AdaptiveQuantization(),
        'Spatial Dropout': SpatialDropout(),
        'Gaussian Blur': GaussianBlur(),
        'Color Jitter': ColorJitter(),
        'Adversarial Noise': AdversarialNoise()
    }
    
    for attack_name, attack_module in attacks.items():
        attack_module = attack_module.to(device)
        attacked = attack_module([watermarked_image, cover_image])
        print(f"{attack_name}: {attacked[0].shape}, range: [{attacked[0].min():.3f}, {attacked[0].max():.3f}]")
    
    # Test enhanced noiser
    print("\n=== Enhanced Noiser ===")
    noiser = create_enhanced_noiser(device)
    noiser.train()  # Enable attacks
    noiser.reset_statistics()
    
    # Test multiple times to see attack combinations
    for i in range(5):
        attacked = noiser([watermarked_image, cover_image])
        print(f"Test {i+1}: output shape {attacked[0].shape}, range: [{attacked[0].min():.3f}, {attacked[0].max():.3f}]")
    
    # Show attack statistics
    stats = noiser.get_attack_statistics()
    print(f"Attack statistics: {stats}")
    
    # Test clean noiser
    print("\n=== Clean Noiser ===")
    clean_noiser = create_clean_enhanced_noiser(device)
    clean_noiser.train()
    clean_attacked = clean_noiser([watermarked_image, cover_image])
    print(f"Clean noiser output: {clean_attacked[0].shape}, range: [{clean_attacked[0].min():.3f}, {clean_attacked[0].max():.3f}]")
    
    print("\nEnhanced noiser testing completed successfully!")
    return noiser, clean_noiser


if __name__ == "__main__":
    noiser, clean_noiser = test_enhanced_noiser()

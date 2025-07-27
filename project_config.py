"""
Project Configuration File
This file contains default configurations and validation functions for the project.
"""

import os
from typing import Dict, Any, List

# Default configurations
DEFAULT_CONFIG = {
    'model': {
        'encoder_mode': 'vit',  # Options: 'vit', 'dino-output', 'dino-attention'
        'image_size': 128,
        'message_length': 30,
        'encoder_blocks': 4,
        'encoder_channels': 32,
        'decoder_blocks': 7,
        'decoder_channels': 64,
        'discriminator_blocks': 3,
        'discriminator_channels': 64,
        'use_discriminator': True,
        'use_vgg': True,
    },
    'training': {
        'batch_size': 32,
        'epochs': 300,
        'learning_rate': 1e-4,
        'decoder_loss': 1.0,
        'encoder_loss': 0.7,
        'adversarial_loss': 1e-3,
        'image_recovery_loss': 1.0,
    },
    'paths': {
        'data_dir': 'data',
        'runs_folder': 'runs',
        'experiments_folder': 'experiments',
    }
}

# Valid encoder modes
VALID_ENCODER_MODES = ['vit', 'dino-output', 'dino-attention']

# Required dependencies
REQUIRED_PACKAGES = [
    'torch', 'torchvision', 'transformers', 'pillow', 
    'numpy', 'scikit-image', 'plotly', 'vit-pytorch',
    'tensorboardX', 'opencv-python'
]


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate encoder mode
    encoder_mode = config.get('model', {}).get('encoder_mode')
    if encoder_mode not in VALID_ENCODER_MODES:
        errors.append(f"Invalid encoder_mode: {encoder_mode}. Must be one of {VALID_ENCODER_MODES}")
    
    # Validate image size
    image_size = config.get('model', {}).get('image_size')
    if image_size and (image_size < 32 or image_size > 512):
        errors.append(f"Image size {image_size} is out of valid range [32, 512]")
    
    # Validate message length
    message_length = config.get('model', {}).get('message_length')
    if message_length and (message_length < 1 or message_length > 100):
        errors.append(f"Message length {message_length} is out of valid range [1, 100]")
    
    # Validate batch size
    batch_size = config.get('training', {}).get('batch_size')
    if batch_size and batch_size < 1:
        errors.append(f"Batch size must be positive, got {batch_size}")
    
    return errors


def validate_data_structure(data_dir: str) -> List[str]:
    """
    Validate that the data directory has the correct structure.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not os.path.exists(data_dir):
        errors.append(f"Data directory does not exist: {data_dir}")
        return errors
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        errors.append(f"Training directory does not exist: {train_dir}")
    
    if not os.path.exists(val_dir):
        errors.append(f"Validation directory does not exist: {val_dir}")
    
    # Check for class subdirectories
    if os.path.exists(train_dir):
        train_subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if not train_subdirs:
            errors.append("Training directory should contain class subdirectories")
    
    if os.path.exists(val_dir):
        val_subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
        if not val_subdirs:
            errors.append("Validation directory should contain class subdirectories")
    
    return errors


def get_project_info() -> Dict[str, Any]:
    """Get comprehensive project information."""
    import torch
    
    info = {
        'project_name': 'ViT Steganography on Images',
        'version': '1.0.0',
        'author': 'DangTrinhCSE2021',
        'description': 'Vision Transformer-based image watermarking scheme',
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'valid_encoder_modes': VALID_ENCODER_MODES,
        'default_config': DEFAULT_CONFIG,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return info


def print_project_info():
    """Print comprehensive project information."""
    info = get_project_info()
    
    print("=" * 60)
    print(f"Project: {info['project_name']}")
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print("=" * 60)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU: {info['gpu_name']}")
        print(f"GPU Memory: {info['gpu_memory_gb']:.2f} GB")
    
    print(f"Valid Encoder Modes: {', '.join(info['valid_encoder_modes'])}")
    print("=" * 60)

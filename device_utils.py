"""
Utility functions for device management and CUDA operations.
"""
import torch
import logging


def get_device_info():
    """Get comprehensive device information."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    info = {
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        info['memory_cached'] = torch.cuda.memory_reserved(0) / 1024**3  # GB
    
    return info


def log_device_info():
    """Log detailed device information."""
    info = get_device_info()
    logging.info(f"Using device: {info['device']}")
    
    if info['cuda_available']:
        logging.info(f"GPU: {info['device_name']}")
        logging.info(f"CUDA Version: {info['cuda_version']}")
        logging.info(f"GPU Memory - Total: {info['memory_total']:.2f} GB, "
                    f"Allocated: {info['memory_allocated']:.2f} GB, "
                    f"Cached: {info['memory_cached']:.2f} GB")
    else:
        logging.info("CUDA not available, using CPU")
    
    return info['device']


def ensure_device_consistency(tensors_or_models, target_device):
    """Ensure all tensors/models are on the same device."""
    for item in tensors_or_models:
        if hasattr(item, 'to'):
            item.to(target_device)
        elif isinstance(item, torch.Tensor):
            item = item.to(target_device)


def get_memory_usage():
    """Get current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB"
    return "CPU mode - no GPU memory usage"

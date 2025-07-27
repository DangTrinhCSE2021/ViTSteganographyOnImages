#!/usr/bin/env python3
"""
Setup and validation script for ViT Steganography project.
Run this script to check dependencies and validate the environment.
"""

import sys
import subprocess
import importlib
from typing import List, Tuple

REQUIRED_PACKAGES = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('transformers', 'transformers'),
    ('PIL', 'pillow'),
    ('numpy', 'numpy'),
    ('skimage', 'scikit-image'),
    ('plotly', 'plotly'),
    ('vit_pytorch', 'vit-pytorch'),
    ('tensorboardX', 'tensorboardX'),
    ('cv2', 'opencv-python'),
]


def check_package(import_name: str, package_name: str) -> Tuple[bool, str]:
    """
    Check if a package is installed and importable.
    
    Args:
        import_name: Name to use for import
        package_name: Name to use for pip install
        
    Returns:
        Tuple of (is_available, version_or_error)
    """
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version}, {device_count} GPU(s), Primary: {device_name}"
        else:
            return False, "CUDA not available"
    except ImportError:
        return False, "PyTorch not installed"


def install_missing_packages(missing_packages: List[str]):
    """Install missing packages using pip."""
    if not missing_packages:
        return
    
    print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
    
    for package in missing_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")


def main():
    """Main setup and validation function."""
    print("=" * 60)
    print("ViT Steganography Project - Environment Check")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version
    print(f"Python Version: {python_version}")
    
    if sys.version_info < (3, 7):
        print("⚠️  Warning: Python 3.7+ is recommended")
    
    print("\nChecking required packages:")
    print("-" * 40)
    
    missing_packages = []
    
    for import_name, package_name in REQUIRED_PACKAGES:
        is_available, info = check_package(import_name, package_name)
        
        if is_available:
            print(f"✓ {package_name}: {info}")
        else:
            print(f"✗ {package_name}: Not installed")
            missing_packages.append(package_name)
    
    print("\nChecking CUDA support:")
    print("-" * 40)
    
    cuda_available, cuda_info = check_cuda()
    if cuda_available:
        print(f"✓ {cuda_info}")
    else:
        print(f"⚠️  {cuda_info}")
    
    # Install missing packages
    if missing_packages:
        print(f"\nFound {len(missing_packages)} missing packages.")
        response = input("Do you want to install them automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            install_missing_packages(missing_packages)
        else:
            print("Please install the missing packages manually:")
            for package in missing_packages:
                print(f"  pip install {package}")
    else:
        print("\n✓ All required packages are installed!")
    
    # Check data directory
    print("\nChecking project structure:")
    print("-" * 40)
    
    import os
    data_dir = "data"
    
    if os.path.exists(data_dir):
        train_exists = os.path.exists(os.path.join(data_dir, "train"))
        val_exists = os.path.exists(os.path.join(data_dir, "val"))
        
        print(f"✓ Data directory exists: {data_dir}")
        print(f"{'✓' if train_exists else '✗'} Training directory: {os.path.join(data_dir, 'train')}")
        print(f"{'✓' if val_exists else '✗'} Validation directory: {os.path.join(data_dir, 'val')}")
        
        if not (train_exists and val_exists):
            print("\n⚠️  Please ensure your data is organized as:")
            print("   data/")
            print("     train/")
            print("       train_class/")
            print("         <image files>")
            print("     val/")
            print("       val_class/")
            print("         <image files>")
    else:
        print(f"✗ Data directory not found: {data_dir}")
        print("Please create the data directory with train/val subdirectories")
    
    print("\n" + "=" * 60)
    print("Environment check complete!")
    
    if not missing_packages and cuda_available:
        print("🎉 Your environment is ready for training!")
    elif not missing_packages:
        print("⚠️  Environment is ready, but CUDA is not available (CPU training only)")
    else:
        print("⚠️  Please resolve the issues above before training")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Quick setup and run script for the enhanced recovery training.
This script sets up and runs the enhanced training with recovery optimization.
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'scikit-image', 'vit-pytorch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'vit-pytorch':
                import vit_pytorch
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available!")
    return True


def check_data_directory():
    """Check if data directory exists."""
    data_dirs = ['./data/train', './data/val']
    
    print("\n📁 Checking data directories...")
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            try:
                # Count images
                from torchvision import datasets
                from torchvision.transforms import ToTensor
                dataset = datasets.ImageFolder(data_dir, transform=ToTensor())
                print(f"  ✅ {data_dir}: {len(dataset)} images")
            except Exception as e:
                print(f"  ⚠️  {data_dir}: exists but has issues - {e}")
        else:
            print(f"  ❌ {data_dir}: not found")
    
    print("Note: If data directories are missing, the script will use dummy data.")


def run_enhanced_training():
    """Run the enhanced training with recovery optimization."""
    print("\n🚀 Starting Enhanced Recovery Training...")
    
    # Basic training command
    cmd = [
        sys.executable, 
        "train_recovery_optimized.py",
        "--use-latent-space",
        "--epochs", "30",
        "--learning-rate", "1e-5",
        "--batch-size", "8",
        "--encoder-loss", "7.5",
        "--decoder-loss", "1.0", 
        "--adversarial-loss", "1e-3",
        "--save-interval", "5"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n📊 Expected outputs:")
    print("  - Enhanced training metrics with recovery PSNR/SSIM")
    print("  - Checkpoints saved in ./latent_runs/recovery_optimized_TIMESTAMP/")
    print("  - CSV with both watermark and recovery quality metrics")
    print("\n⏱️  Training will take approximately 2-4 hours on GPU...")
    
    return cmd


def run_quick_test():
    """Run a quick test with fewer epochs to verify everything works."""
    print("\n🧪 Running Quick Test (5 epochs)...")
    
    cmd = [
        sys.executable,
        "train_recovery_optimized.py", 
        "--use-latent-space",
        "--epochs", "5",
        "--learning-rate", "1e-5",
        "--batch-size", "4",  # Smaller batch for testing
        "--encoder-loss", "7.5",
        "--decoder-loss", "1.0",
        "--adversarial-loss", "1e-3"
    ]
    
    print(f"Quick test command: {' '.join(cmd)}")
    return cmd


def display_commands():
    """Display all available training commands."""
    print("\n📋 Available Training Commands:")
    print("\n1. 🚀 Full Enhanced Training (30 epochs):")
    print("   python train_recovery_optimized.py --use-latent-space --epochs 30")
    
    print("\n2. 🧪 Quick Test (5 epochs):")
    print("   python train_recovery_optimized.py --use-latent-space --epochs 5 --batch-size 4")
    
    print("\n3. 📊 With Custom Settings:")
    print("   python train_recovery_optimized.py \\")
    print("     --use-latent-space \\")
    print("     --epochs 30 \\") 
    print("     --learning-rate 1e-5 \\")
    print("     --batch-size 8 \\")
    print("     --encoder-loss 7.5 \\")
    print("     --decoder-loss 1.0 \\")
    print("     --adversarial-loss 1e-3 \\")
    print("     --data-dir ./data/train")
    
    print("\n4. 🔄 Continue from Original Training:")
    print("   python train_latent_space.py --use-latent-space --epochs 30")
    print("   (Your original script - for comparison)")


def main():
    """Main setup and run function."""
    print("=" * 60)
    print("🎯 Enhanced ViT Steganography Recovery Training Setup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        return
    
    # Check data
    check_data_directory()
    
    # Display options
    display_commands()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED: Start with Quick Test")
    print("=" * 60)
    
    # Ask user what to do
    choice = input("\nWhat would you like to do?\n"
                  "1. Run quick test (5 epochs)\n"
                  "2. Run full training (30 epochs)\n"
                  "3. Just show commands\n"
                  "Enter choice (1-3): ").strip()
    
    if choice == "1":
        cmd = run_quick_test()
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
    
    elif choice == "2":
        cmd = run_enhanced_training()
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
    
    elif choice == "3":
        print("\n✅ Commands displayed above. Copy and run manually.")
    
    else:
        print("❌ Invalid choice. Please run the script again.")
    
    print("\n🔍 After training, check the output directory for:")
    print("  - enhanced_training_metrics.csv (recovery metrics)")
    print("  - best_model.pth (best checkpoint)")
    print("  - good_recovery_epoch_X.pth (good recovery checkpoints)")


if __name__ == "__main__":
    main()

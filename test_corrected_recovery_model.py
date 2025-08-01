"""
CORRECTED Advanced Recovery Model Test Script
This script properly loads the complete trained model and tests it correctly.
The issue was that the previous test script was not loading the trained weights properly.
"""

import torch
import torch.nn as nn
import argparse
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Import the COMPLETE training system (not just decoder)
from train_high_recovery_latent import HighRecoveryLatentSteganographySystem
from options import HiDDenConfiguration

def compute_psnr(img1, img2):
    """Compute PSNR between two image tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0  # Images are in [-1, 1] range
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr.item()

def compute_ssim(img1, img2):
    """Compute SSIM between two image tensors."""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    if len(img1_np.shape) == 4:
        ssim_values = []
        for i in range(img1_np.shape[0]):
            img1_hwc = img1_np[i].transpose(1, 2, 0)
            img2_hwc = img2_np[i].transpose(1, 2, 0)
            img1_norm = (img1_hwc + 1) / 2
            img2_norm = (img2_hwc + 1) / 2
            ssim_val = ssim(img1_norm, img2_norm, multichannel=True, channel_axis=2, data_range=1.0)
            ssim_values.append(ssim_val)
        return sum(ssim_values) / len(ssim_values)
    else:
        img1_norm = (img1_np + 1) / 2
        img2_norm = (img2_np + 1) / 2
        return ssim(img1_norm, img2_norm, multichannel=True, channel_axis=2, data_range=1.0)

def calculate_bit_accuracy(original_message, decoded_message):
    """Calculate bit-level accuracy."""
    original_bits = (original_message > 0.5).float()
    decoded_bits = (torch.sigmoid(decoded_message) > 0.5).float()
    accuracy = (original_bits == decoded_bits).float().mean()
    return accuracy.item()

class CorrectAdvancedRecoveryTester:
    """CORRECTED tester that properly loads the complete trained model."""
    
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model_path = model_path
        
        # Create the COMPLETE configuration (matching training)
        self.config = HiDDenConfiguration(
            H=128, W=128,
            message_length=30,
            encoder_channels=64,
            encoder_blocks=4,
            decoder_channels=64,
            decoder_blocks=7,
            use_discriminator=False,
            use_vgg=False,
            discriminator_blocks=3,
            discriminator_channels=64,
            encoder_loss=1.0,
            decoder_loss=1.0,
            adversarial_loss=0.0
        )
        
        # Load the COMPLETE model system (not just decoder!)
        self.model = self._load_complete_model()
        
    def _load_complete_model(self):
        """Load the complete HighRecoveryLatentSteganographySystem model."""
        print(f"Loading COMPLETE model from: {self.model_path}")
        
        # Create the complete model system
        model = HighRecoveryLatentSteganographySystem(
            self.config, self.device, use_robust_training=True
        )
        
        # Load the checkpoint
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Model loaded successfully from epoch {epoch}")
        else:
            state_dict = checkpoint
            print("Model loaded successfully")
        
        # Load the state dict into the complete model
        model.load_state_dict(state_dict)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}")
        
        return model
    
    def load_and_preprocess_image(self, image_path: str):
        """Load and preprocess image for testing."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to model input size
        image = image.resize((self.config.W, self.config.H), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def test_model_performance(self, image_tensor: torch.Tensor, num_tests: int = 5):
        """Test model performance with multiple random messages."""
        print(f"=== Testing Model Performance ({num_tests} samples) ===")
        
        results = {
            'watermark_psnr': [],
            'watermark_ssim': [],
            'recovery_psnr': [],
            'recovery_ssim': [],
            'bit_accuracy': [],
            'message_loss': []
        }
        
        with torch.no_grad():
            for test_idx in range(num_tests):
                # Generate random message
                message = torch.randint(0, 2, (1, self.config.message_length)).float().to(self.device)
                
                # COMPLETE pipeline test (using the trained model properly)
                # 1. Encode message into image
                watermarked_image = self.model.encoder(image_tensor, message)
                
                # 2. Decode message and recover image
                decoded_message_logits, recovered_image = self.model.decoder(watermarked_image)
                
                # Calculate metrics
                watermark_psnr = compute_psnr(image_tensor, watermarked_image)
                watermark_ssim = compute_ssim(image_tensor, watermarked_image)
                recovery_psnr = compute_psnr(image_tensor, recovered_image)
                recovery_ssim = compute_ssim(image_tensor, recovered_image)
                bit_accuracy = calculate_bit_accuracy(message, decoded_message_logits)
                
                # Message reconstruction loss
                message_loss = nn.BCEWithLogitsLoss()(decoded_message_logits, message).item()
                
                # Store results
                results['watermark_psnr'].append(watermark_psnr)
                results['watermark_ssim'].append(watermark_ssim)
                results['recovery_psnr'].append(recovery_psnr)
                results['recovery_ssim'].append(recovery_ssim)
                results['bit_accuracy'].append(bit_accuracy)
                results['message_loss'].append(message_loss)
                
                # Print individual test results
                print(f"Test {test_idx + 1}/{num_tests}:")
                print(f"  Watermark Quality: PSNR={watermark_psnr:.2f}dB, SSIM={watermark_ssim:.4f}")
                
                # Highlight recovery performance
                recovery_status = "🎯 EXCELLENT!" if recovery_psnr >= 32 else "📈 GOOD" if recovery_psnr >= 28 else "🔧 POOR"
                print(f"  Recovery Quality:  PSNR={recovery_psnr:.2f}dB, SSIM={recovery_ssim:.4f} {recovery_status}")
                print(f"  Message Accuracy:  {bit_accuracy * 100:.1f}%")
                print(f"  Message Loss:      {message_loss:.4f}")
        
        return results
    
    def summarize_results(self, results: dict):
        """Summarize and display test results."""
        print(f"=== Performance Summary ===")
        
        # Calculate means and standard deviations
        watermark_psnr_mean = np.mean(results['watermark_psnr'])
        watermark_psnr_std = np.std(results['watermark_psnr'])
        watermark_ssim_mean = np.mean(results['watermark_ssim'])
        watermark_ssim_std = np.std(results['watermark_ssim'])
        
        recovery_psnr_mean = np.mean(results['recovery_psnr'])
        recovery_psnr_std = np.std(results['recovery_psnr'])
        recovery_ssim_mean = np.mean(results['recovery_ssim'])
        recovery_ssim_std = np.std(results['recovery_ssim'])
        
        bit_accuracy_mean = np.mean(results['bit_accuracy'])
        bit_accuracy_std = np.std(results['bit_accuracy'])
        
        # Display results
        print(f"Watermark Quality:")
        print(f"  PSNR: {watermark_psnr_mean:.2f} ± {watermark_psnr_std:.2f} dB")
        print(f"  SSIM: {watermark_ssim_mean:.4f} ± {watermark_ssim_std:.4f}")
        
        # Highlight recovery performance 
        recovery_status = "🎯 TARGET ACHIEVED!" if recovery_psnr_mean >= 32 else "📈 CLOSE TO TARGET" if recovery_psnr_mean >= 30 else "🔧 NEEDS IMPROVEMENT"
        print(f"Recovery Quality:")
        print(f"  PSNR: {recovery_psnr_mean:.2f} ± {recovery_psnr_std:.2f} dB {recovery_status}")
        print(f"  SSIM: {recovery_ssim_mean:.4f} ± {recovery_ssim_std:.4f}")
        
        print(f"Message Accuracy: {bit_accuracy_mean * 100:.1f}% ± {bit_accuracy_std * 100:.1f}%")
        
        return {
            'watermark_psnr': watermark_psnr_mean,
            'recovery_psnr': recovery_psnr_mean,
            'bit_accuracy': bit_accuracy_mean
        }

def main():
    """Main testing function with CORRECTED model loading."""
    parser = argparse.ArgumentParser(description='CORRECTED Advanced Recovery Model Testing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_image', type=str, required=True, help='Path to test image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_tests', type=int, default=5, help='Number of test samples')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create CORRECTED tester
        tester = CorrectAdvancedRecoveryTester(args.model_path, device)
        print("✅ Model loaded successfully!")
        
        # Load test image
        image_tensor = tester.load_and_preprocess_image(args.test_image)
        print(f"Test image loaded: {args.test_image}")
        
        # Run tests
        results = tester.test_model_performance(image_tensor, args.num_tests)
        
        # Summarize results
        summary = tester.summarize_results(results)
        
        print("🎉 Testing completed successfully!")
        print(f"Model: {args.model_path}")
        print(f"Summary: Watermark PSNR={summary['watermark_psnr']:.1f}dB, Recovery PSNR={summary['recovery_psnr']:.1f}dB, Accuracy={summary['bit_accuracy']:.1%}")
        
        # Check if target achieved
        if summary['recovery_psnr'] >= 32:
            print("🎯 32dB RECOVERY TARGET ACHIEVED!")
        elif summary['recovery_psnr'] >= 28:
            print("📈 Close to target! Continue training for 32dB+")
        else:
            print("🔧 Recovery performance needs improvement")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
Debug script to investigate the massive discrepancy between training metrics (48dB recovery) 
and testing results (11dB recovery). This suggests an architectural mismatch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Import our models
from train_high_recovery_latent import HighRecoveryLatentSteganographySystem, AdvancedRecoveryDecoder
from test_advanced_recovery_model import AdvancedRecoveryDecoder as TestAdvancedRecoveryDecoder
from options import HiDDenConfiguration

def compute_psnr(img1, img2):
    """Compute PSNR between two image tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0  # Assuming images are in [-1, 1] range
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr.item()

def analyze_architectural_differences():
    """Compare training vs testing decoder architectures."""
    print("=== ARCHITECTURAL ANALYSIS ===")
    
    config = HiDDenConfiguration(
        H=128, W=128, message_length=30, 
        encoder_channels=64, encoder_blocks=4,
        decoder_channels=64, decoder_blocks=7,
        use_discriminator=False, use_vgg=False,
        discriminator_blocks=3, discriminator_channels=64,
        encoder_loss=1.0, decoder_loss=1.0, adversarial_loss=0.0
    )
    
    # Training decoder
    train_decoder = AdvancedRecoveryDecoder(config)
    
    # Testing decoder  
    test_decoder = TestAdvancedRecoveryDecoder(config)
    
    # Count parameters
    train_params = sum(p.numel() for p in train_decoder.parameters())
    test_params = sum(p.numel() for p in test_decoder.parameters())
    
    print(f"Training Decoder Parameters: {train_params:,}")
    print(f"Testing Decoder Parameters: {test_params:,}")
    print(f"Parameter Difference: {abs(train_params - test_params):,}")
    
    if train_params != test_params:
        print("❌ ARCHITECTURAL MISMATCH DETECTED!")
        print("This explains the performance discrepancy!")
    else:
        print("✅ Architectures match - issue elsewhere")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 128, 128)
    
    try:
        train_msg, train_recovery = train_decoder(dummy_input)
        print(f"Training Decoder Output Shapes: msg={list(train_msg.shape)}, recovery={list(train_recovery.shape)}")
    except Exception as e:
        print(f"Training Decoder Error: {e}")
    
    try:
        test_msg, test_recovery = test_decoder(dummy_input)
        print(f"Testing Decoder Output Shapes: msg={list(test_msg.shape)}, recovery={list(test_recovery.shape)}")
    except Exception as e:
        print(f"Testing Decoder Error: {e}")

def debug_model_loading():
    """Debug the model loading process."""
    print("\n=== MODEL LOADING DEBUG ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = HiDDenConfiguration(
        H=128, W=128, message_length=30, 
        encoder_channels=64, encoder_blocks=4,
        decoder_channels=64, decoder_blocks=7,
        use_discriminator=False, use_vgg=False,
        discriminator_blocks=3, discriminator_channels=64,
        encoder_loss=1.0, decoder_loss=1.0, adversarial_loss=0.0
    )
    
    model_path = "latent_runs/high_recovery_latent_20250730_203148/high_recovery_latent_epoch_10.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print("✅ Checkpoint loaded successfully")
    
    # Analyze checkpoint structure
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys()))
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Print model layers
    print("\nModel layers in checkpoint:")
    for key in list(state_dict.keys())[:10]:  # First 10 keys
        print(f"  {key}: {state_dict[key].shape}")
    
    # Check if decoder layers match
    decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
    print(f"\nDecoder layers in checkpoint: {len(decoder_keys)}")
    for key in decoder_keys[:5]:
        print(f"  {key}: {state_dict[key].shape}")
    
    return state_dict

def test_training_vs_inference_modes():
    """Test if the issue is related to training vs inference modes."""
    print("\n=== TRAINING VS INFERENCE MODE TEST ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = HiDDenConfiguration(
        H=128, W=128, message_length=30, 
        encoder_channels=64, encoder_blocks=4,
        decoder_channels=64, decoder_blocks=7,
        use_discriminator=False, use_vgg=False,
        discriminator_blocks=3, discriminator_channels=64,
        encoder_loss=1.0, decoder_loss=1.0, adversarial_loss=0.0
    )
    
    # Create model
    model = HighRecoveryLatentSteganographySystem(config, device, use_robust_training=False)
    
    # Test with dummy data
    dummy_image = torch.randn(1, 3, 128, 128).to(device)
    dummy_message = torch.randint(0, 2, (1, 30)).float().to(device)
    
    # Test in training mode
    model.train()
    with torch.no_grad():
        losses, outputs = model.train_on_batch([dummy_image, dummy_message], training_mode='balanced')
        watermarked, noised, decoded_msg, recovered = outputs
        train_recovery_psnr = compute_psnr(dummy_image, recovered)
        print(f"Training Mode Recovery PSNR: {train_recovery_psnr:.2f}dB")
    
    # Test in eval mode
    model.eval()
    with torch.no_grad():
        watermarked = model.encoder(dummy_image, dummy_message)
        decoded_msg, recovered = model.decoder(watermarked)
        eval_recovery_psnr = compute_psnr(dummy_image, recovered)
        print(f"Eval Mode Recovery PSNR: {eval_recovery_psnr:.2f}dB")
    
    print(f"Mode difference: {abs(train_recovery_psnr - eval_recovery_psnr):.2f}dB")

def visualize_training_metrics():
    """Visualize the training metrics to understand the performance."""
    print("\n=== TRAINING METRICS VISUALIZATION ===")
    
    metrics_file = "latent_runs/high_recovery_latent_20250730_203148/high_recovery_training_metrics.csv"
    
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        return
    
    # Read the single line of metrics
    with open(metrics_file, 'r') as f:
        line = f.read().strip()
    
    # Parse the metrics
    values = line.split(',')
    if len(values) >= 11:
        epoch = int(values[0])
        train_loss = float(values[1])
        train_psnr = float(values[2])
        train_ssim = float(values[3])
        train_recovery_psnr = float(values[4])
        train_recovery_ssim = float(values[5])
        eval_loss = float(values[6])
        eval_psnr = float(values[7])
        eval_ssim = float(values[8])
        eval_recovery_psnr = float(values[9])
        eval_recovery_ssim = float(values[10])
        
        print(f"Epoch {epoch} Training Metrics:")
        print(f"  🎯 Train Recovery PSNR: {train_recovery_psnr:.2f}dB")
        print(f"  🎯 Eval Recovery PSNR: {eval_recovery_psnr:.2f}dB")
        print(f"  📊 Train Watermark PSNR: {train_psnr:.2f}dB")
        print(f"  📊 Eval Watermark PSNR: {eval_psnr:.2f}dB")
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Recovery PSNR comparison
        ax1.bar(['Training', 'Evaluation', 'Test Result'], 
                [train_recovery_psnr, eval_recovery_psnr, 11.56], 
                color=['green', 'blue', 'red'])
        ax1.set_ylabel('Recovery PSNR (dB)')
        ax1.set_title('Recovery PSNR: Training vs Testing')
        ax1.axhline(y=32, color='orange', linestyle='--', label='Target (32dB)')
        ax1.legend()
        
        # Watermark PSNR comparison
        ax2.bar(['Training', 'Evaluation'], 
                [train_psnr, eval_psnr], 
                color=['green', 'blue'])
        ax2.set_ylabel('Watermark PSNR (dB)')
        ax2.set_title('Watermark PSNR During Training')
        
        # SSIM comparison
        ax3.bar(['Train Recovery', 'Eval Recovery', 'Train Watermark', 'Eval Watermark'], 
                [train_recovery_ssim, eval_recovery_ssim, train_ssim, eval_ssim], 
                color=['lightgreen', 'lightblue', 'green', 'blue'])
        ax3.set_ylabel('SSIM')
        ax3.set_title('SSIM Values During Training')
        
        # Loss values
        ax4.bar(['Training', 'Evaluation'], 
                [train_loss, eval_loss], 
                color=['green', 'blue'])
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss Values')
        
        plt.tight_layout()
        plt.savefig('training_vs_testing_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualization saved as 'training_vs_testing_analysis.png'")
        plt.show()
        
        # Analysis
        print(f"\n🔍 ANALYSIS:")
        print(f"  Training achieved {train_recovery_psnr:.1f}dB recovery PSNR")
        print(f"  Testing only achieves 11.6dB recovery PSNR")
        print(f"  Discrepancy: {train_recovery_psnr - 11.6:.1f}dB")
        print(f"  This indicates a MAJOR architectural mismatch!")

def check_layer_by_layer_differences():
    """Check layer-by-layer differences between training and testing decoders."""
    print("\n=== LAYER-BY-LAYER ANALYSIS ===")
    
    config = HiDDenConfiguration(
        H=128, W=128, message_length=30, 
        encoder_channels=64, encoder_blocks=4,
        decoder_channels=64, decoder_blocks=7,
        use_discriminator=False, use_vgg=False,
        discriminator_blocks=3, discriminator_channels=64,
        encoder_loss=1.0, decoder_loss=1.0, adversarial_loss=0.0
    )
    
    # Training decoder
    train_decoder = AdvancedRecoveryDecoder(config)
    
    # Testing decoder
    test_decoder = TestAdvancedRecoveryDecoder(config)
    
    print("Training Decoder Modules:")
    for name, module in train_decoder.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            print(f"  {name}: {module}")
    
    print("\nTesting Decoder Modules:")
    for name, module in test_decoder.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            print(f"  {name}: {module}")

def main():
    """Main debugging function to identify the training-testing mismatch."""
    print("🔍 DEBUGGING TRAINING-TESTING PERFORMANCE MISMATCH")
    print("Training shows 48dB recovery, testing shows 11dB recovery!")
    print("=" * 60)
    
    # Run all analysis functions
    analyze_architectural_differences()
    debug_model_loading()
    test_training_vs_inference_modes()
    visualize_training_metrics()
    check_layer_by_layer_differences()
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION: The massive performance difference suggests:")
    print("1. Architectural mismatch between training and testing decoders")
    print("2. Model loading issues")
    print("3. Different evaluation procedures")
    print("4. Need to fix the testing architecture to match training!")

if __name__ == "__main__":
    main()

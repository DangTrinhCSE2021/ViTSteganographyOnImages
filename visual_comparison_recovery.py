"""
Visual Comparison Script for Advanced Recovery Model
This script creates side-by-side comparisons of original, watermarked, and recovered images.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# Import the corrected model
from test_corrected_recovery_model import CorrectAdvancedRecoveryTester

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to numpy
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    image_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    # Convert to PIL Image
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    return image_pil

def create_visual_comparison(model_path, image_path, output_path="recovery_comparison.png"):
    """Create a visual comparison showing original, watermarked, and recovered images."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the corrected model
    print("Loading corrected model...")
    tester = CorrectAdvancedRecoveryTester(model_path, device)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image_tensor = tester.load_and_preprocess_image(image_path)
    
    # Generate a test message
    message = torch.randint(0, 2, (1, 30)).float().to(device)
    message_str = ''.join(['1' if b > 0.5 else '0' for b in message.squeeze()])
    
    print(f"Test message (30 bits): {message_str}")
    
    with torch.no_grad():
        # Forward pass through the model
        watermarked_image = tester.model.encoder(image_tensor, message)
        decoded_message_logits, recovered_image = tester.model.decoder(watermarked_image)
        
        # Calculate metrics
        watermark_psnr = tester.model.__class__.__bases__[0].__dict__.get('compute_psnr', lambda x, y: 0)(image_tensor, watermarked_image)
        recovery_psnr = tester.model.__class__.__bases__[0].__dict__.get('compute_psnr', lambda x, y: 0)(image_tensor, recovered_image)
        
        # Use our own PSNR calculation
        def compute_psnr(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 2.0
            psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
            return psnr.item()
        
        watermark_psnr = compute_psnr(image_tensor, watermarked_image)
        recovery_psnr = compute_psnr(image_tensor, recovered_image)
        
        # Calculate message accuracy
        decoded_bits = (torch.sigmoid(decoded_message_logits) > 0.5).float()
        bit_accuracy = (message == decoded_bits).float().mean().item()
        decoded_str = ''.join(['1' if b > 0.5 else '0' for b in torch.sigmoid(decoded_message_logits).squeeze()])
    
    # Convert tensors to images
    original_img = tensor_to_image(image_tensor)
    watermarked_img = tensor_to_image(watermarked_image)
    recovered_img = tensor_to_image(recovered_image)
    
    # Create the comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image\n(Ground Truth)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Watermarked image
    axes[1].imshow(watermarked_img)
    axes[1].set_title(f'Watermarked Image\nPSNR: {watermark_psnr:.2f}dB\n(Message Embedded)', 
                      fontsize=14, fontweight='bold', color='blue')
    axes[1].axis('off')
    
    # Recovered image
    recovery_color = 'green' if recovery_psnr >= 32 else 'orange' if recovery_psnr >= 28 else 'red'
    axes[2].imshow(recovered_img)
    axes[2].set_title(f'Recovered Image\nPSNR: {recovery_psnr:.2f}dB 🎯\n(Message Extracted)', 
                      fontsize=14, fontweight='bold', color=recovery_color)
    axes[2].axis('off')
    
    # Add overall information
    plt.suptitle(f'Advanced Recovery Steganography Results\n'
                f'Target: 32dB Recovery PSNR | Achieved: {recovery_psnr:.2f}dB | Status: {"✅ TARGET ACHIEVED!" if recovery_psnr >= 32 else "📈 CLOSE" if recovery_psnr >= 30 else "🔧 TRAINING NEEDED"}\n'
                f'Message Accuracy: {bit_accuracy*100:.1f}% | Model: high_recovery_latent_epoch_10', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Visual comparison saved as: {output_path}")
    
    # Display detailed results
    print(f"\n=== DETAILED RESULTS ===")
    print(f"📊 Watermark Quality: {watermark_psnr:.2f}dB PSNR")
    print(f"🎯 Recovery Quality: {recovery_psnr:.2f}dB PSNR {'(TARGET ACHIEVED!)' if recovery_psnr >= 32 else '(Close to target)' if recovery_psnr >= 30 else '(Needs improvement)'}")
    print(f"💬 Message Accuracy: {bit_accuracy*100:.1f}%")
    print(f"📝 Original Message:  {message_str}")
    print(f"📝 Decoded Message:   {decoded_str}")
    print(f"🔍 Bit Differences:   {sum(c1 != c2 for c1, c2 in zip(message_str, decoded_str))}/30")
    
    # Create difference images for detailed analysis
    create_difference_analysis(image_tensor, watermarked_image, recovered_image, 
                              watermark_psnr, recovery_psnr)
    
    plt.show()
    return recovery_psnr, bit_accuracy

def create_difference_analysis(original, watermarked, recovered, watermark_psnr, recovery_psnr):
    """Create difference analysis visualization."""
    
    # Calculate difference images
    watermark_diff = torch.abs(watermarked - original)
    recovery_diff = torch.abs(recovered - original)
    
    # Convert to images
    original_img = tensor_to_image(original)
    watermark_diff_img = tensor_to_image(watermark_diff * 10)  # Amplify for visibility
    recovery_diff_img = tensor_to_image(recovery_diff * 10)   # Amplify for visibility
    
    # Create difference analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Original images
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tensor_to_image(watermarked))
    axes[0, 1].set_title(f'Watermarked\nPSNR: {watermark_psnr:.2f}dB', fontsize=12, fontweight='bold', color='blue')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(tensor_to_image(recovered))
    axes[0, 2].set_title(f'Recovered\nPSNR: {recovery_psnr:.2f}dB', fontsize=12, fontweight='bold', color='green' if recovery_psnr >= 32 else 'orange')
    axes[0, 2].axis('off')
    
    # Bottom row: Difference maps (amplified)
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title('Reference\n(Original)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(watermark_diff_img, cmap='hot')
    axes[1, 1].set_title('Watermark Artifacts\n(10x amplified)', fontsize=12, color='blue')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(recovery_diff_img, cmap='hot')
    axes[1, 2].set_title('Recovery Errors\n(10x amplified)', fontsize=12, color='red')
    axes[1, 2].axis('off')
    
    plt.suptitle('Steganography Analysis: Original vs Watermarked vs Recovered\n'
                'Red areas in difference maps show distortions', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_recovery_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as: detailed_recovery_analysis.png")
    
    plt.show()

def main():
    """Main function to create visual comparison."""
    model_path = "latent_runs/high_recovery_latent_20250730_203148/high_recovery_latent_epoch_10.pth"
    image_path = "data/val/val_class/ILSVRC2012_val_00007255.JPEG"
    
    print("🎯 Creating Visual Comparison of Advanced Recovery Model")
    print("=" * 60)
    
    try:
        recovery_psnr, bit_accuracy = create_visual_comparison(model_path, image_path)
        
        print("\n" + "=" * 60)
        print("🎉 VISUAL COMPARISON COMPLETED!")
        print(f"✅ Recovery PSNR: {recovery_psnr:.2f}dB (Target: 32dB)")
        print(f"✅ Message Accuracy: {bit_accuracy*100:.1f}%")
        print("📸 Check the generated images: recovery_comparison.png and detailed_recovery_analysis.png")
        
        if recovery_psnr >= 32:
            print("🎯 TARGET ACHIEVED! The model successfully recovers high-quality images!")
        
    except Exception as e:
        print(f"❌ Error creating visual comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

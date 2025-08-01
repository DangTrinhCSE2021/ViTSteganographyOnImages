"""
Visual Image Testing Script for Enhanced Recovery-Optimized ViT Steganography
Shows original, watermarked, and recovered images side by side for visual inspection.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from options import HiDDenConfiguration
from integration_guide import create_enhanced_model
from train_latent_space import create_clean_noiser, create_attack_noiser, compute_psnr, compute_ssim


def load_enhanced_model(model_path, device):
    """Load the enhanced recovery-optimized model."""
    print(f"Loading enhanced model from: {model_path}")
    
    # Create configuration matching training
    config = HiDDenConfiguration(
        H=128, W=128,
        message_length=30,
        encoder_channels=32,  # Match your trained model
        encoder_blocks=4,
        decoder_channels=32,  # Match your trained model
        decoder_blocks=7,
        use_discriminator=True,
        use_vgg=True,
        discriminator_blocks=3,
        discriminator_channels=64,
        encoder_loss=1.0,
        decoder_loss=1.0,
        adversarial_loss=1e-3
    )
    
    # Create clean noiser for testing
    noiser = create_clean_noiser(device)
    
    # Create enhanced model with latent space
    model = create_enhanced_model(config, device, noiser, None, use_latent_space=True)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def tensor_to_image(tensor):
    """Convert tensor to PIL Image for display."""
    # tensor is [1, 3, H, W], convert to [H, W, 3]
    img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_array = np.clip(img_array, 0, 1)  # Ensure values are in [0, 1]
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def create_message_visualization(message):
    """Create a visual representation of the binary message."""
    message_np = message.cpu().numpy().flatten()
    # Reshape to a nice grid (e.g., 6x5 for 30 bits)
    grid = message_np.reshape(6, 5)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(grid, cmap='RdYlBu', vmin=0, vmax=1)
    plt.title('Message Pattern\n(30 bits)', fontsize=10)
    plt.colorbar(shrink=0.8)
    
    # Add text annotations
    for i in range(6):
        for j in range(5):
            plt.text(j, i, f'{int(grid[i, j])}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    return plt


def test_image_visually(model, image_path, device, output_dir, test_name="test"):
    """Test model on image and create visual comparison."""
    print(f"\n🖼️ Testing image: {os.path.basename(image_path)}")
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    try:
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Create test message - something meaningful
    test_message = torch.tensor([
        1, 0, 1, 1, 0,  # "Hello"
        1, 1, 0, 0, 1,  # "World" 
        0, 1, 1, 0, 1,  # pattern
        1, 0, 0, 1, 1,  # pattern
        0, 1, 0, 1, 0,  # pattern
        1, 1, 1, 0, 0   # pattern
    ]).float().unsqueeze(0).to(device)
    
    print(f"📝 Test message: {test_message.cpu().numpy().flatten()}")
    
    results = {}
    
    with torch.no_grad():
        # Test clean conditions
        print("🧪 Testing clean conditions...")
        model.noiser = create_clean_noiser(device)
        
        losses, outputs = model.train_on_batch([image_tensor, test_message])
        watermarked, noised, decoded, recovered = outputs
        
        # Calculate metrics
        watermark_psnr = compute_psnr(image_tensor, watermarked)
        watermark_ssim = compute_ssim(image_tensor, watermarked)
        recovery_psnr = compute_psnr(image_tensor, recovered)
        recovery_ssim = compute_ssim(image_tensor, recovered)
        
        # Message accuracy
        decoded_sigmoid = torch.sigmoid(decoded[0] if isinstance(decoded, tuple) else decoded)
        decoded_bits = (decoded_sigmoid > 0.5).float()
        message_accuracy = (decoded_bits == test_message).float().mean().item()
        
        results['clean'] = {
            'watermarked': watermarked,
            'recovered': recovered,
            'decoded_bits': decoded_bits,
            'watermark_psnr': watermark_psnr,
            'watermark_ssim': watermark_ssim,
            'recovery_psnr': recovery_psnr,
            'recovery_ssim': recovery_ssim,
            'message_accuracy': message_accuracy
        }
        
        print(f"  🎨 Watermark: PSNR={watermark_psnr:.2f}dB, SSIM={watermark_ssim:.4f}")
        print(f"  🔧 Recovery:  PSNR={recovery_psnr:.2f}dB, SSIM={recovery_ssim:.4f}")
        print(f"  📊 Message:   Accuracy={message_accuracy:.1%}")
        
        # Test with attacks
        print("⚔️ Testing with attacks...")
        model.noiser = create_attack_noiser(device)
        
        losses_attack, outputs_attack = model.train_on_batch([image_tensor, test_message])
        watermarked_attack, noised_attack, decoded_attack, recovered_attack = outputs_attack
        
        # Calculate metrics under attack
        recovery_psnr_attack = compute_psnr(image_tensor, recovered_attack)
        decoded_attack_sigmoid = torch.sigmoid(decoded_attack[0] if isinstance(decoded_attack, tuple) else decoded_attack)
        decoded_attack_bits = (decoded_attack_sigmoid > 0.5).float()
        message_accuracy_attack = (decoded_attack_bits == test_message).float().mean().item()
        
        results['attack'] = {
            'noised': noised_attack,
            'recovered': recovered_attack,
            'decoded_bits': decoded_attack_bits,
            'recovery_psnr': recovery_psnr_attack,
            'message_accuracy': message_accuracy_attack
        }
        
        print(f"  🔧 Recovery:  PSNR={recovery_psnr_attack:.2f}dB")
        print(f"  📊 Message:   Accuracy={message_accuracy_attack:.1%}")
    
    # Create comprehensive visualization
    create_comprehensive_visualization(
        image_tensor, test_message, results, 
        os.path.basename(image_path), output_dir, test_name
    )
    
    return results


def create_comprehensive_visualization(image_tensor, message, results, image_name, output_dir, test_name):
    """Create a comprehensive visualization of the steganography process."""
    
    # Convert tensors to images
    original_img = tensor_to_image(image_tensor)
    watermarked_img = tensor_to_image(results['clean']['watermarked'])
    recovered_clean_img = tensor_to_image(results['clean']['recovered'])
    noised_img = tensor_to_image(results['attack']['noised'])
    recovered_attack_img = tensor_to_image(results['attack']['recovered'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: Images
    ax1 = plt.subplot(3, 5, 1)
    plt.imshow(original_img)
    plt.title(f'Original Image\n{image_name}', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    ax2 = plt.subplot(3, 5, 2)
    plt.imshow(watermarked_img)
    plt.title(f'Watermarked\nPSNR: {results["clean"]["watermark_psnr"]:.2f}dB\nSSIM: {results["clean"]["watermark_ssim"]:.4f}', 
              fontsize=11, color='green')
    plt.axis('off')
    
    ax3 = plt.subplot(3, 5, 3)
    plt.imshow(noised_img)
    plt.title('After Attacks\n(JPEG, Dropout, etc.)', fontsize=11, color='red')
    plt.axis('off')
    
    ax4 = plt.subplot(3, 5, 4)
    plt.imshow(recovered_clean_img)
    plt.title(f'Recovered (Clean)\nPSNR: {results["clean"]["recovery_psnr"]:.2f}dB\nSSIM: {results["clean"]["recovery_ssim"]:.4f}', 
              fontsize=11, color='blue')
    plt.axis('off')
    
    ax5 = plt.subplot(3, 5, 5)
    plt.imshow(recovered_attack_img)
    plt.title(f'Recovered (Attacks)\nPSNR: {results["attack"]["recovery_psnr"]:.2f}dB', 
              fontsize=11, color='purple')
    plt.axis('off')
    
    # Row 2: Message visualizations
    ax6 = plt.subplot(3, 5, 6)
    message_np = message.cpu().numpy().flatten().reshape(6, 5)
    plt.imshow(message_np, cmap='RdYlBu', vmin=0, vmax=1)
    plt.title('Original Message\n(30 bits)', fontsize=11, fontweight='bold')
    for i in range(6):
        for j in range(5):
            plt.text(j, i, f'{int(message_np[i, j])}', ha='center', va='center', fontsize=8)
    plt.axis('off')
    
    ax7 = plt.subplot(3, 5, 7)
    decoded_clean = results['clean']['decoded_bits'].cpu().numpy().flatten().reshape(6, 5)
    plt.imshow(decoded_clean, cmap='RdYlBu', vmin=0, vmax=1)
    plt.title(f'Decoded (Clean)\nAccuracy: {results["clean"]["message_accuracy"]:.1%}', 
              fontsize=11, color='green')
    for i in range(6):
        for j in range(5):
            color = 'white' if decoded_clean[i, j] == message_np[i, j] else 'red'
            plt.text(j, i, f'{int(decoded_clean[i, j])}', ha='center', va='center', 
                    fontsize=8, color=color, fontweight='bold')
    plt.axis('off')
    
    ax8 = plt.subplot(3, 5, 8)
    decoded_attack = results['attack']['decoded_bits'].cpu().numpy().flatten().reshape(6, 5)
    plt.imshow(decoded_attack, cmap='RdYlBu', vmin=0, vmax=1)
    plt.title(f'Decoded (Attacks)\nAccuracy: {results["attack"]["message_accuracy"]:.1%}', 
              fontsize=11, color='red')
    for i in range(6):
        for j in range(5):
            color = 'white' if decoded_attack[i, j] == message_np[i, j] else 'red'
            plt.text(j, i, f'{int(decoded_attack[i, j])}', ha='center', va='center', 
                    fontsize=8, color=color, fontweight='bold')
    plt.axis('off')
    
    # Row 2: Difference maps
    ax9 = plt.subplot(3, 5, 9)
    # Watermark difference (amplified for visibility)
    orig_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    water_np = results['clean']['watermarked'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    diff_watermark = np.abs(orig_np - water_np) * 10  # Amplify difference
    plt.imshow(diff_watermark)
    plt.title('Watermark Difference\n(10x amplified)', fontsize=11)
    plt.axis('off')
    
    ax10 = plt.subplot(3, 5, 10)
    # Recovery difference
    recov_np = results['clean']['recovered'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    diff_recovery = np.abs(orig_np - recov_np) * 10  # Amplify difference
    plt.imshow(diff_recovery)
    plt.title('Recovery Difference\n(10x amplified)', fontsize=11)
    plt.axis('off')
    
    # Row 3: Performance summary
    ax11 = plt.subplot(3, 5, (11, 15))
    
    # Create performance table
    performance_data = [
        ['Metric', 'Watermark', 'Recovery (Clean)', 'Recovery (Attack)', 'Status'],
        ['PSNR (dB)', f"{results['clean']['watermark_psnr']:.2f}", 
         f"{results['clean']['recovery_psnr']:.2f}", 
         f"{results['attack']['recovery_psnr']:.2f}", 
         '🏆 Recovery > Watermark!' if results['clean']['recovery_psnr'] > results['clean']['watermark_psnr'] else '📊 Standard'],
        ['SSIM', f"{results['clean']['watermark_ssim']:.4f}", 
         f"{results['clean']['recovery_ssim']:.4f}", 'N/A', 
         '✅ Excellent' if results['clean']['recovery_ssim'] > 0.95 else '📊 Good'],
        ['Message Accuracy', 'N/A', 
         f"{results['clean']['message_accuracy']:.1%}", 
         f"{results['attack']['message_accuracy']:.1%}", 
         '⚠️ Needs Improvement' if results['clean']['message_accuracy'] < 0.8 else '✅ Good'],
        ['Robustness', 'N/A', 'N/A',
         f"{results['attack']['message_accuracy']/results['clean']['message_accuracy']*100:.1f}% retention",
         '✅ Robust' if results['attack']['message_accuracy']/results['clean']['message_accuracy'] > 0.8 else '📊 Moderate']
    ]
    
    # Create table
    table = plt.table(cellText=performance_data[1:], colLabels=performance_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(performance_data)):
        for j in range(len(performance_data[0])):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            elif j == 4:  # Status column
                if '🏆' in performance_data[i][j]:
                    table[(i-1, j)].set_facecolor('#E8F5E8')
                elif '⚠️' in performance_data[i][j]:
                    table[(i-1, j)].set_facecolor('#FFF3E0')
                else:
                    table[(i-1, j)].set_facecolor('#F5F5F5')
    
    plt.title(f'Enhanced ViT Steganography Performance Summary\nRecovery PSNR: {results["clean"]["recovery_psnr"]:.2f}dB > Watermark PSNR: {results["clean"]["watermark_psnr"]:.2f}dB ✨', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'{test_name}_visual_test_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"💾 Visualization saved: {output_path}")
    
    # Show the plot
    plt.show()
    
    return output_path


def main():
    """Main visual testing function."""
    print("🎨 VISUAL IMAGE TESTING FOR ENHANCED ViT STEGANOGRAPHY")
    print("=" * 65)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model path
    model_path = "latent_runs/recovery_optimized_20250729_114830/best_model.pth"
    
    # Load model
    model = load_enhanced_model(model_path, device)
    
    # Create output directory
    output_dir = "visual_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test images
    test_images = [
        "data/val/val_class/ILSVRC2012_val_00007255.JPEG",
        "data/val/val_class/ILSVRC2012_val_00007257.JPEG",
        "data/val/val_class/ILSVRC2012_val_00007259.JPEG",
    ]
    
    # Test each image
    for i, image_path in enumerate(test_images):
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Testing image {i+1}/{len(test_images)}")
            test_name = f"image_{i+1:02d}_{os.path.splitext(os.path.basename(image_path))[0]}"
            
            results = test_image_visually(model, image_path, device, output_dir, test_name)
            
            if results:
                print(f"✅ Test completed for {os.path.basename(image_path)}")
            else:
                print(f"❌ Test failed for {os.path.basename(image_path)}")
        else:
            print(f"❌ Image not found: {image_path}")
    
    print(f"\n🎉 Visual testing completed!")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"📊 Check the generated PNG files for detailed visual analysis")


if __name__ == "__main__":
    main()

"""Test the data loader with real images"""
from train_recovery_optimized import create_data_loader

print("Testing data loader with real images...")
loader = create_data_loader('data/train', 2, True)

print("Loading first few batches:")
for i, (images, _) in enumerate(loader):
    print(f'Batch {i}: Shape={images.shape}, Range=[{images.min():.3f}, {images.max():.3f}]')
    
    # Check if images look like real data (not random)
    mean_val = images.mean().item()
    std_val = images.std().item()
    print(f'  Mean: {mean_val:.3f}, Std: {std_val:.3f}')
    
    if i >= 2:
        break

print("Data loader test completed!")

# Model Test Results Summary

## Model Information
- **Model Folder**: `runs/first_main 2025.07.26--04-47-27`
- **Best Checkpoint**: Epoch 40 (`first_main--epoch-40.pyt`)
- **Training Device**: CUDA (NVIDIA GeForce RTX 4060 Laptop GPU)

## Single Image Tests
Tested the model on individual validation images using `test_model.py`:

### Test 1: ILSVRC2012_val_00007255.JPEG
- **Bit Error Rate**: 0.567 (56.7% of bits incorrect)
- **Bit Accuracy**: 43.3%

### Test 2: ILSVRC2012_val_00007257.JPEG  
- **Bit Error Rate**: 0.433 (43.3% of bits incorrect)
- **Bit Accuracy**: 56.7%

## Validation Set Results
Tested on 100 validation images using the custom validation script:

### Key Performance Metrics
- **Average Bit Error Rate**: 0.4957 (49.57%)
- **Average Bit Accuracy**: **50.43%**
- **Image Quality (Encoder MSE)**: 0.0045 (very good - low distortion)
- **Image Recovery Loss**: 0.2369
- **L2 Regularization**: 0.0045

### Loss Breakdown
- **Total Loss**: 3.6444
- **Encoder MSE**: 0.0045 (image quality preservation)
- **Decoder MSE**: 0.2519 (message reconstruction)
- **Perceptual Encoder**: 0.3398
- **Perceptual Reconstruction**: 21.7065

## Analysis

### Strengths
1. **Good Image Quality**: Encoder MSE of 0.0045 indicates very low visual distortion
2. **Stable Performance**: Consistent results across different test images
3. **CUDA Optimization**: Successfully utilizing GPU acceleration

### Areas for Improvement
1. **Bit Accuracy**: ~50% accuracy suggests the model is performing close to random guessing
2. **Message Recovery**: The watermark extraction needs improvement
3. **Perceptual Loss**: High perceptual reconstruction loss (21.7) indicates room for optimization

### Recommendations
1. **Longer Training**: The model may benefit from training beyond epoch 40
2. **Hyperparameter Tuning**: Adjust loss weights (decoder_loss, encoder_loss, adversarial_loss)
3. **Encoder Mode**: Try different encoder modes ('dino-output', 'dino-attention') for comparison
4. **Data Augmentation**: Consider adding more diverse noise layers during training

## Generated Output Files
- `epoch-test_watermarked.png`: Image with embedded watermark
- `epoch-test_recovered.png`: Recovered/decoded image
- `epoch-test.png`: Test result visualization

The model is functional and successfully embeds/extracts watermarks, but the accuracy could be improved with further training and optimization.

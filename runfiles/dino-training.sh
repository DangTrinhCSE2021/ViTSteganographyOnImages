#!/bin/bash

# DINO ViT Training Script for Steganography
# This script trains the model using DINO ViT as the encoder

echo "=============================================="
echo "Starting DINO ViT Steganography Training"
echo "=============================================="

# Set experiment name with timestamp
TIMESTAMP=$(date +"%Y.%m.%d--%H-%M-%S")
EXPERIMENT_NAME="dino_output_$TIMESTAMP"

echo "Experiment Name: $EXPERIMENT_NAME"
echo "Encoder Mode: dino-output"
echo ""

# Training parameters
DATA_DIR="data"
BATCH_SIZE=16  # Reduced batch size for DINO to fit in memory
EPOCHS=100
MESSAGE_LENGTH=30
IMAGE_SIZE=128

# Run training with DINO output mode
python main.py new \
    --data-dir "$DATA_DIR" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --name "$EXPERIMENT_NAME" \
    --size $IMAGE_SIZE \
    --message $MESSAGE_LENGTH \
    --encoder-mode dino-output \
    --tensorboard \
    --noise 'dropout(0.3)' 'cropout((0.1, 0.3), (0.1, 0.3))' 'jpeg(50, 95)'

echo ""
echo "=============================================="
echo "DINO ViT Training Completed!"
echo "Results saved in: experiments/$EXPERIMENT_NAME"
echo "=============================================="

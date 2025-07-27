@echo off
echo ==============================================
echo Starting ViT Steganography Training
echo ==============================================

REM Set experiment name with timestamp
set TIMESTAMP=%date:~-4,4%.%date:~-10,2%.%date:~-7,2%--%time:~0,2%-%time:~3,2%-%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set EXPERIMENT_NAME=vit_training_%TIMESTAMP%

echo Experiment Name: %EXPERIMENT_NAME%
echo Encoder Mode: ViT (Vision Transformer)
echo.

REM Training parameters
set DATA_DIR=data
set BATCH_SIZE=16
set EPOCHS=100
set MESSAGE_LENGTH=30
set IMAGE_SIZE=128

echo Starting training with ViT encoder...
echo Batch Size: %BATCH_SIZE%
echo Epochs: %EPOCHS%
echo Image Size: %IMAGE_SIZE%x%IMAGE_SIZE%
echo Message Length: %MESSAGE_LENGTH% bits
echo.

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo Error: Data directory '%DATA_DIR%' not found!
    echo Please make sure the data directory exists with train/ and val/ subdirectories.
    pause
    exit /b 1
)

REM Run training with ViT encoder
python main.py new ^
    --data-dir "%DATA_DIR%" ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --name "%EXPERIMENT_NAME%" ^
    --size %IMAGE_SIZE% ^
    --message %MESSAGE_LENGTH% ^
    --tensorboard ^
    --noise "dropout(0.3)" "cropout((0.1, 0.3), (0.1, 0.3))" "jpeg(50, 95)"

if %errorlevel% neq 0 (
    echo.
    echo Training failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ==============================================
echo ViT Training Completed Successfully!
echo Results saved in: experiments\%EXPERIMENT_NAME%
echo ==============================================
pause

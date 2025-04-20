#!/bin/bash
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Installing PyTorch CPU..."
pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

echo "Creating necessary directories..."
mkdir -p templates
mkdir -p model

echo "Build completed successfully"
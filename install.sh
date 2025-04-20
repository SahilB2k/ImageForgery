#!/bin/bash
set -e

# Use specific pip version known to work well
echo "Installing specific pip version..."
python -m pip install --upgrade pip==21.3.1

# Install minimal requirements first
echo "Installing minimal requirements..."
pip install --no-cache-dir -r requirements-minimal.txt

# Install PyTorch and numpy
echo "Installing PyTorch and numpy..."
pip install --no-cache-dir -r requirements-torch.txt

echo "Installation completed successfully"
#!/bin/bash
set -e

echo "Upgrading pip..."
pip install --upgrade pip==22.3.1

echo "Installing base requirements..."
pip install --no-cache-dir -r requirements.txt

echo "Installing PyTorch CPU..."
pip install --no-cache-dir -r torch-requirements.txt

echo "Installation completed successfully"
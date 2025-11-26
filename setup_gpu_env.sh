#!/bin/bash
# Setup script for TensorFlow Metal GPU support on Mac M4
# This creates a conda environment with Python 3.11 (ARM64) and installs required packages

echo "Creating conda environment 'tf-metal' with Python 3.11 (ARM64)..."
CONDA_SUBDIR=osx-arm64 conda create -n tf-metal python=3.11 -y

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tf-metal

echo ""
echo "Configuring environment to use ARM64 packages..."
conda config --env --set subdir osx-arm64

echo ""
echo "Installing TensorFlow for Mac with Metal support..."
echo "Note: Install tensorflow-macos FIRST, then tensorflow-metal"
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

echo ""
echo "Installing other dependencies..."
python -m pip install pandas scikit-learn numpy huggingface-hub pyarrow

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use this environment:"
echo "  conda activate tf-metal"
echo "  python model.py"
echo ""
echo "To verify GPU support:"
echo "  python check_gpu.py"


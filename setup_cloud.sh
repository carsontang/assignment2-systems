#!/bin/bash
set -euo pipefail

echo "Setting up CS336 Assignment 2 on cloud GPU..."

# Check if running on GPU instance
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: No GPU detected. This might be a CPU-only instance."
fi

# Install Python if needed (most cloud GPU instances have it)
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.11+ first."
    exit 1
fi

echo "Python version: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install cs336-basics as editable
echo "Installing cs336-basics module..."
pip install -e ./cs336-basics

# Install cs336-systems as editable
echo "Installing cs336-systems module..."
pip install -e .

# Run a quick test
echo "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cs336_basics; import cs336_systems; print('Modules imported successfully')"

echo ""
echo "Setup complete! To run benchmarks:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run benchmarking: python cs336_systems/benchmarking_script.py"
echo ""
echo "To run tests:"
echo "pytest -v ./tests"
"""
Setup script for running CS336 Assignment 2 in Google Colab
Copy and run this in a Colab cell to set up the environment
"""

# First cell - Setup and installation
setup_code = '''
# Check GPU availability
import subprocess
import sys
!nvidia-smi

# Clone the repository (replace with your repo URL)
!git clone https://github.com/YOUR_USERNAME/assignment2-systems.git
%cd assignment2-systems

# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q numpy tqdm matplotlib pandas pytest regex humanfriendly wandb triton

# Install local packages
!pip install -e ./cs336-basics
!pip install -e .

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Import modules
import cs336_basics
import cs336_systems
print("Modules imported successfully!")
'''

# Second cell - Run benchmarking
benchmark_code = '''
# Run the benchmarking script
!python cs336_systems/benchmarking_script.py --batchsize 32 --context 128 --num-runs 5
'''

# Third cell - Run tests
test_code = '''
# Run specific tests
!pytest -v ./tests/test_attention.py -k "flashattention_pytorch"
'''

# Fourth cell - Interactive benchmarking
interactive_code = '''
import sys
sys.path.append('/content/assignment2-systems')

from cs336_systems.benchmarking_script import Config, benchmark_model, generate_random_data
from cs336_basics.model import BasicsTransformerLM
import torch

# Create config
config = Config(
    batch_size=64,
    vocab_size=50000,
    context_length=256,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    d_ff=4096
)

# Create model
model = BasicsTransformerLM(
    vocab_size=config.vocab_size,
    context_length=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
    rope_theta=config.rope_theta
).cuda()

# Generate data
x = generate_random_data(config).cuda()

# Benchmark
print(f"Benchmarking model with input shape {x.shape}")
timing_stats = benchmark_model(model, x, num_runs=10, num_warmup=2)
print(f"Average time: {timing_stats['avg']:.4f} seconds")
print(f"Min time: {timing_stats['min']:.4f} seconds")
'''

print("Copy the code sections above into separate Colab cells")
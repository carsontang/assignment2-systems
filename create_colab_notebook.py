#!/usr/bin/env python3
"""
Script to create a Colab notebook URL or generate code snippets for CS336 Assignment 2
"""

def create_colab_url(github_url):
    """Convert GitHub URL to Colab URL"""
    if "github.com" in github_url:
        colab_url = github_url.replace("github.com", "colab.research.google.com/github")
        if colab_url.endswith(".ipynb"):
            return colab_url
        else:
            # Assume the notebook is in the root
            return colab_url.rstrip("/") + "/blob/main/CS336_Assignment2_Colab.ipynb"
    return None

def print_colab_setup():
    """Print the setup code for manual copy-paste into Colab"""
    print("=" * 60)
    print("CS336 Assignment 2 - Google Colab Setup")
    print("=" * 60)
    print("\nCopy and paste these code blocks into separate Colab cells:\n")
    
    print("### Cell 1: Setup and Clone Repository ###")
    print("""
# Check GPU
!nvidia-smi

# Method 1: Clone from GitHub (replace with your repo)
!git clone https://github.com/YOUR_USERNAME/assignment2-systems.git
%cd assignment2-systems

# Method 2: Upload tar.gz file
# from google.colab import files
# uploaded = files.upload()  # Upload cs336_assignment2_cloud.tar.gz
# !tar -xzf cs336_assignment2_cloud.tar.gz

# Method 3: Mount Google Drive (if you've uploaded there)
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/assignment2-systems .
# %cd assignment2-systems
""")
    
    print("\n### Cell 2: Install Dependencies ###")
    print("""
# Install dependencies
!pip install -q numpy tqdm matplotlib pandas pytest regex humanfriendly wandb triton

# Install local packages
!pip install -e ./cs336-basics
!pip install -e .

# Verify
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
""")
    
    print("\n### Cell 3: Run Benchmarks ###")
    print("""
# Quick benchmark
!python cs336_systems/benchmarking_script.py

# Custom benchmark with larger model
!python cs336_systems/benchmarking_script.py \\
    --batchsize 64 \\
    --context 256 \\
    --dmodel 1024 \\
    --nlayers 24 \\
    --num-runs 5
""")
    
    print("\n### Cell 4: Run Tests ###")
    print("""
# Run all tests
!pytest -v ./tests/

# Run specific test
# !pytest -v ./tests/test_attention.py::test_flashattention_pytorch
""")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        colab_url = create_colab_url(sys.argv[1])
        if colab_url:
            print(f"Open in Colab: {colab_url}")
    else:
        print_colab_setup()
        print("\n" + "=" * 60)
        print("To create a Colab link from your GitHub repo:")
        print("python create_colab_notebook.py https://github.com/YOUR_USERNAME/assignment2-systems")
        print("=" * 60)
#!/bin/bash
# Setup script for Qwen2.5-7B QLoRA fine-tuning environment
# Target: RTX 2080 (8GB VRAM) with CUDA 11.8

set -e

echo "=== Setting up Qwen2.5-7B QLoRA Fine-tuning Environment ==="

# Create conda environment
echo "Creating conda environment 'examiner'..."
conda create -n examiner python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate examiner

# Install PyTorch with CUDA 11.8 support
echo "Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install core dependencies
echo "Installing core ML dependencies..."
pip install transformers==4.36.2
pip install peft==0.7.1
pip install datasets==2.16.1
pip install bitsandbytes==0.41.3
pip install accelerate==0.25.0
pip install trl==0.7.10

# Install PDF processing and utilities
echo "Installing utilities..."
pip install pdfplumber==0.10.3
pip install tqdm==4.66.1
pip install scipy==1.11.4
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

# Verify CUDA availability
echo "Verifying CUDA setup..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create project directories
echo "Creating project directories..."
mkdir -p data/pdfs
mkdir -p data/datasets
mkdir -p models/checkpoints
mkdir -p models/final
mkdir -p logs

echo ""
echo "=== Setup Complete ==="
echo "Activate environment with: conda activate examiner"
echo "Place PDF files in: data/pdfs/"
echo "Start fine-tuning with: python qwen_finetune.py --corpus_dir data/pdfs --output_dir models/final"
#!/bin/bash
# GCP GPU Instance Setup Script
# Prepares a GCP instance for running the Aristotle verification pipeline
# Target: Ubuntu 22.04 LTS with NVIDIA T4 GPU

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================="
echo "        GCP GPU INSTANCE SETUP FOR ARISTOTLE VERIFICATION             "
echo "======================================================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}✗${NC} Do not run this script as root!"
    echo "   Run as regular user: bash gcp_setup.sh"
    exit 1
fi

# Update system packages
echo -e "${BLUE}[1/10]${NC} Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
echo -e "${GREEN}✓${NC} System packages updated"
echo ""

# Check and install NVIDIA drivers if needed
echo -e "${BLUE}[2/10]${NC} Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA drivers already installed"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
else
    echo -e "${YELLOW}⚠${NC}  NVIDIA drivers not found. Installing..."
    
    # Install drivers
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers install
    
    echo -e "${YELLOW}⚠${NC}  NVIDIA drivers installed"
    echo -e "${RED}▶${NC}  REBOOT REQUIRED!"
    echo ""
    echo "   After reboot, run this script again to continue setup:"
    echo "   bash gcp_setup.sh"
    echo ""
    read -p "Reboot now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        echo "   Please reboot manually: sudo reboot"
        exit 0
    fi
fi
echo ""

# Verify GPU with nvidia-smi
echo -e "${BLUE}[3/10]${NC} Verifying GPU..."
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU detected: ${GPU_NAME}"
    echo "   Memory: ${GPU_MEMORY} MB"
    echo "   Driver: ${DRIVER_VERSION}"
else
    echo -e "${RED}✗${NC} GPU verification failed!"
    echo "   Run: nvidia-smi"
    exit 1
fi
echo ""

# Install Python 3.11 and dependencies
echo -e "${BLUE}[4/10]${NC} Installing Python 3.11 and dependencies..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl wget
echo -e "${GREEN}✓${NC} Python 3.11 installed"
python3.11 --version
echo ""

# Clone examiner repository if not exists
echo -e "${BLUE}[5/10]${NC} Setting up examiner repository..."
REPO_DIR="$HOME/examiner"
if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}⚠${NC}  Repository already exists: $REPO_DIR"
    cd "$REPO_DIR"
    echo "   Pulling latest changes..."
    git pull origin main || echo "   (Could not pull, continuing...)"
else
    echo "   Cloning repository..."
    cd "$HOME"
    git clone https://github.com/humanaiconvention/examiner.git
    cd examiner
    echo -e "${GREEN}✓${NC} Repository cloned"
fi
echo ""

# Create Python virtual environment
echo -e "${BLUE}[6/10]${NC} Creating Python virtual environment..."
if [ -d "$REPO_DIR/.venv" ]; then
    echo -e "${YELLOW}⚠${NC}  Virtual environment already exists"
else
    python3.11 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created: .venv"
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Upgrade pip
echo -e "${BLUE}[7/10]${NC} Upgrading pip..."
pip install --upgrade pip -q
echo -e "${GREEN}✓${NC} pip upgraded"
echo ""

# Install PyTorch with CUDA 12.1 support
echo -e "${BLUE}[8/10]${NC} Installing PyTorch with CUDA support..."
echo "   This may take several minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
echo -e "${GREEN}✓${NC} PyTorch installed"
echo ""

# Install project dependencies
echo -e "${BLUE}[9/10]${NC} Installing project dependencies..."
if [ -f "requirements-gcp.txt" ]; then
    pip install -r requirements-gcp.txt -q
    echo -e "${GREEN}✓${NC} Installed from requirements-gcp.txt"
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo -e "${GREEN}✓${NC} Installed from requirements.txt"
else
    echo -e "${YELLOW}⚠${NC}  No requirements file found, installing core dependencies..."
    pip install transformers peft datasets bitsandbytes accelerate trl PyMuPDF tqdm scipy sentencepiece protobuf -q
fi

# Install additional dependencies
echo "   Installing Aristotle verifier dependencies..."
pip install aiohttp requests -q
echo -e "${GREEN}✓${NC} All dependencies installed"
echo ""

# Verify installation with Python GPU test
echo -e "${BLUE}[10/10]${NC} Verifying installation..."
VERIFY_OUTPUT=$(python3 << 'EOF'
import sys
try:
    import torch
    import transformers
    import aiohttp
    import requests
    
    print("=" * 70)
    print("Installation Verification")
    print("=" * 70)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=" * 70)
        print("✓ GPU verification PASSED")
    else:
        print("=" * 70)
        print("✗ GPU verification FAILED - CUDA not available!")
        sys.exit(1)
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF
)

echo "$VERIFY_OUTPUT"
echo ""

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Installation verification passed!"
else
    echo -e "${RED}✗${NC} Installation verification failed!"
    echo "   Check error messages above"
    exit 1
fi
echo ""

# Display next steps
echo "======================================================================="
echo "                        SETUP COMPLETE!                                "
echo "======================================================================="
echo ""
echo -e "${GREEN}✓${NC} All components installed and verified"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure Aristotle API key:"
echo "   export ARISTOTLE_API_KEY='your-api-key-here'"
echo "   echo \"export ARISTOTLE_API_KEY='your-key'\" >> ~/.bashrc"
echo ""
echo "2. Activate virtual environment:"
echo "   cd ~/examiner"
echo "   source .venv/bin/activate"
echo ""
echo "3. Run verification:"
echo "   bash run_gcp_verification.sh"
echo ""
echo "   Or manually:"
echo "   python aristotle_verifier.py --failed-samples evaluation_results/failed_samples.json"
echo ""
echo "4. Monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "5. When done, stop the instance (from local machine):"
echo "   gcloud compute instances stop [INSTANCE-NAME] --zone=[ZONE]"
echo ""
echo "======================================================================="
echo ""
echo -e "${YELLOW}⚠  Remember to stop your GCP instance when not in use to avoid charges!${NC}"
echo "======================================================================="

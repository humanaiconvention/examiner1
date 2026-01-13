#!/bin/bash
# Automated Aristotle Verification Script for GCP GPU Instances
# Runs formal verification on failed samples with GPU acceleration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================="
echo "        ARISTOTLE VERIFICATION ON GCP GPU                             "
echo "======================================================================="
echo ""

# Check if running on GCP
echo -e "${BLUE}[1/9]${NC} Checking GCP environment..."
if curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/id" > /dev/null 2>&1; then
    INSTANCE_ID=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/id")
    INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/name")
    ZONE=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | cut -d'/' -f4)
    echo -e "${GREEN}✓${NC} Running on GCP instance: ${INSTANCE_NAME} (Zone: ${ZONE})"
else
    echo -e "${YELLOW}⚠${NC}  Not running on GCP (or metadata server unavailable)"
    echo "   This script is optimized for GCP GPU instances"
fi
echo ""

# Check if virtual environment is activated
echo -e "${BLUE}[2/9]${NC} Checking Python virtual environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓${NC} Virtual environment active: $VIRTUAL_ENV"
else
    echo -e "${YELLOW}⚠${NC}  Virtual environment not activated"
    if [ -d ".venv" ]; then
        echo "   Activating .venv..."
        source .venv/bin/activate
        echo -e "${GREEN}✓${NC} Virtual environment activated"
    elif [ -d "venv" ]; then
        echo "   Activating venv..."
        source venv/bin/activate
        echo -e "${GREEN}✓${NC} Virtual environment activated"
    else
        echo -e "${RED}✗${NC} No virtual environment found (.venv or venv)"
        echo "   Run: python3 -m venv .venv && source .venv/bin/activate"
        exit 1
    fi
fi
echo ""

# Verify GPU availability
echo -e "${BLUE}[3/9]${NC} Verifying GPU availability with PyTorch..."
GPU_CHECK=$(python3 << EOF
import torch
import sys
if torch.cuda.is_available():
    print(f"AVAILABLE|{torch.cuda.get_device_name(0)}|{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
    sys.exit(0)
else:
    print("NOT_AVAILABLE")
    sys.exit(1)
EOF
)

if [ $? -eq 0 ]; then
    IFS='|' read -r STATUS GPU_NAME GPU_MEMORY <<< "$GPU_CHECK"
    echo -e "${GREEN}✓${NC} GPU detected: ${GPU_NAME} (${GPU_MEMORY} GB)"
else
    echo -e "${RED}✗${NC} GPU not available in PyTorch!"
    echo "   Check CUDA installation and PyTorch GPU support"
    echo "   Run: nvidia-smi"
    exit 1
fi
echo ""

# Validate ARISTOTLE_API_KEY
echo -e "${BLUE}[4/9]${NC} Validating Aristotle API key..."
if [ -z "$ARISTOTLE_API_KEY" ]; then
    echo -e "${RED}✗${NC} ARISTOTLE_API_KEY environment variable not set!"
    echo "   Set it with: export ARISTOTLE_API_KEY='your-api-key-here'"
    echo "   Or add to ~/.bashrc for persistence"
    exit 1
else
    # Mask the key for display (show first 8 and last 4 characters)
    KEY_LENGTH=${#ARISTOTLE_API_KEY}
    if [ $KEY_LENGTH -gt 12 ]; then
        MASKED_KEY="${ARISTOTLE_API_KEY:0:8}...${ARISTOTLE_API_KEY: -4}"
    else
        MASKED_KEY="***"
    fi
    echo -e "${GREEN}✓${NC} API key configured: ${MASKED_KEY}"
fi
echo ""

# Check for failed samples file
echo -e "${BLUE}[5/9]${NC} Checking for failed samples..."
FAILED_SAMPLES_FILE="evaluation_results/failed_samples.json"
if [ ! -f "$FAILED_SAMPLES_FILE" ]; then
    echo -e "${RED}✗${NC} Failed samples file not found: ${FAILED_SAMPLES_FILE}"
    echo "   Run evaluation first: python evaluate_model.py"
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(python3 -c "import json; data=json.load(open('$FAILED_SAMPLES_FILE')); print(len(data))")
echo -e "${GREEN}✓${NC} Found failed samples file: ${FAILED_SAMPLES_FILE}"
echo "   Total samples to verify: ${SAMPLE_COUNT}"
echo ""

# Estimate time and cost
echo -e "${BLUE}[6/9]${NC} Estimating time and cost..."
# Assume ~6 seconds per sample average (can vary based on complexity)
ESTIMATED_SECONDS=$((SAMPLE_COUNT * 6))
ESTIMATED_MINUTES=$((ESTIMATED_SECONDS / 60))
ESTIMATED_HOURS=$((ESTIMATED_MINUTES / 60))
REMAINING_MINUTES=$((ESTIMATED_MINUTES % 60))

# GCP T4 cost: ~$0.54/hour
COST_PER_HOUR=0.54
ESTIMATED_COST=$(python3 -c "print(f'{$ESTIMATED_SECONDS / 3600 * $COST_PER_HOUR:.2f}')")

if [ $ESTIMATED_HOURS -gt 0 ]; then
    echo "   Estimated time: ${ESTIMATED_HOURS}h ${REMAINING_MINUTES}m"
else
    echo "   Estimated time: ${ESTIMATED_MINUTES} minutes"
fi
echo "   Estimated cost: ~\$${ESTIMATED_COST} (T4 GPU @ \$0.54/hour)"
echo ""

# Confirmation prompt
echo -e "${YELLOW}[7/9]${NC} Ready to start verification"
echo "   Samples: ${SAMPLE_COUNT}"
echo "   GPU: ${GPU_NAME}"
echo "   Output: verification_results/"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠${NC}  Verification cancelled"
    exit 0
fi
echo ""

# Run verification
echo -e "${BLUE}[8/9]${NC} Running Aristotle verification..."
echo "======================================================================="
START_TIME=$(date +%s)

# Run the verifier
python aristotle_verifier.py \
    --failed-samples "$FAILED_SAMPLES_FILE" \
    --output-dir verification_results

VERIFICATION_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MINUTES=$((DURATION / 60))
DURATION_SECONDS=$((DURATION % 60))

echo "======================================================================="
echo ""

if [ $VERIFICATION_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Verification completed successfully!"
else
    echo -e "${RED}✗${NC} Verification failed with exit code: ${VERIFICATION_EXIT_CODE}"
    exit $VERIFICATION_EXIT_CODE
fi
echo ""

# Display results summary
echo -e "${BLUE}[9/9]${NC} Verification Summary"
echo "-----------------------------------------------------------------------"
echo "   Duration: ${DURATION_MINUTES}m ${DURATION_SECONDS}s"

# Parse results if available
if [ -f "verification_results/verification_results.json" ]; then
    RESULTS=$(python3 << EOF
import json
try:
    with open('verification_results/verification_results.json') as f:
        data = json.load(f)
    print(f"{data.get('total_samples', 0)}|{data.get('verified', 0)}|{data.get('errors', 0)}")
except:
    print("0|0|0")
EOF
    )
    IFS='|' read -r TOTAL VERIFIED ERRORS <<< "$RESULTS"
    echo "   Total samples: ${TOTAL}"
    echo "   Verified: ${VERIFIED}"
    echo "   Errors: ${ERRORS}"
    echo ""
    echo "   Results saved to: verification_results/verification_results.json"
    echo "   Training data: verification_results/formal_verification_training.json"
else
    echo "   Results file not found (check for errors above)"
fi
echo "-----------------------------------------------------------------------"
echo ""

# Calculate actual cost
ACTUAL_COST=$(python3 -c "print(f'{$DURATION / 3600 * $COST_PER_HOUR:.2f}')")
echo -e "${GREEN}✓${NC} Actual cost: ~\$${ACTUAL_COST}"
echo ""

# Reminder to stop instance
echo "======================================================================="
echo -e "${YELLOW}⚠  IMPORTANT: Remember to stop your GCP instance!${NC}"
echo "======================================================================="
echo ""
echo "To stop the instance (from your local machine):"
echo "  gcloud compute instances stop ${INSTANCE_NAME:-examiner-aristotle-gpu} --zone=${ZONE:-us-central1-a}"
echo ""
echo "To download results (from your local machine):"
echo "  gcloud compute scp ${INSTANCE_NAME:-examiner-aristotle-gpu}:~/examiner/verification_results/*.json ./ --zone=${ZONE:-us-central1-a} --recurse"
echo ""
echo "Estimated savings by stopping: ~\$12.96/day (~\$394/month)"
echo "======================================================================="
echo ""

echo -e "${GREEN}✓${NC} All done!"

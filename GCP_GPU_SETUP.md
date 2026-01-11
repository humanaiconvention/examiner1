# Google Cloud Platform (GCP) GPU Setup Guide

Complete guide for running the Aristotle verification pipeline on GCP GPU instances.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Create GPU Instance](#step-1-create-gpu-instance)
- [Step 2: Connect via SSH](#step-2-connect-via-ssh)
- [Step 3: Install NVIDIA Drivers and CUDA](#step-3-install-nvidia-drivers-and-cuda)
- [Step 4: Setup Python and Dependencies](#step-4-setup-python-and-dependencies)
- [Step 5: Configure Aristotle API Key](#step-5-configure-aristotle-api-key)
- [Step 6: Verify GPU in Python](#step-6-verify-gpu-in-python)
- [Step 7: Run the Verifier](#step-7-run-the-verifier)
- [Step 8: Monitor GPU Usage](#step-8-monitor-gpu-usage)
- [Step 9: Transfer Results](#step-9-transfer-results)
- [Cost Management](#cost-management)
- [Estimated Costs](#estimated-costs)
- [Troubleshooting](#troubleshooting)
- [Automation](#automation)

---

## Prerequisites

Before starting, ensure you have:

1. **GCP Account** with billing enabled
2. **Aristotle API Key** from https://api.aristotle.ai/
3. **gcloud CLI** installed on your local machine
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```
4. **GPU Quota** in your GCP project (request if needed)
5. **GCP Project** selected and configured

---

## Step 1: Create GPU Instance

### Option A: Using gcloud CLI (Recommended)

```bash
# Create GPU instance with NVIDIA T4
gcloud compute instances create examiner-aristotle-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-standard \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

**Instance Specifications:**
- **Machine Type:** `n1-standard-4` (4 vCPUs, 15 GB RAM)
- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **OS:** Ubuntu 22.04 LTS
- **Disk:** 100 GB standard persistent disk
- **Zone:** `us-central1-a` (change based on your location/quota)

### Option B: Using GCP Console UI

1. Navigate to **Compute Engine** > **VM instances**
2. Click **Create Instance**
3. Configure:
   - **Name:** `examiner-aristotle-gpu`
   - **Region:** `us-central1` (Iowa)
   - **Zone:** `us-central1-a`
   - **Machine configuration:**
     - **Series:** N1
     - **Machine type:** `n1-standard-4`
   - **GPU:** Click "Add GPU"
     - **GPU type:** NVIDIA Tesla T4
     - **Number of GPUs:** 1
   - **Boot disk:**
     - **Operating system:** Ubuntu
     - **Version:** Ubuntu 22.04 LTS
     - **Boot disk type:** Standard persistent disk
     - **Size:** 100 GB
   - **Firewall:** Allow HTTP/HTTPS (optional)
4. Click **Management** tab:
   - Under **Metadata**, add:
     - **Key:** `install-nvidia-driver`
     - **Value:** `True`
5. Click **Create**

**Note:** Instance creation takes 2-3 minutes. The NVIDIA driver installation happens automatically on first boot and takes an additional 5-10 minutes.

---

## Step 2: Connect via SSH

```bash
# SSH into the instance
gcloud compute ssh examiner-aristotle-gpu --zone=us-central1-a

# If NVIDIA drivers are still installing, wait and check with:
sudo journalctl -u google-startup-scripts.service -f
```

---

## Step 3: Install NVIDIA Drivers and CUDA

The `install-nvidia-driver=True` metadata should automatically install drivers. Verify:

```bash
# Check if NVIDIA driver is installed
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P0    26W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Manual NVIDIA Driver Installation (if needed)

If `nvidia-smi` is not found:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install NVIDIA driver
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install

# Reboot (REQUIRED after driver installation)
sudo reboot

# After reboot, reconnect and verify
gcloud compute ssh examiner-aristotle-gpu --zone=us-central1-a
nvidia-smi
```

### Install CUDA Toolkit 12.3

```bash
# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
```

---

## Step 4: Setup Python and Dependencies

### Automated Setup (Recommended)

Use the provided setup script:

```bash
# Download and run the setup script
cd ~
git clone https://github.com/humanaiconvention/examiner.git
cd examiner
chmod +x gcp_setup.sh
bash gcp_setup.sh
```

The script will:
- Install Python 3.11
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Verify GPU availability

### Manual Setup (Alternative)

```bash
# Install Python 3.11
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl

# Clone repository
cd ~
git clone https://github.com/humanaiconvention/examiner.git
cd examiner

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements-gcp.txt

# Install additional dependencies
pip install aiohttp requests

# Verify GPU in Python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
PyTorch: 2.x.x+cu121
CUDA Available: True
GPU: Tesla T4
```

---

## Step 5: Configure Aristotle API Key

```bash
# Set API key for current session
export ARISTOTLE_API_KEY='your-api-key-here'

# Add to .bashrc for persistence
echo "export ARISTOTLE_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc

# Verify API key is set
echo $ARISTOTLE_API_KEY
```

**Security Note:** Never commit API keys to repositories. Use environment variables.

---

## Step 6: Verify GPU in Python

Create a test script to verify GPU access:

```bash
# Test GPU availability
python3 << EOF
import torch
print("=" * 50)
print("GPU Verification")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 50)
EOF
```

---

## Step 7: Run the Verifier

### Quick Start

```bash
# Activate virtual environment (if not already active)
cd ~/examiner
source .venv/bin/activate

# Run the automated verification script
chmod +x run_gcp_verification.sh
bash run_gcp_verification.sh
```

### Manual Execution

```bash
# Ensure you're in the examiner directory
cd ~/examiner
source .venv/bin/activate

# Check for failed samples
ls -lh evaluation_results/failed_samples.json

# Run verification (with sample limit for testing)
python aristotle_verifier.py \
  --failed-samples evaluation_results/failed_samples.json \
  --output-dir verification_results \
  --max-samples 10

# Run full verification (all samples)
python aristotle_verifier.py \
  --failed-samples evaluation_results/failed_samples.json \
  --output-dir verification_results
```

### Example Output

```
======================================================================
ARISTOTLE FORMAL VERIFICATION PIPELINE
======================================================================
Failed samples: evaluation_results/failed_samples.json
API base: https://api.aristotle.ai/v1
Output directory: verification_results
======================================================================

[LOADED] 50 failed samples

[STEP 1/3] Running formal verification...
[ARISTOTLE] Verifying 50 failed samples...
Progress: 100%|██████████| 50/50 [05:23<00:00,  6.47s/sample]

[VERIFICATION RESULTS]
  Total samples: 50
  Verified: 42
  Errors: 8
  Saved to: verification_results/verification_results.json

[STEP 2/3] Extracting Lean 4 proof traces...
[EXTRACTED] 42 valid proof traces

[STEP 3/3] Formatting as instruction tuning data...
[TRAINING DATA] Generated 126 samples
[SAVED] verification_results/formal_verification_training.json

======================================================================
VERIFICATION COMPLETE
======================================================================
Verification results: verification_results/verification_results.json
Training data: verification_results/formal_verification_training.json
Training samples: 126

Next: Merge with existing training data and run next training cycle
  python merge_training_data.py
  .venv/bin/python3 train_consolidated.py
======================================================================
```

---

## Step 8: Monitor GPU Usage

Monitor GPU utilization during verification:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use continuous monitoring
nvidia-smi dmon -s u
```

**Key metrics:**
- **GPU Utilization:** Should be high (80-100%) during active verification
- **Memory Usage:** Monitor to avoid OOM errors
- **Temperature:** Should stay below 80°C
- **Power Usage:** T4 max is 70W

---

## Step 9: Transfer Results

### Option A: Using SCP (Google Cloud SDK)

From your **local machine**:

```bash
# Download verification results
gcloud compute scp examiner-aristotle-gpu:~/examiner/verification_results/*.json ./local_results/ --zone=us-central1-a --recurse

# Download specific file
gcloud compute scp examiner-aristotle-gpu:~/examiner/verification_results/verification_results.json ./ --zone=us-central1-a
```

### Option B: Using Google Cloud Storage

From the **GCP instance**:

```bash
# Install gsutil (if not already installed)
sudo apt-get install -y google-cloud-sdk

# Upload to Cloud Storage
gsutil -m cp -r verification_results/ gs://your-bucket-name/examiner-results/

# From local machine, download
gsutil -m cp -r gs://your-bucket-name/examiner-results/ ./local_results/
```

### Option C: Direct SCP (if SSH keys configured)

```bash
# From local machine
scp -r examiner-aristotle-gpu:~/examiner/verification_results ./local_results/
```

---

## Cost Management

### Stop Instance (Retain Disk, Save Costs)

```bash
# From local machine
gcloud compute instances stop examiner-aristotle-gpu --zone=us-central1-a
```

**Cost savings:** Stops compute and GPU charges (~$0.54/hour), but keeps disk (~$0.04/GB/month = ~$4/month for 100GB).

### Start Instance Again

```bash
gcloud compute instances start examiner-aristotle-gpu --zone=us-central1-a
gcloud compute ssh examiner-aristotle-gpu --zone=us-central1-a
```

### Delete Instance (Permanently Remove)

```bash
# WARNING: This deletes all data on the instance
gcloud compute instances delete examiner-aristotle-gpu --zone=us-central1-a
```

### Monitor Billing

1. Go to **GCP Console** > **Billing** > **Reports**
2. Filter by:
   - **Service:** Compute Engine
   - **SKU:** GPU, VM instances
3. Set up **Budget Alerts**:
   - **Billing** > **Budgets & Alerts**
   - Set threshold (e.g., $50/month)
   - Configure email notifications

---

## Estimated Costs

### T4 GPU Instance (n1-standard-4 + T4)

**Hourly costs (us-central1):**
- `n1-standard-4` VM: ~$0.19/hour
- NVIDIA T4 GPU: ~$0.35/hour
- **Total:** ~$0.54/hour

**Daily costs (24 hours):**
- ~$12.96/day

**Monthly costs (730 hours):**
- ~$394/month (if running continuously)

**Storage costs:**
- 100 GB standard disk: ~$4/month (persistent even when stopped)

### Cost Optimization Tips

1. **Stop when not in use:** Reduces cost to only disk storage (~$4/month)
2. **Use preemptible instances:** 60-91% cheaper, but can be terminated
3. **Batch processing:** Process all samples in one session
4. **Delete when done:** Zero cost if you don't need the instance anymore
5. **Use committed use discounts:** Save up to 57% for long-term usage

### Example: Processing 100 samples

- **Estimated time:** ~15-20 minutes
- **Cost:** ~$0.15-$0.20 per run
- **Recommendation:** Batch multiple runs together

---

## Troubleshooting

### Issue: CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
```bash
# 1. Reduce batch size in verification
# Edit aristotle_verifier.py or use smaller max-samples

# 2. Clear GPU cache
python3 << EOF
import torch
torch.cuda.empty_cache()
EOF

# 3. Monitor memory before running
nvidia-smi

# 4. Use a larger GPU (V100, A100) if budget allows
```

### Issue: API Rate Limiting

**Symptoms:**
```
[ERROR] Aristotle API request failed: 429 Too Many Requests
```

**Solutions:**
- Add delays between requests (modify `aristotle_verifier.py`)
- Contact Aristotle support for rate limit increase
- Process in smaller batches with `--max-samples`

### Issue: Module Not Found Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'aiohttp'
```

**Solutions:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements-gcp.txt
pip install aiohttp requests
```

### Issue: NVIDIA Driver Not Found

**Symptoms:**
```
nvidia-smi: command not found
```

**Solutions:**
```bash
# Check if driver installation is still running
sudo journalctl -u google-startup-scripts.service -f

# Manual installation
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

### Issue: PyTorch Not Detecting GPU

**Symptoms:**
```python
torch.cuda.is_available() # Returns False
```

**Solutions:**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA version compatibility
python3 -c "import torch; print(torch.version.cuda)"
nvidia-smi  # Check NVIDIA driver CUDA version
```

### Issue: SSH Connection Timeout

**Symptoms:**
```
Connection timed out
```

**Solutions:**
```bash
# Check instance status
gcloud compute instances list

# Ensure instance is running
gcloud compute instances start examiner-aristotle-gpu --zone=us-central1-a

# Check firewall rules
gcloud compute firewall-rules list
```

### Issue: Insufficient GPU Quota

**Symptoms:**
```
Quota 'NVIDIA_T4_GPUS' exceeded
```

**Solutions:**
1. Go to **IAM & Admin** > **Quotas**
2. Search for "GPU"
3. Select region and GPU type
4. Click "Edit Quotas"
5. Request increase (usually approved within 24-48 hours)

---

## Automation

The `run_gcp_verification.sh` script automates the entire verification process:

```bash
cd ~/examiner
source .venv/bin/activate
bash run_gcp_verification.sh
```

**Features:**
- ✅ Checks GCP environment
- ✅ Verifies GPU availability
- ✅ Validates API key
- ✅ Counts samples and estimates cost
- ✅ Prompts for confirmation
- ✅ Runs verification
- ✅ Tracks execution time
- ✅ Displays summary
- ✅ Reminds to stop instance

---

## Additional Resources

- **GCP Documentation:** https://cloud.google.com/compute/docs/gpus
- **Aristotle API Docs:** https://docs.aristotle.ai/
- **PyTorch CUDA Guide:** https://pytorch.org/get-started/locally/
- **NVIDIA CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit

---

## Quick Reference Commands

```bash
# Create instance
gcloud compute instances create examiner-aristotle-gpu --zone=us-central1-a --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1 --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=100GB --maintenance-policy=TERMINATE --metadata=install-nvidia-driver=True

# SSH
gcloud compute ssh examiner-aristotle-gpu --zone=us-central1-a

# Setup
bash gcp_setup.sh
export ARISTOTLE_API_KEY='your-key'

# Run verification
cd examiner && source .venv/bin/activate
bash run_gcp_verification.sh

# Monitor
watch -n 1 nvidia-smi

# Transfer results
gcloud compute scp examiner-aristotle-gpu:~/examiner/verification_results/*.json ./ --zone=us-central1-a --recurse

# Stop instance
gcloud compute instances stop examiner-aristotle-gpu --zone=us-central1-a

# Start instance
gcloud compute instances start examiner-aristotle-gpu --zone=us-central1-a

# Delete instance
gcloud compute instances delete examiner-aristotle-gpu --zone=us-central1-a
```

---

**Note:** Always remember to stop or delete your GCP instance when not in use to avoid unexpected charges!

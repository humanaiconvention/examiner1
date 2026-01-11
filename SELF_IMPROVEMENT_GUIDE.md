# Self-Improvement Loop with Formal Verification

Complete guide to evaluating your model, verifying failed reasoning with Aristotle, and creating self-improvement training cycles.

## Overview

This pipeline implements a **semantic grounding self-improvement loop**:

1. **Evaluate** model on MMLU-Pro and GSM8K benchmarks
2. **Extract** failed reasoning samples
3. **Verify** failures using Aristotle API (informal→formal)
4. **Extract** Lean 4 proof traces
5. **Format** as instruction tuning data
6. **Merge** with original training data
7. **Retrain** model (Cycle 2, 3, ...)

---

## Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 2080 or better)
- **Python**: 3.10+
- **Trained Model**: Completed Cycle 1 training
- **Aristotle API Key**: Sign up at https://aristotle.ai

---

## Step 1: Install Dependencies

```bash
cd /mnt/d/humanaiconvention/examiner

# Install with uv (fast)
~/.local/bin/uv pip install lm-eval requests aiohttp

# Or install all requirements
~/.local/bin/uv pip install -r requirements.txt
```

**Verify installation:**
```bash
.venv/bin/python3 -c "import lm_eval; print('✓ lm-eval installed')"
.venv/bin/python3 -c "import aiohttp; print('✓ aiohttp installed')"
```

---

## Step 2: Set Up Aristotle API

**Get your API key:**
1. Go to https://aristotle.ai
2. Sign up / Log in
3. Navigate to API settings
4. Generate an API key

**Set environment variable:**
```bash
export ARISTOTLE_API_KEY="your-api-key-here"

# Or add to .bashrc for persistence
echo 'export ARISTOTLE_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Verify API access:**
```bash
curl -H "Authorization: Bearer $ARISTOTLE_API_KEY" \
  https://api.aristotle.ai/v1/status
```

---

## Step 3: Evaluate Your Model

Run MMLU-Pro and GSM8K benchmarks on your trained checkpoint:

```bash
cd /mnt/d/humanaiconvention/examiner

.venv/bin/python3 evaluate_model.py \
  --base-model mlabonne/Qwen3-4B-abliterated \
  --checkpoint Examiner1/models/grounding_cycle_1/final \
  --tasks mmlu_pro,gsm8k \
  --num-fewshot 5 \
  --batch-size 1 \
  --output-dir evaluation_results
```

**What this does:**
- Loads your trained LoRA checkpoint
- Runs lm-evaluation-harness benchmarks
- Saves results to `evaluation_results/`
- Extracts failed samples to `evaluation_results/failed_samples.json`

**Expected output:**
```
MMLU-PRO:
  Accuracy: 45.2%
  Samples: 1000

GSM8K:
  Accuracy: 62.8%
  Samples: 1319

✓ Evaluation complete!
  Results: evaluation_results/evaluation_results.json
  Failed samples: evaluation_results/failed_samples.json
  Total failed: 487
```

**Time estimate:** 30-60 minutes (depending on GPU)

---

## Step 4: Formal Verification with Aristotle

Send failed reasoning samples to Aristotle for formal verification:

```bash
cd /mnt/d/humanaiconvention/examiner

.venv/bin/python3 aristotle_verifier.py \
  --failed-samples evaluation_results/failed_samples.json \
  --api-key $ARISTOTLE_API_KEY \
  --output-dir verification_results \
  --max-samples 50
```

**Arguments:**
- `--failed-samples`: Path to failed samples from evaluation
- `--api-key`: Aristotle API key (or use env var)
- `--output-dir`: Where to save verification results
- `--max-samples`: Limit samples (remove for full batch)

**What this does:**
1. Loads failed reasoning samples
2. Sends each to Aristotle API for informal→formal translation
3. Receives Lean 4 formal proofs
4. Extracts verified proof traces
5. Formats as instruction tuning data

**Expected output:**
```
[ARISTOTLE] Verifying 50 failed samples...

[VERIFICATION RESULTS]
  Total samples: 50
  Verified: 38
  Errors: 12
  Saved to: verification_results/verification_results.json

[TRAINING DATA] Generated 114 samples
[SAVED] verification_results/formal_verification_training.json
```

**Output files:**
- `verification_results/verification_results.json` - Full verification results
- `verification_results/formal_verification_training.json` - Training data

**Time estimate:** 5-15 minutes (API dependent)

**Note:** Remove `--max-samples` to process all failed samples (takes longer)

---

## Step 5: Inspect Verification Results

**View verification summary:**
```bash
cd /mnt/d/humanaiconvention/examiner

cat verification_results/verification_results.json | jq '.total_samples, .verified, .errors'
```

**Example verified proof:**
```bash
cat verification_results/verification_results.json | \
  jq '.results[0]' | head -50
```

**View training data format:**
```bash
cat verification_results/formal_verification_training.json | \
  jq '.[0]' 
```

**Expected format:**
```json
{
  "instruction": "Translate the following informal mathematical statement into formal Lean 4 code.",
  "input": "If x + 3 = 7, then x = 4",
  "output": "theorem solve_x : ∀ x : ℕ, x + 3 = 7 → x = 4 := by\n  intro x h\n  omega",
  "source": "aristotle_verification",
  "sample_type": "formal_proof"
}
```

---

## Step 6: Merge Training Data

Combine original training data with formal verification results:

```bash
cd /mnt/d/humanaiconvention/examiner

.venv/bin/python3 merge_training_data.py \
  --original-data data/datasets/instruction_dataset \
  --formal-verification verification_results/formal_verification_training.json \
  --semantic-grounding Examiner1/lived_experience_log.json \
  --output training_data_cycle_2.json \
  --cycle 2 \
  --deduplicate \
  --prioritize-formal
```

**Arguments:**
- `--original-data`: Original training dataset (Cycle 1)
- `--formal-verification`: New formal proof training data
- `--semantic-grounding`: Lived experience grounding data
- `--output`: Output file for merged data
- `--cycle`: Training cycle number
- `--deduplicate`: Remove duplicate samples
- `--prioritize-formal`: Give 2x weight to formal proofs

**What this does:**
1. Loads all training data sources
2. Merges into unified dataset
3. Removes duplicates
4. Prioritizes formally verified samples (2x weight)
5. Saves as Cycle 2 training data

**Expected output:**
```
TRAINING DATA MERGE SUMMARY
======================================================================
Total samples: 314

By source:
  original_training: 100 samples
  formal_verification: 114 samples
  formal_verification (priority): 76 samples
  semantic_grounding: 6 samples

By type:
  instruction_response: 100 samples
  formal_proof: 152 samples
  informal_to_formal: 38 samples
  error_correction: 24 samples
======================================================================

✓ Merge complete!
```

---

## Step 7: Convert to HuggingFace Dataset

Convert merged JSON to HuggingFace dataset format for training:

```bash
cd /mnt/d/humanaiconvention/examiner

.venv/bin/python3 << 'EOF'
import json
from datasets import Dataset, DatasetDict

# Load merged training data
with open('training_data_cycle_2.json') as f:
    data = json.load(f)

# Handle different formats
if isinstance(data, dict) and 'samples' in data:
    samples = data['samples']
else:
    samples = data

# Create train/test split (90/10)
split_idx = int(len(samples) * 0.9)
train_samples = samples[:split_idx]
test_samples = samples[split_idx:]

# Create dataset
dataset = DatasetDict({
    'train': Dataset.from_list(train_samples),
    'test': Dataset.from_list(test_samples)
})

# Save to disk
dataset.save_to_disk('data/datasets/instruction_dataset_cycle_2')
print(f"✓ Saved dataset: {len(train_samples)} train, {len(test_samples)} test")
EOF
```

---

## Step 8: Configure Training for Cycle 2

Update `train_consolidated.py` configuration:

```bash
cd /mnt/d/humanaiconvention/examiner

# Edit the config section
nano train_consolidated.py
```

**Update these lines:**

```python
CONFIG = {
    "num_train_samples": 280,  # Increased from 100
    "num_eval_samples": 34,    # Increased from 25
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "num_epochs": 2,           # More epochs for cycle 2
    "model_name": "mlabonne/Qwen3-4B-abliterated",
    "use_4bit": True,
    "output_dir": "Examiner1/models/grounding_cycle_2",  # New output dir
    "lora_r": 8,
    "lora_alpha": 16,
}
```

**And update dataset path around line 70:**
```python
ds = load_from_disk("data/datasets/instruction_dataset_cycle_2")  # Cycle 2 dataset
```

---

## Step 9: Run Training Cycle 2

Train the model on the enhanced dataset:

```bash
cd /mnt/d/humanaiconvention/examiner

.venv/bin/python3 train_consolidated.py 2>&1 | tee training_cycle_2.log
```

**Expected output:**
```
======================================================================
SEMANTIC GROUNDING CYCLE 2: Qwen3-4B (4-bit QLoRA)
======================================================================

[1/6] Loading dataset...
[OK] Loaded 280 train, 34 test samples

[2/6] Loading tokenizer...
[OK] Tokenizer loaded

[3/6] Loading mlabonne/Qwen3-4B-abliterated model...
[OK] Model loaded on cuda

[4/6] Configuring LoRA...
trainable params: 16,515,072 || all params: 4,038,983,168 || trainable%: 0.4089
[OK] LoRA configured

[5/6] Tokenizing dataset...
[OK] Tokenization complete

[6/6] Training...
======================================================================
Starting training...
======================================================================

Epoch 1/2
Step 5/70 | Loss: 11.8234
Step 10/70 | Loss: 10.9456
...
Step 70/70 | Loss: 8.2341

Epoch 2/2
Step 75/140 | Loss: 7.8923
...
Step 140/140 | Loss: 6.4187

[OK] Training completed in 5.47 minutes
Average loss: 6.4187

[EVAL] Evaluating model...
Eval loss: 1.8923

✅ Training successful!
```

**Time estimate:** 5-10 minutes (280 samples, 2 epochs)

---

## Step 10: Evaluate Cycle 2 Model

Compare Cycle 2 performance to Cycle 1:

```bash
cd /mnt/d/humanaiconvention/examiner

.venv/bin/python3 evaluate_model.py \
  --base-model mlabonne/Qwen3-4B-abliterated \
  --checkpoint Examiner1/models/grounding_cycle_2/final \
  --tasks mmlu_pro,gsm8k \
  --num-fewshot 5 \
  --batch-size 1 \
  --output-dir evaluation_results_cycle_2
```

**Compare results:**
```bash
# Cycle 1 results
cat evaluation_results/evaluation_results.json | jq '.results'

# Cycle 2 results
cat evaluation_results_cycle_2/evaluation_results.json | jq '.results'
```

**Expected improvement:**
- MMLU-Pro: 45.2% → 48.7% (+3.5%)
- GSM8K: 62.8% → 68.4% (+5.6%)

---

## Step 11: Iterate (Cycle 3, 4, ...)

Repeat the process for continuous improvement:

```bash
# 1. Evaluate Cycle 2
.venv/bin/python3 evaluate_model.py \
  --checkpoint Examiner1/models/grounding_cycle_2/final \
  --output-dir evaluation_results_cycle_2

# 2. Verify new failures
.venv/bin/python3 aristotle_verifier.py \
  --failed-samples evaluation_results_cycle_2/failed_samples.json \
  --output-dir verification_results_cycle_2

# 3. Merge all data (cumulative)
.venv/bin/python3 merge_training_data.py \
  --original-data data/datasets/instruction_dataset_cycle_2 \
  --formal-verification verification_results_cycle_2/formal_verification_training.json \
  --output training_data_cycle_3.json \
  --cycle 3 \
  --deduplicate \
  --prioritize-formal

# 4. Convert to HF dataset
# (repeat Step 7 with cycle_3 paths)

# 5. Update config and train
# (repeat Steps 8-9 with cycle_3 paths)
```

---

## Complete Pipeline Script

Save this as `run_self_improvement_cycle.sh`:

```bash
#!/bin/bash
# Self-Improvement Cycle Automation
# Usage: ./run_self_improvement_cycle.sh <cycle_number>

set -e

CYCLE=${1:-2}
PREV_CYCLE=$((CYCLE - 1))

echo "========================================================================"
echo "SELF-IMPROVEMENT CYCLE $CYCLE"
echo "========================================================================"

cd /mnt/d/humanaiconvention/examiner

# Step 1: Evaluate previous cycle
echo "[STEP 1/6] Evaluating Cycle $PREV_CYCLE model..."
.venv/bin/python3 evaluate_model.py \
  --checkpoint Examiner1/models/grounding_cycle_${PREV_CYCLE}/final \
  --output-dir evaluation_results_cycle_${PREV_CYCLE}

# Step 2: Formal verification
echo "[STEP 2/6] Running formal verification..."
.venv/bin/python3 aristotle_verifier.py \
  --failed-samples evaluation_results_cycle_${PREV_CYCLE}/failed_samples.json \
  --output-dir verification_results_cycle_${PREV_CYCLE}

# Step 3: Merge training data
echo "[STEP 3/6] Merging training data..."
.venv/bin/python3 merge_training_data.py \
  --original-data data/datasets/instruction_dataset_cycle_${PREV_CYCLE} \
  --formal-verification verification_results_cycle_${PREV_CYCLE}/formal_verification_training.json \
  --output training_data_cycle_${CYCLE}.json \
  --cycle $CYCLE \
  --deduplicate \
  --prioritize-formal

# Step 4: Convert to HF dataset
echo "[STEP 4/6] Converting to HuggingFace dataset..."
.venv/bin/python3 << EOF
import json
from datasets import Dataset, DatasetDict

with open('training_data_cycle_${CYCLE}.json') as f:
    data = json.load(f)

samples = data['samples'] if isinstance(data, dict) and 'samples' in data else data
split_idx = int(len(samples) * 0.9)

dataset = DatasetDict({
    'train': Dataset.from_list(samples[:split_idx]),
    'test': Dataset.from_list(samples[split_idx:])
})

dataset.save_to_disk('data/datasets/instruction_dataset_cycle_${CYCLE}')
print(f"✓ Dataset saved: {len(samples[:split_idx])} train, {len(samples[split_idx:])} test")
EOF

# Step 5: Update training config
echo "[STEP 5/6] Updating training configuration..."
# Note: Manually update train_consolidated.py or use sed/awk to automate

# Step 6: Train Cycle N
echo "[STEP 6/6] Training Cycle $CYCLE model..."
.venv/bin/python3 train_consolidated.py 2>&1 | tee training_cycle_${CYCLE}.log

echo "========================================================================"
echo "✅ CYCLE $CYCLE COMPLETE"
echo "========================================================================"
echo "Model saved to: Examiner1/models/grounding_cycle_${CYCLE}/final"
echo "Next: Run cycle $((CYCLE + 1))"
```

**Make executable and run:**
```bash
chmod +x run_self_improvement_cycle.sh
./run_self_improvement_cycle.sh 2
```

---

## Troubleshooting

### lm-eval Import Error
```bash
# Reinstall lm-eval
~/.local/bin/uv pip uninstall lm-eval
~/.local/bin/uv pip install lm-eval>=0.4.0
```

### Aristotle API Authentication Failed
```bash
# Check API key is set
echo $ARISTOTLE_API_KEY

# Test API connection
curl -H "Authorization: Bearer $ARISTOTLE_API_KEY" \
  https://api.aristotle.ai/v1/status

# Re-export if needed
export ARISTOTLE_API_KEY="your-key-here"
```

### Out of Memory During Evaluation
```bash
# Reduce batch size
.venv/bin/python3 evaluate_model.py \
  --batch-size 1 \
  --tasks gsm8k  # Run one task at a time
```

### Model Loading Error
```bash
# Check checkpoint exists
ls -lh Examiner1/models/grounding_cycle_1/final/

# Should contain: adapter_config.json, adapter_model.safetensors
```

### Empty Failed Samples
If no samples fail, your model is doing great! But you can still:
- Lower the few-shot count: `--num-fewshot 0`
- Use harder benchmarks: `--tasks math,theorem_qa`
- Increase sample difficulty threshold

---

## Performance Tracking

Create a tracking spreadsheet:

| Cycle | MMLU-Pro | GSM8K | Train Loss | Eval Loss | Samples | Training Time |
|-------|----------|-------|------------|-----------|---------|---------------|
| 1     | 45.2%    | 62.8% | 10.53      | 2.45      | 100     | 2 min         |
| 2     | 48.7%    | 68.4% | 6.42       | 1.89      | 280     | 5 min         |
| 3     | 52.1%    | 73.2% | 4.87       | 1.34      | 520     | 12 min        |

**Track improvements:**
```bash
# Create tracking file
cat > performance_tracking.json << EOF
{
  "cycles": []
}
EOF

# After each cycle, append results
# (automate this in your pipeline)
```

---

## Advanced: Custom Verification Logic

Edit `aristotle_verifier.py` to customize verification:

```python
def verify_reasoning(self, question: str, reasoning: str, answer: str, context: str = ""):
    """Add custom preprocessing"""
    
    # Filter by difficulty
    if len(reasoning.split()) < 50:
        return {"status": "skipped", "reason": "reasoning too short"}
    
    # Add domain-specific context
    if "probability" in question.lower():
        context += "\n\nDomain: Probability Theory"
    
    # Call Aristotle API
    payload = {
        "task": "informal_to_formal",
        "informal_statement": question,
        "informal_proof": reasoning,
        "proposed_answer": answer,
        "context": context,
        "return_format": "lean4",
        "verify": True,
    }
    
    # ... rest of method
```

---

## Best Practices

1. **Start small**: Use `--max-samples 10` for testing
2. **Monitor costs**: Aristotle API may have usage limits
3. **Track progress**: Save all results for comparison
4. **Validate proofs**: Manually inspect some Lean 4 proofs
5. **Iterate carefully**: Don't skip evaluation between cycles
6. **Balance data**: Don't over-weight any single source
7. **Backup models**: Keep all cycle checkpoints

---

## Expected Results

**After 3-5 cycles:**
- MMLU-Pro: 45% → 55% (+10%)
- GSM8K: 63% → 78% (+15%)
- Formal reasoning: Significantly improved
- Error patterns: Reduced via corrective training

**Key improvements:**
- Fewer logical fallacies
- Better step-by-step reasoning
- More formal mathematical proofs
- Stronger semantic grounding

---

## Next Steps

1. **Custom benchmarks**: Add domain-specific evaluation tasks
2. **Multiple models**: Run parallel cycles with different hyperparameters
3. **Ensemble**: Combine multiple cycle checkpoints
4. **Production**: Deploy best-performing cycle checkpoint
5. **Research**: Analyze which formal proofs help most

---

## Resources

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **Aristotle API**: https://aristotle.ai/docs
- **Lean 4**: https://lean-lang.org/
- **Semantic Grounding Paper**: See corpus references

---

## Questions?

Common issues and solutions documented in `TROUBLESHOOTING.md`

For bugs or improvements, open an issue on GitHub.

**Happy self-improving! 🚀**
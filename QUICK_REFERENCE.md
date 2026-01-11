# Self-Improvement Loop - Quick Reference

## One-Command Setup

```bash
# Install dependencies
cd /mnt/d/humanaiconvention/examiner
~/.local/bin/uv pip install lm-eval requests aiohttp

# Set Aristotle API key
export ARISTOTLE_API_KEY="your-api-key-here"
```

---

## Complete Cycle Commands

### Cycle 2 (From Cycle 1 Model)

```bash
cd /mnt/d/humanaiconvention/examiner

# 1. Evaluate Cycle 1
.venv/bin/python3 evaluate_model.py \
  --checkpoint Examiner1/models/grounding_cycle_1/final \
  --output-dir evaluation_results

# 2. Verify failures with Aristotle
.venv/bin/python3 aristotle_verifier.py \
  --failed-samples evaluation_results/failed_samples.json \
  --output-dir verification_results

# 3. Merge training data
.venv/bin/python3 merge_training_data.py \
  --original-data data/datasets/instruction_dataset \
  --formal-verification verification_results/formal_verification_training.json \
  --output training_data_cycle_2.json \
  --cycle 2 \
  --deduplicate \
  --prioritize-formal

# 4. Convert to HF dataset
.venv/bin/python3 << 'EOF'
import json
from datasets import Dataset, DatasetDict
with open('training_data_cycle_2.json') as f:
    data = json.load(f)
samples = data['samples'] if 'samples' in data else data
split_idx = int(len(samples) * 0.9)
dataset = DatasetDict({
    'train': Dataset.from_list(samples[:split_idx]),
    'test': Dataset.from_list(samples[split_idx:])
})
dataset.save_to_disk('data/datasets/instruction_dataset_cycle_2')
print(f"✓ Saved: {len(samples[:split_idx])} train, {len(samples[split_idx:])} test")
EOF

# 5. Update train_consolidated.py:
#    - Change output_dir to: Examiner1/models/grounding_cycle_2
#    - Change dataset to: data/datasets/instruction_dataset_cycle_2
#    - Update num_train_samples to match new dataset size

# 6. Train Cycle 2
.venv/bin/python3 train_consolidated.py 2>&1 | tee training_cycle_2.log
```

---

## Individual Commands

### Evaluation Only
```bash
.venv/bin/python3 evaluate_model.py \
  --checkpoint Examiner1/models/grounding_cycle_1/final \
  --tasks mmlu_pro,gsm8k \
  --output-dir evaluation_results
```

### Verification Only (Test with 10 samples)
```bash
.venv/bin/python3 aristotle_verifier.py \
  --failed-samples evaluation_results/failed_samples.json \
  --max-samples 10 \
  --output-dir verification_results
```

### Merge Only
```bash
.venv/bin/python3 merge_training_data.py \
  --original-data data/datasets/instruction_dataset \
  --formal-verification verification_results/formal_verification_training.json \
  --output training_data_cycle_2.json \
  --cycle 2 \
  --deduplicate
```

---

## Quick Checks

### Check Evaluation Results
```bash
cat evaluation_results/evaluation_results.json | jq '.results'
```

### Count Failed Samples
```bash
cat evaluation_results/failed_samples.json | jq 'length'
```

### View First Failed Sample
```bash
cat evaluation_results/failed_samples.json | jq '.[0]'
```

### Check Verification Status
```bash
cat verification_results/verification_results.json | \
  jq '{total: .total_samples, verified: .verified, errors: .errors}'
```

### Count Training Samples
```bash
cat training_data_cycle_2.json | jq '.samples | length'
# OR
cat training_data_cycle_2.json | jq 'length'
```

---

## Troubleshooting

### Test Aristotle API
```bash
curl -H "Authorization: Bearer $ARISTOTLE_API_KEY" \
  https://api.aristotle.ai/v1/status
```

### Check GPU Memory
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Verify Model Checkpoint Exists
```bash
ls -lh Examiner1/models/grounding_cycle_1/final/
# Should see: adapter_config.json, adapter_model.safetensors
```

### Check Dataset Format
```bash
.venv/bin/python3 << 'EOF'
from datasets import load_from_disk
ds = load_from_disk('data/datasets/instruction_dataset')
print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")
print(f"Columns: {ds['train'].column_names}")
EOF
```

---

## Performance Comparison

```bash
# Compare Cycle 1 vs Cycle 2
echo "=== CYCLE 1 ==="
cat evaluation_results/evaluation_results.json | jq '.results'

echo "=== CYCLE 2 ==="
cat evaluation_results_cycle_2/evaluation_results.json | jq '.results'
```

---

## File Structure

```
examiner/
├── evaluate_model.py              # Step 1: Evaluation
├── aristotle_verifier.py          # Step 2: Formal verification
├── merge_training_data.py         # Step 3: Data merging
├── train_consolidated.py          # Step 4: Training
├── evaluation_results/
│   ├── evaluation_results.json
│   └── failed_samples.json
├── verification_results/
│   ├── verification_results.json
│   └── formal_verification_training.json
├── training_data_cycle_2.json
└── data/datasets/
    ├── instruction_dataset          # Cycle 1
    └── instruction_dataset_cycle_2  # Cycle 2
```

---

## Time Estimates

| Step | Time (approx) |
|------|---------------|
| Evaluation | 30-60 min |
| Verification (50 samples) | 5-10 min |
| Verification (all samples) | 15-30 min |
| Data merging | < 1 min |
| Training Cycle 2 | 5-10 min |
| **Total** | **1-2 hours** |

---

## Best Practices

1. **Test first**: Use `--max-samples 10` for verification testing
2. **Save everything**: Don't overwrite previous cycle results
3. **Check results**: Inspect samples before merging
4. **Backup models**: Keep all cycle checkpoints
5. **Track metrics**: Log performance for each cycle

---

## Environment Variables

```bash
# Required for Aristotle
export ARISTOTLE_API_KEY="sk-..."

# Optional: Python unbuffered output
export PYTHONUNBUFFERED=1

# Optional: CUDA device selection
export CUDA_VISIBLE_DEVICES=0
```

---

## Common Patterns

### Run Multiple Benchmarks Separately
```bash
# MMLU-Pro only
.venv/bin/python3 evaluate_model.py --tasks mmlu_pro --output-dir eval_mmlu

# GSM8K only  
.venv/bin/python3 evaluate_model.py --tasks gsm8k --output-dir eval_gsm8k
```

### Process Verification in Batches
```bash
# First 50
.venv/bin/python3 aristotle_verifier.py --max-samples 50 --output-dir verify_batch1

# Next 50
.venv/bin/python3 aristotle_verifier.py --max-samples 100 --output-dir verify_batch2
```

### Merge Multiple Verification Results
```bash
.venv/bin/python3 merge_training_data.py \
  --original-data data/datasets/instruction_dataset \
  --formal-verification verify_batch1/formal_verification_training.json \
  --output training_data_cycle_2.json
  
# Manually merge batch2 afterward or combine JSONs first
```

---

## Output Formats

### Evaluation Results
```json
{
  "timestamp": "2025-01-10T...",
  "model": "Examiner1/models/grounding_cycle_1/final",
  "results": {
    "mmlu_pro": {
      "accuracy": 0.452,
      "num_samples": 1000
    },
    "gsm8k": {
      "accuracy": 0.628,
      "num_samples": 1319
    }
  }
}
```

### Failed Sample
```json
{
  "task": "gsm8k",
  "question": "If x + 3 = 7, what is x?",
  "model_answer": "x = 5",
  "correct_answer": "x = 4",
  "reasoning_trace": "Adding 3 to both sides..."
}
```

### Training Data
```json
{
  "instruction": "Translate to Lean 4",
  "input": "x + 3 = 7 implies x = 4",
  "output": "theorem solve_x : ...",
  "source": "aristotle_verification",
  "sample_type": "formal_proof"
}
```

---

## Quick Wins

### Fast Test Run (5 minutes)
```bash
# Evaluate on small subset
.venv/bin/python3 evaluate_model.py --tasks gsm8k --output-dir test_eval

# Verify just 5 samples
.venv/bin/python3 aristotle_verifier.py --max-samples 5 --output-dir test_verify

# Merge and train on small dataset
.venv/bin/python3 merge_training_data.py --output test_training.json --max-per-source 20
```

### Full Production Run (2 hours)
```bash
# Everything at once
./run_self_improvement_cycle.sh 2
```

---

## Success Indicators

✅ **Evaluation**: Failed samples < 500  
✅ **Verification**: Verified > 70% of failed samples  
✅ **Training**: Loss decreasing consistently  
✅ **Performance**: Accuracy improvement 3-5% per cycle  

---

## Getting Help

See detailed guide: `SELF_IMPROVEMENT_GUIDE.md`  
Troubleshooting: `TROUBLESHOOTING.md` (if exists)  
Issues: GitHub issues page
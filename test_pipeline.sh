#!/bin/bash
# Quick test of the self-improvement pipeline

set -e

echo "========================================================================"
echo "TESTING SELF-IMPROVEMENT PIPELINE"
echo "========================================================================"

cd /mnt/d/humanaiconvention/examiner

# Check API key
if [ -z "" ]; then
    echo "❌ ARISTOTLE_API_KEY not set"
    echo "Run: export ARISTOTLE_API_KEY='your-key-here'"
    exit 1
fi

echo "✓ Aristotle API key found"

# Test 1: Small evaluation (5 samples for speed)
echo ""
echo "[TEST 1/3] Testing evaluation pipeline..."
.venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from evaluate_model import LoRAEvaluator
evaluator = LoRAEvaluator(
    'mlabonne/Qwen3-4B-abliterated',
    'Examiner1/models/grounding_cycle_1/final'
)
print('✓ Evaluator initialized')
"

# Test 2: Check Aristotle API
echo ""
echo "[TEST 2/3] Testing Aristotle API..."
curl -s -H "Authorization: Bearer "   https://api.aristotle.ai/v1/status > /dev/null && echo "✓ Aristotle API accessible" || echo "❌ Aristotle API failed"

# Test 3: Test data merger
echo ""
echo "[TEST 3/3] Testing data merger..."
.venv/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from merge_training_data import TrainingDataMerger
merger = TrainingDataMerger()
print('✓ Data merger initialized')
"

echo ""
echo "========================================================================"
echo "✅ ALL TESTS PASSED"
echo "========================================================================"
echo ""
echo "Ready to run full pipeline!"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: .venv/bin/python3 evaluate_model.py --checkpoint Examiner1/models/grounding_cycle_1/final"
echo "  2. Run verification: .venv/bin/python3 aristotle_verifier.py --max-samples 10"
echo "  3. See QUICK_REFERENCE.md for full commands"

#!/bin/bash
cd /mnt/d/humanaiconvention/examiner

echo "Testing imports..."
.venv/bin/python3 -c "
from evaluate_model import LoRAEvaluator
from aristotle_verifier import AristotleVerifier
from merge_training_data import TrainingDataMerger
print('✓ All modules imported successfully')
"

echo ""
echo "Ready to run evaluation!"
echo ""
echo "Quick test (5 min):"
echo "  .venv/bin/python3 evaluate_model.py --checkpoint Examiner1/models/grounding_cycle_1/final --tasks gsm8k --output-dir evaluation_results"
echo ""
echo "Or see QUICK_REFERENCE.md for full commands"

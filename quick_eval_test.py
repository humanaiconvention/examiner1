#!/usr/bin/env python3
"""Quick 3-sample evaluation and Aristotle test"""
import json
from pathlib import Path

# Create 3 realistic failed samples for testing
failed_samples = [
    {
        "task": "gsm8k",
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for  per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "model_answer": "6",
        "correct_answer": "8",
        "reasoning_trace": "Janet gets 16 eggs per day. She eats 3 and uses 4 for muffins, so she uses 7 eggs. That leaves 16 - 7 = 9 eggs. Wait, let me recalculate: she has 16 eggs, uses 3 + 4 = 7, so 16 - 7 = 9. She sells them for  each, so 9 ×  = 8... Actually I calculated 6 somehow.",
        "context": "Word problem with arithmetic operations",
        "timestamp": "2025-01-10T19:00:00"
    },
    {
        "task": "gsm8k", 
        "question": "A store sells apples for  per pound and oranges for  per pound. If you buy 4 pounds of apples and 5 pounds of oranges, how much do you spend in total?",
        "model_answer": "3",
        "correct_answer": "2",
        "reasoning_trace": "Apples cost  per pound, and I'm buying 4 pounds, so that's 3 × 4 = 2. Oranges cost  per pound, and I'm buying 5 pounds, so that's 2 × 5 = 0. Total is 2 + 0 = 2. But I answered 3 which is incorrect.",
        "context": "Basic multiplication and addition",
        "timestamp": "2025-01-10T19:01:00"
    },
    {
        "task": "gsm8k",
        "question": "If a rectangle has a length of 8 meters and a width of 5 meters, what is its perimeter?",
        "model_answer": "40 meters",
        "correct_answer": "26 meters", 
        "reasoning_trace": "Perimeter of a rectangle is 2 × (length + width). So it's 2 × (8 + 5) = 2 × 13 = 26 meters. I incorrectly calculated 8 × 5 = 40, confusing perimeter with area.",
        "context": "Geometry - rectangle perimeter formula",
        "timestamp": "2025-01-10T19:02:00"
    }
]

# Save to file
output_dir = Path("evaluation_results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "failed_samples_quick_test.json"
with open(output_file, 'w') as f:
    json.dump(failed_samples, f, indent=2)

print("✓ Created 3 test failed samples")
print(f"✓ Saved to: {output_file}")
print(f"\nSamples:")
for i, sample in enumerate(failed_samples, 1):
    print(f"  {i}. {sample['question'][:60]}...")
    print(f"     Model: {sample['model_answer']} | Correct: {sample['correct_answer']}")

print("\nNext: Run Aristotle verification")
print(f"  .venv/bin/python3 aristotle_verifier.py --failed-samples {output_file} --max-samples 3 --output-dir verification_results")

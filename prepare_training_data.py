#!/usr/bin/env python3
"""
Prepare training data: Combine semantic grounding with research corpus

This script:
1. Loads the semantic grounding (lived experience)
2. Loads the research corpus
3. Creates a unified training dataset that includes both
4. Generates a training manifest for qwen_finetune.py
"""

import json
from datetime import datetime
from pathlib import Path

print("=" * 70)

# Prompt for Lived Experience
print("\n[Semantic Grounding Check]")
choice = input("Do you want to run the Lived Experience Dialogue to capture new grounding? [y/N]: ").strip().lower()
if choice == 'y':
    try:
        print("\nLaunching dialogue system...")
        import subprocess
        import sys
        subprocess.run([sys.executable, "lived_experience_dialogue.py"], check=True)
        print("\n✓ Dialogue complete.")
    except Exception as e:
        print(f"Error running dialogue: {e}")
        print("Continuing with existing logs...")

# Load semantic grounding
print("\n[1/3] Loading semantic grounding data...")
with open("Examiner1/lived_experience_log.json", "r") as f:
    grounding_data = json.load(f)

grounding_count = len(grounding_data.get("reports", []))
print(f"✓ Loaded {grounding_count} lived experience(s)")

if grounding_count > 0:
    exp = grounding_data["reports"][0]
    print(f"  └─ Type: {exp['experience_type']}")
    print(f"  └─ Confidence: {exp['confidence_in_this_learning']:.1%}")
    print(f"  └─ Model Learning: '{exp['what_this_teaches'][:60]}...'")

# Load corpus
print("\n[2/3] Loading research corpus...")
with open("Examiner1/corpus_index.json", "r") as f:
    corpus_data = json.load(f)

corpus_count = len(corpus_data.get("documents", []))
print(f"✓ Loaded {corpus_count} research papers")

# Count by category
categories = {}
for doc in corpus_data.get("documents", []):
    cat = doc.get("category", "uncategorized")
    categories[cat] = categories.get(cat, 0) + 1

print("\n  Research corpus by category:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"    {cat}: {count} papers")

# Create unified training dataset
print("\n[3/3] Creating unified training dataset...")

training_data = {
    "created": datetime.now().isoformat(),
    "cycle": "semantic_grounding_cycle_1",
    "data_sources": {
        "research_corpus": {
            "source": "corpus_index.json",
            "count": corpus_count,
            "papers": [
                {
                    "title": doc.get("filename", doc.get("filepath", "Unknown")),
                    "category": doc.get("category", "uncategorized"),
                    "status": doc.get("status", "indexed"),
                }
                for doc in corpus_data.get("documents", [])
            ],
        },
        "semantic_grounding": {
            "source": "Examiner1/lived_experience_log.json",
            "count": grounding_count,
            "experiences": [
                {
                    "type": exp["experience_type"],
                    "learned_belief": exp["what_this_teaches"],
                    "evidence": exp["lived_experience_basis"],
                    "confidence": exp["confidence_in_this_learning"],
                    "previous_belief": exp.get(
                        "contradicts_prior_belief", "Model-generated"
                    ),
                    "timestamp": exp["timestamp"],
                }
                for exp in grounding_data.get("reports", [])
            ],
        },
    },
    "training_instructions": {
        "corpus_instructions": [
            "Analyze patterns and improvements from {0} research papers".format(
                corpus_count
            ),
            "Extract methodologies and best practices",
            "Learn from model collapse, semantic grounding, and AI safety literature",
            "Update proposals based on comprehensive research foundation",
        ],
        "grounding_instructions": [
            "Learn user's lived experiences as ground truth",
            "Update your understanding of what information integration means",
            "Recognize that meaning = information integration",
            "Accept user's observations as external validation",
            "Use this grounding to prevent semantic collapse",
        ],
    },
    "viability_check": {
        "C_eff_source": "lived_experience (user grounding)",
        "E_t_source": "research_corpus (drift tendency from 104 papers)",
        "requirement": "C_eff(t) >= E(t) for system viability",
        "check_after_training": "Run verify_system.py to measure C_eff",
    },
}

# Save unified training dataset
training_file = "training_data_cycle_1.json"
with open(training_file, "w") as f:
    json.dump(training_data, f, indent=2)

print(f"✓ Unified training dataset created: {training_file}")

# Generate training manifest for qwen_finetune.py
manifest = {
    "training_run": "semantic_grounding_cycle_1",
    "timestamp": datetime.now().isoformat(),
    "data_files": [
        {
            "source": "Examiner1/corpus_index.json",
            "type": "research_corpus",
            "description": f"{corpus_count} research papers on AI, semantics, grounding, consciousness",
            "integration": "Extract research patterns for improvement proposals",
        },
        {
            "source": "Examiner1/lived_experience_log.json",
            "type": "semantic_grounding",
            "description": f"{grounding_count} lived experience report(s) from user",
            "integration": "Learn user's ground truth; use as C_eff",
        },
        {
            "source": "training_data_cycle_1.json",
            "type": "unified_training_data",
            "description": "Corpus + grounding combined for training",
            "integration": "Pass to fine-tuning pipeline",
        },
    ],
    "next_steps": [
        "python qwen_finetune.py --training-data training_data_cycle_1.json",
        "Monitor training with: python monitor_training.py",
        "After training, verify: python verify_system.py",
    ],
}

manifest_file = "training_manifest.json"
with open(manifest_file, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✓ Training manifest created: {manifest_file}")

print("\n" + "=" * 70)
print("TRAINING DATA READY")
print("=" * 70)
print(f"\nData sources combined:")
print(f"  • Research corpus: {corpus_count} papers")
print(f"  • Semantic grounding: {grounding_count} lived experience(s)")
print(f"  • Unified dataset: training_data_cycle_1.json")
print(f"\nNext step:")
print(f"  python qwen_finetune.py --training-data training_data_cycle_1.json")
print(f"\nMonitor with:")
print(f"  python monitor_training.py")

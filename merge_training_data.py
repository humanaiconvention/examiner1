#!/usr/bin/env python3
"""
Training Data Merger for Self-Improvement Cycles
Combines original training data with formal verification results
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class TrainingDataMerger:
    """Merge multiple training data sources for self-improvement"""

    def __init__(self):
        self.merged_data = []
        self.sources = []
        self.stats = {
            "total_samples": 0,
            "by_source": {},
            "by_type": {},
        }

    def load_training_data(self, file_path: str, source_name: str = None) -> List[Dict]:
        """
        Load training data from JSON file

        Args:
            file_path: Path to training data JSON
            source_name: Name to identify this data source

        Returns:
            list: Loaded training samples
        """
        path = Path(file_path)
        if not path.exists():
            print(f"[WARNING] File not found: {file_path}")
            return []

        with open(path) as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        elif isinstance(data, dict) and "data" in data:
            samples = data["data"]
        else:
            print(f"[WARNING] Unknown format in {file_path}")
            return []

        # Tag with source
        source = source_name or path.stem
        for sample in samples:
            if "source" not in sample:
                sample["source"] = source
            if "loaded_from" not in sample:
                sample["loaded_from"] = str(path)

        print(f"[LOADED] {len(samples)} samples from {file_path}")
        self.sources.append(source)
        return samples

    def add_samples(self, samples: List[Dict]):
        """Add samples to merged dataset"""
        self.merged_data.extend(samples)

        # Update stats
        for sample in samples:
            source = sample.get("source", "unknown")
            sample_type = sample.get("sample_type", "unknown")

            self.stats["by_source"][source] = self.stats["by_source"].get(source, 0) + 1
            self.stats["by_type"][sample_type] = (
                self.stats["by_type"].get(sample_type, 0) + 1
            )

        self.stats["total_samples"] = len(self.merged_data)

    def deduplicate(self, key: str = "instruction"):
        """
        Remove duplicate samples based on key field

        Args:
            key: Field to use for deduplication (instruction, input, output)
        """
        seen = set()
        unique_samples = []

        for sample in self.merged_data:
            identifier = sample.get(key, "")
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_samples.append(sample)

        removed = len(self.merged_data) - len(unique_samples)
        if removed > 0:
            print(f"[DEDUP] Removed {removed} duplicate samples")

        self.merged_data = unique_samples
        self.stats["total_samples"] = len(self.merged_data)

    def balance_sources(self, max_per_source: int = None, min_per_source: int = None):
        """
        Balance samples across sources

        Args:
            max_per_source: Maximum samples per source
            min_per_source: Minimum samples per source
        """
        if not max_per_source and not min_per_source:
            return

        from collections import defaultdict

        samples_by_source = defaultdict(list)

        for sample in self.merged_data:
            source = sample.get("source", "unknown")
            samples_by_source[source].append(sample)

        balanced_data = []

        for source, samples in samples_by_source.items():
            count = len(samples)

            if max_per_source and count > max_per_source:
                # Randomly sample
                import random

                samples = random.sample(samples, max_per_source)
                print(
                    f"[BALANCE] {source}: Limited from {count} to {max_per_source} samples"
                )
            elif min_per_source and count < min_per_source:
                print(
                    f"[WARNING] {source}: Only {count} samples (min: {min_per_source})"
                )

            balanced_data.extend(samples)

        self.merged_data = balanced_data
        self.stats["total_samples"] = len(self.merged_data)

    def prioritize_formal_proofs(self):
        """
        Prioritize formally verified samples by duplicating them
        This gives more weight to formally verified reasoning
        """
        formal_samples = [
            s for s in self.merged_data if s.get("sample_type") == "formal_proof"
        ]

        if formal_samples:
            # Add formal samples again (2x weight)
            self.merged_data.extend(formal_samples)
            self.stats["total_samples"] = len(self.merged_data)
            print(f"[PRIORITY] Doubled {len(formal_samples)} formal proof samples")

    def create_cycle_metadata(self, cycle_number: int) -> Dict:
        """Create metadata for this training cycle"""
        return {
            "cycle": cycle_number,
            "timestamp": datetime.now().isoformat(),
            "sources": list(set(self.sources)),
            "stats": self.stats,
            "description": f"Self-improvement cycle {cycle_number} with formal verification",
        }

    def save_merged_data(
        self, output_file: str, cycle_number: int = 2, format: str = "huggingface"
    ):
        """
        Save merged training data

        Args:
            output_file: Path to save merged data
            cycle_number: Training cycle number
            format: Output format (huggingface, alpaca, raw)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "huggingface":
            # Format for Hugging Face datasets
            output_data = {
                "metadata": self.create_cycle_metadata(cycle_number),
                "samples": self.merged_data,
            }
        elif format == "alpaca":
            # Alpaca format
            output_data = [
                {
                    "instruction": s.get("instruction", ""),
                    "input": s.get("input", ""),
                    "output": s.get("output", ""),
                }
                for s in self.merged_data
            ]
        else:
            # Raw format
            output_data = self.merged_data

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n[SAVED] {output_path}")

    def print_summary(self):
        """Print summary of merged data"""
        print("\n" + "=" * 70)
        print("TRAINING DATA MERGE SUMMARY")
        print("=" * 70)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"\nBy source:")
        for source, count in sorted(
            self.stats["by_source"].items(), key=lambda x: -x[1]
        ):
            print(f"  {source}: {count} samples")
        print(f"\nBy type:")
        for sample_type, count in sorted(
            self.stats["by_type"].items(), key=lambda x: -x[1]
        ):
            print(f"  {sample_type}: {count} samples")
        print("=" * 70)


def main():
    """Main merging pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge training data for self-improvement"
    )
    parser.add_argument(
        "--original-data",
        type=str,
        default="data/datasets/instruction_dataset",
        help="Original training dataset path",
    )
    parser.add_argument(
        "--formal-verification",
        type=str,
        default="verification_results/formal_verification_training.json",
        help="Formal verification training data",
    )
    parser.add_argument(
        "--semantic-grounding",
        type=str,
        default="Examiner1/lived_experience_log.json",
        help="Semantic grounding data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data_cycle_2.json",
        help="Output file for merged data",
    )
    parser.add_argument("--cycle", type=int, default=2, help="Training cycle number")
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate samples",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=None,
        help="Maximum samples per source",
    )
    parser.add_argument(
        "--prioritize-formal",
        action="store_true",
        help="Give 2x weight to formally verified proofs",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["huggingface", "alpaca", "raw"],
        default="huggingface",
        help="Output format",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING DATA MERGER - CYCLE {args.cycle}")
    print("=" * 70)
    print(f"Original data: {args.original_data}")
    print(f"Formal verification: {args.formal_verification}")
    print(f"Semantic grounding: {args.semantic_grounding}")
    print(f"Output: {args.output}")
    print("=" * 70 + "\n")

    # Initialize merger
    merger = TrainingDataMerger()

    # Load original training data
    print("[STEP 1/5] Loading original training data...")
    # Note: For HF datasets, you may need to load differently
    # For now, assume JSON format or handle accordingly
    try:
        from datasets import load_from_disk

        original_dataset = load_from_disk(args.original_data)
        # Convert to list of dicts
        original_samples = []
        for split in original_dataset:
            for item in original_dataset[split]:
                original_samples.append(dict(item))
        merger.add_samples(original_samples)
    except Exception as e:
        print(f"[WARNING] Could not load as HF dataset: {e}")
        print("[INFO] Trying as JSON file...")
        original_samples = merger.load_training_data(
            args.original_data, "original_training"
        )
        merger.add_samples(original_samples)

    # Load formal verification data
    print("\n[STEP 2/5] Loading formal verification training data...")
    formal_samples = merger.load_training_data(
        args.formal_verification, "formal_verification"
    )
    merger.add_samples(formal_samples)

    # Load additional semantic grounding (optional)
    print("\n[STEP 3/5] Loading semantic grounding data...")
    if Path(args.semantic_grounding).exists():
        grounding_samples = merger.load_training_data(
            args.semantic_grounding, "semantic_grounding"
        )
        merger.add_samples(grounding_samples)

    # Process data
    print("\n[STEP 4/5] Processing merged data...")

    if args.deduplicate:
        merger.deduplicate()

    if args.max_per_source:
        merger.balance_sources(max_per_source=args.max_per_source)

    if args.prioritize_formal:
        merger.prioritize_formal_proofs()

    # Save merged data
    print("\n[STEP 5/5] Saving merged training data...")
    merger.save_merged_data(args.output, args.cycle, args.format)

    # Print summary
    merger.print_summary()

    print(f"\n✅ Merge complete!")
    print(f"\nNext: Run training cycle {args.cycle}")
    print(f"  .venv/bin/python3 train_consolidated.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

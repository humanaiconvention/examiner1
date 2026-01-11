#!/usr/bin/env python3
"""
Model Evaluation using lm-evaluation-harness
Runs MMLU-Pro and GSM8K benchmarks on trained checkpoint
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import lm-eval
try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("[ERROR] lm-eval not installed. Run: uv pip install lm-eval")
    sys.exit(1)


class LoRAEvaluator:
    """Evaluator for LoRA-trained models"""

    def __init__(
        self, base_model_path: str, checkpoint_path: str, device: str = "cuda"
    ):
        """
        Initialize evaluator

        Args:
            base_model_path: Path to base model (e.g., mlabonne/Qwen3-4B-abliterated)
            checkpoint_path: Path to LoRA checkpoint (e.g., Examiner1/models/grounding_cycle_1/final)
            device: Device to use (cuda/cpu)
        """
        self.base_model_path = base_model_path
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load base model and apply LoRA adapters"""
        print(f"\n[1/3] Loading base model: {self.base_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print(f"[2/3] Loading model to {self.device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Load LoRA adapters
        print(f"[3/3] Loading LoRA adapters from: {self.checkpoint_path}")
        self.model = PeftModel.from_pretrained(base_model, str(self.checkpoint_path))
        self.model.eval()

        print(f"[OK] Model loaded successfully\n")

    def run_evaluation(self, tasks: list, num_fewshot: int = 5, batch_size: int = 1):
        """
        Run evaluation on specified tasks

        Args:
            tasks: List of task names (e.g., ['mmlu_pro', 'gsm8k'])
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation

        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            self.load_model()

        print("=" * 70)
        print(f"RUNNING EVALUATION: {', '.join(tasks)}")
        print("=" * 70)
        print(f"Model: {self.checkpoint_path}")
        print(f"Tasks: {tasks}")
        print(f"Few-shot: {num_fewshot}")
        print(f"Batch size: {batch_size}")
        print("=" * 70 + "\n")

        # Create HF model wrapper for lm-eval
        lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            device=self.device,
        )

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=self.device,
            log_samples=True,  # Keep individual samples for failure analysis
        )

        return results

    def extract_failed_samples(
        self, results: dict, output_file: str = "failed_samples.json"
    ):
        """
        Extract samples where model failed for formal verification

        Args:
            results: Evaluation results from lm-eval
            output_file: Path to save failed samples

        Returns:
            list: Failed samples with context
        """
        failed_samples = []

        for task_name, task_results in results.get("samples", {}).items():
            for sample in task_results:
                # Check if answer was incorrect
                if not sample.get("correct", False):
                    failed_sample = {
                        "task": task_name,
                        "question": sample.get("doc", {}).get("question", ""),
                        "context": sample.get("doc", {}).get("context", ""),
                        "model_answer": sample.get("response", ""),
                        "correct_answer": sample.get("target", ""),
                        "reasoning_trace": sample.get("full_response", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
                    failed_samples.append(failed_sample)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(failed_samples, f, indent=2)

        print(f"\n[FAILED SAMPLES] Extracted {len(failed_samples)} failed samples")
        print(f"[SAVED] {output_path}")

        return failed_samples

    def save_results(self, results: dict, output_file: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract summary metrics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": str(self.checkpoint_path),
            "base_model": self.base_model_path,
            "results": {},
        }

        for task_name, task_results in results.get("results", {}).items():
            summary["results"][task_name] = {
                "accuracy": task_results.get("acc", 0.0),
                "acc_norm": task_results.get("acc_norm", 0.0),
                "num_samples": task_results.get("alias", {}).get("num_samples", 0),
            }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[RESULTS SAVED] {output_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        for task_name, metrics in summary["results"].items():
            print(f"\n{task_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            if metrics.get("acc_norm"):
                print(f"  Normalized Accuracy: {metrics['acc_norm']:.2%}")
            print(f"  Samples: {metrics['num_samples']}")
        print("\n" + "=" * 70)

        return summary


def main():
    """Main evaluation pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained model with lm-eval")
    parser.add_argument(
        "--base-model",
        type=str,
        default="mlabonne/Qwen3-4B-abliterated",
        help="Base model path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Examiner1/models/grounding_cycle_1/final",
        help="LoRA checkpoint path",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu_pro,gsm8k",
        help="Comma-separated list of tasks (mmlu_pro,gsm8k)",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Base Model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}")
    print("=" * 70 + "\n")

    # Initialize evaluator
    evaluator = LoRAEvaluator(
        base_model_path=args.base_model,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Load model
    evaluator.load_model()

    # Run evaluation
    results = evaluator.run_evaluation(
        tasks=tasks, num_fewshot=args.num_fewshot, batch_size=args.batch_size
    )

    # Save results
    results_file = output_dir / "evaluation_results.json"
    summary = evaluator.save_results(results, str(results_file))

    # Extract failed samples for formal verification
    failed_file = output_dir / "failed_samples.json"
    failed_samples = evaluator.extract_failed_samples(results, str(failed_file))

    print(f"\n✅ Evaluation complete!")
    print(f"   Results: {results_file}")
    print(f"   Failed samples: {failed_file}")
    print(f"   Total failed: {len(failed_samples)}")
    print(f"\nNext: Run formal verification with aristotle_verifier.py\n")


if __name__ == "__main__":
    main()

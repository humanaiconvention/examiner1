#!/usr/bin/env python3
"""
Aristotle API Formal Verification Loop
Sends failed reasoning samples to Aristotle for informal-to-formal verification
Extracts Lean 4 proof traces for self-improvement training
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import requests


class AristotleVerifier:
    """Interface to Aristotle API for formal verification"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://aristotle.harmonic.fun/v1",
        timeout: int = 300,
    ):
        """
        Initialize Aristotle verifier

        Args:
            api_key: Aristotle API key (or set ARISTOTLE_API_KEY env var)
            api_base: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ARISTOTLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Aristotle API key required. Set ARISTOTLE_API_KEY env var or pass api_key"
            )

        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def verify_reasoning(
        self, question: str, reasoning: str, answer: str, context: str = ""
    ) -> Dict:
        """
        Send reasoning to Aristotle for formal verification

        Args:
            question: The problem statement
            reasoning: Model's reasoning trace
            answer: Model's answer
            context: Additional context

        Returns:
            dict: Verification results with Lean 4 proof
        """
        payload = {
            "task": "informal_to_formal",
            "informal_statement": question,
            "informal_proof": reasoning,
            "proposed_answer": answer,
            "context": context,
            "return_format": "lean4",
            "verify": True,
        }

        try:
            response = requests.post(
                f"{self.api_base}/verify",
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Aristotle API request failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "question": question,
                "reasoning": reasoning,
            }

    async def verify_reasoning_async(
        self, question: str, reasoning: str, answer: str, context: str = ""
    ) -> Dict:
        """Async version of verify_reasoning"""
        payload = {
            "task": "informal_to_formal",
            "informal_statement": question,
            "informal_proof": reasoning,
            "proposed_answer": answer,
            "context": context,
            "return_format": "lean4",
            "verify": True,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base}/verify",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    return await response.json()

            except Exception as e:
                print(f"[ERROR] Async verification failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "question": question,
                    "reasoning": reasoning,
                }

    async def verify_batch_async(self, failed_samples: List[Dict]) -> List[Dict]:
        """
        Verify multiple failed samples concurrently

        Args:
            failed_samples: List of failed samples from evaluation

        Returns:
            list: Verification results for each sample
        """
        print(f"\n[ARISTOTLE] Verifying {len(failed_samples)} failed samples...")

        tasks = []
        for sample in failed_samples:
            task = self.verify_reasoning_async(
                question=sample.get("question", ""),
                reasoning=sample.get("reasoning_trace", ""),
                answer=sample.get("model_answer", ""),
                context=sample.get("context", ""),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        verified_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                verified_results.append(
                    {
                        "status": "error",
                        "error": str(result),
                        "original_sample": failed_samples[i],
                    }
                )
            else:
                verified_results.append(result)

        return verified_results

    def verify_batch(self, failed_samples: List[Dict]) -> List[Dict]:
        """Synchronous wrapper for batch verification"""
        return asyncio.run(self.verify_batch_async(failed_samples))

    def extract_lean4_proofs(self, verification_results: List[Dict]) -> List[Dict]:
        """
        Extract Lean 4 proof traces from verification results

        Args:
            verification_results: Results from Aristotle verification

        Returns:
            list: Extracted proof traces with metadata
        """
        proofs = []

        for result in verification_results:
            if result.get("status") == "verified":
                proof = {
                    "informal_statement": result.get("informal_statement", ""),
                    "formal_statement": result.get("formal_statement", ""),
                    "lean4_proof": result.get("lean4_proof", ""),
                    "proof_status": result.get("proof_status", ""),
                    "verification_time": result.get("verification_time", 0),
                    "correctness": result.get("is_correct", False),
                    "error_analysis": result.get("error_analysis", ""),
                    "corrected_reasoning": result.get("corrected_reasoning", ""),
                    "timestamp": datetime.now().isoformat(),
                }
                proofs.append(proof)

        return proofs

    def format_as_training_data(
        self, proofs: List[Dict], output_file: str = "formal_verification_training.json"
    ) -> List[Dict]:
        """
        Convert Lean 4 proofs to instruction tuning format

        Args:
            proofs: Extracted Lean 4 proofs
            output_file: Path to save training data

        Returns:
            list: Training data in instruction-response format
        """
        training_data = []

        for proof in proofs:
            # Format 1: Informal to Formal translation
            if proof.get("formal_statement"):
                training_sample = {
                    "instruction": "Translate the following informal mathematical statement into formal Lean 4 code.",
                    "input": proof["informal_statement"],
                    "output": proof["formal_statement"],
                    "source": "aristotle_verification",
                    "sample_type": "informal_to_formal",
                    "metadata": {
                        "verification_status": proof.get("proof_status", ""),
                        "is_correct": proof.get("correctness", False),
                    },
                }
                training_data.append(training_sample)

            # Format 2: Proof construction
            if proof.get("lean4_proof"):
                training_sample = {
                    "instruction": "Provide a formal Lean 4 proof for the following statement.",
                    "input": f"Statement: {proof['formal_statement']}\n\nProve this formally in Lean 4.",
                    "output": proof["lean4_proof"],
                    "source": "aristotle_verification",
                    "sample_type": "formal_proof",
                    "metadata": {
                        "verification_status": proof.get("proof_status", ""),
                        "verification_time": proof.get("verification_time", 0),
                    },
                }
                training_data.append(training_sample)

            # Format 3: Error correction (if reasoning was incorrect)
            if not proof.get("correctness") and proof.get("corrected_reasoning"):
                training_sample = {
                    "instruction": "Identify and correct the logical error in the following reasoning.",
                    "input": proof["informal_statement"],
                    "output": f"Error Analysis: {proof['error_analysis']}\n\nCorrected Reasoning: {proof['corrected_reasoning']}",
                    "source": "aristotle_verification",
                    "sample_type": "error_correction",
                    "metadata": {
                        "original_incorrect": True,
                        "verification_status": proof.get("proof_status", ""),
                    },
                }
                training_data.append(training_sample)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(training_data, f, indent=2)

        print(f"\n[TRAINING DATA] Generated {len(training_data)} samples")
        print(f"[SAVED] {output_path}")

        return training_data

    def save_verification_results(
        self, results: List[Dict], output_file: str = "verification_results.json"
    ):
        """Save verification results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "verified": sum(1 for r in results if r.get("status") == "verified"),
            "errors": sum(1 for r in results if r.get("status") == "error"),
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[VERIFICATION RESULTS]")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Verified: {summary['verified']}")
        print(f"  Errors: {summary['errors']}")
        print(f"  Saved to: {output_path}")


def main():
    """Main verification pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Formal verification with Aristotle API"
    )
    parser.add_argument(
        "--failed-samples",
        type=str,
        default="evaluation_results/failed_samples.json",
        help="Path to failed samples JSON",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Aristotle API key (or set ARISTOTLE_API_KEY)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://aristotle.harmonic.fun/v1",
        help="Aristotle API base URL",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="verification_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to verify (for testing)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ARISTOTLE FORMAL VERIFICATION PIPELINE")
    print("=" * 70)
    print(f"Failed samples: {args.failed_samples}")
    print(f"API base: {args.api_base}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70 + "\n")

    # Load failed samples
    failed_samples_path = Path(args.failed_samples)
    if not failed_samples_path.exists():
        print(f"[ERROR] Failed samples file not found: {failed_samples_path}")
        print("Run evaluate_model.py first to generate failed samples")
        sys.exit(1)

    with open(failed_samples_path) as f:
        failed_samples = json.load(f)

    print(f"[LOADED] {len(failed_samples)} failed samples")

    # Limit samples if requested
    if args.max_samples:
        failed_samples = failed_samples[: args.max_samples]
        print(f"[LIMITED] Processing first {len(failed_samples)} samples")

    # Initialize verifier
    try:
        verifier = AristotleVerifier(
            api_key=args.api_key, api_base=args.api_base, timeout=args.timeout
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify batch
    print("\n[STEP 1/3] Running formal verification...")
    verification_results = verifier.verify_batch(failed_samples)

    # Save verification results
    results_file = output_dir / "verification_results.json"
    verifier.save_verification_results(verification_results, str(results_file))

    # Extract Lean 4 proofs
    print("\n[STEP 2/3] Extracting Lean 4 proof traces...")
    proofs = verifier.extract_lean4_proofs(verification_results)
    print(f"[EXTRACTED] {len(proofs)} valid proof traces")

    # Format as training data
    print("\n[STEP 3/3] Formatting as instruction tuning data...")
    training_file = output_dir / "formal_verification_training.json"
    training_data = verifier.format_as_training_data(proofs, str(training_file))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"Verification results: {results_file}")
    print(f"Training data: {training_file}")
    print(f"Training samples: {len(training_data)}")
    print("\nNext: Merge with existing training data and run next training cycle")
    print("  python merge_training_data.py")
    print("  .venv/bin/python3 train_consolidated.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

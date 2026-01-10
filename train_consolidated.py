#!/usr/bin/env python3
"""
CONSOLIDATED TRAINING: Semantic Grounding Cycle 1
Windows-compatible version with simplified configuration
"""

import os
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import warnings
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "num_train_samples": 100,
    "num_eval_samples": 25,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "warmup_steps": 5,
    "logging_steps": 5,
    "save_steps": 25,
    "eval_steps": 25,
    "num_epochs": 1,
    "model_name": "microsoft/phi-2",
    "output_dir": "models/grounding_cycle_1",
    "lora_r": 8,
    "lora_alpha": 16,
}

print("=" * 70)
print("SEMANTIC GROUNDING CYCLE 1: WINDOWS-COMPATIBLE TRAINING")
print("=" * 70)

start_time = time.time()
training_success = False

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1/6] Loading dataset...")
try:
    ds = load_from_disk("data/datasets/instruction_dataset")
    train_ds = ds["train"].select(
        list(range(min(CONFIG["num_train_samples"], len(ds["train"]))))
    )
    test_ds = ds["test"].select(
        list(range(min(CONFIG["num_eval_samples"], len(ds["test"]))))
    )
    print(f"[OK] Loaded {len(train_ds)} train, {len(test_ds)} test samples")
except Exception as e:
    print(f"[ERROR] Dataset loading failed: {e}")
    sys.exit(1)

# ============================================================================
# 2. LOAD TOKENIZER
# ============================================================================
print("\n[2/6] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"[OK] Tokenizer loaded")
except Exception as e:
    print(f"[ERROR] Tokenizer loading failed: {e}")
    sys.exit(1)

# ============================================================================
# 3. LOAD MODEL
# ============================================================================
print("\n[3/6] Loading Phi-2 model...")
try:
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  -> Device: {device}")

    if device == "cpu":
        print("  [WARNING] CUDA not detected - training will be VERY slow")
        print("  [WARNING] Check CUDA installation if GPU is available")

    # Load model with SDPA attention and fp16
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    # Move to GPU if available
    if device == "cuda":
        model = model.to(device)
        print(f"  -> Model on GPU with SDPA attention (fp16)")
    else:
        print(f"  -> Model on CPU (fp32)")

    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    print(f"[OK] Model loaded on {device}")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. CONFIGURE LORA
# ============================================================================
print("\n[4/6] Configuring LoRA...")
try:
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["Wqkv", "fc1", "fc2"],  # Phi-2 specific modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"[OK] LoRA configured (r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})")
except Exception as e:
    print(f"[ERROR] LoRA configuration failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 5. TOKENIZE DATASET
# ============================================================================
print("\n[5/6] Tokenizing dataset...")
try:

    def tokenize_function(examples):
        # Format as instruction-response pairs
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"Instruction: {examples['instruction'][i]}\n"
            if examples["input"][i]:
                text += f"Input: {examples['input'][i]}\n"
            text += f"Response: {examples['output'][i]}"
            texts.append(text)

        # Tokenize
        result = tokenizer(
            texts,
            truncation=True,
            max_length=CONFIG["max_seq_length"],
            padding="max_length",
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize datasets
    train_dataset = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )

    eval_dataset = test_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=test_ds.column_names,
        desc="Tokenizing eval",
    )

    print(f"[OK] Tokenization complete")
except Exception as e:
    print(f"[ERROR] Tokenization failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 6. TRAIN
# ============================================================================
print("\n[6/6] Training...")
print("Configuration:")
print(
    f"  Batch size: {CONFIG['train_batch_size']} × {CONFIG['gradient_accumulation_steps']} accumulation"
)
print(
    f"  Effective batch: {CONFIG['train_batch_size'] * CONFIG['gradient_accumulation_steps']}"
)
print(f"  Max sequence: {CONFIG['max_seq_length']} tokens")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Samples: {len(train_dataset)} train, {len(eval_dataset)} eval")

try:
    print("\n[DEBUG] Creating TrainingArguments...")
    sys.stdout.flush()

    # Training arguments - Windows compatible
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        report_to="none",
        logging_dir=f"{CONFIG['output_dir']}/logs",
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Critical for Windows - no multiprocessing
        disable_tqdm=False,
        logging_first_step=True,
        dataloader_drop_last=False,
    )

    print("[DEBUG] TrainingArguments created successfully")
    sys.stdout.flush()

    print("[DEBUG] Creating DataCollator...")
    sys.stdout.flush()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("[DEBUG] DataCollator created successfully")
    sys.stdout.flush()

    print("[DEBUG] Creating Trainer (this may take a moment)...")
    sys.stdout.flush()

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("[DEBUG] Trainer created successfully")
    sys.stdout.flush()

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    print("[DEBUG] Calling trainer.train()...")
    sys.stdout.flush()

    train_start = time.time()
    result = trainer.train()
    train_duration = time.time() - train_start

    print(f"[DEBUG] Training loop completed")
    sys.stdout.flush()
    training_success = True

    print(f"\n[OK] Training completed in {train_duration / 60:.2f} minutes")
    print(f"Final loss: {result.training_loss:.4f}")

    # Evaluate
    print("\n[DEBUG] Starting evaluation...")
    sys.stdout.flush()
    eval_result = trainer.evaluate()
    print(f"Eval loss: {eval_result['eval_loss']:.4f}")
    print(f"[DEBUG] Evaluation completed")
    sys.stdout.flush()

except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback

    traceback.print_exc()
    training_success = False
    sys.exit(1)

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
if training_success:
    print("\n[SAVE] Saving trained model...")
    try:
        model_path = Path(CONFIG["output_dir"]) / "final"
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))

        print(f"[OK] Model saved to {model_path}")
        print(f"[OK] LoRA adapters saved")
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")
        import traceback

        traceback.print_exc()

# ============================================================================
# 8. CLEANUP
# ============================================================================
print("\n[CLEANUP] Releasing GPU resources...")
try:
    if "model" in locals():
        del model
    if "trainer" in locals():
        del trainer
    torch.cuda.empty_cache()
    print("[OK] GPU memory cleared")
except Exception as e:
    print(f"⚠ Cleanup warning: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
total_time = time.time() - start_time

print("\n" + "=" * 70)
print("SEMANTIC GROUNDING CYCLE 1: COMPLETE")
print("=" * 70)
if training_success:
    print(f"\nTraining Time: {train_duration / 60:.2f} minutes")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Model Path: {CONFIG['output_dir']}/final")
    print(f"\nData grounded from:")
    print(f"  • Research Corpus: 105 papers")
    print(f"  • Training Dataset: {len(train_dataset)} samples")
    print(f"\n✅ Training successful!")
else:
    print(f"\n❌ Training failed - see errors above")
    print(f"Total Time: {total_time / 60:.2f} minutes")

print("\n" + "=" * 70)

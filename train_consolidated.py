#!/usr/bin/env python3
"""
CONSOLIDATED TRAINING: Semantic Grounding Cycle 1
Windows-compatible version with simplified configuration
Using Qwen3-4B-abliterated (uncensored) model
"""

import os
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import time
import warnings
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose model size:
# False = Qwen3-4B with FP16 (safer, ~6-7GB VRAM)
# True  = Qwen3-8B with 4-bit quantization (~7-8GB VRAM, tighter fit)
USE_8B_MODEL = False

# Enable 4-bit quantization even for 4B model (QLoRA)
USE_QLORA = True  # Use 4-bit for better memory efficiency

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
    "model_name": "mlabonne/Qwen3-8B-abliterated"
    if USE_8B_MODEL
    else "mlabonne/Qwen3-4B-abliterated",
    "use_4bit": USE_QLORA or USE_8B_MODEL,  # Use 4-bit quantization (QLoRA)
    "output_dir": "Examiner1/models/grounding_cycle_1",
    "lora_r": 8,
    "lora_alpha": 16,
}

if USE_8B_MODEL:
    model_size = "8B (4-bit QLoRA)"
elif USE_QLORA:
    model_size = "4B (4-bit QLoRA)"
else:
    model_size = "4B (FP16)"
print("=" * 70)
print(f"SEMANTIC GROUNDING CYCLE 1: Qwen3-{model_size}")
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
print(f"\n[3/6] Loading {CONFIG['model_name']} model...")
try:
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  -> Device: {device}")

    if device == "cpu":
        print("  [WARNING] CUDA not detected - training will be VERY slow")
        print("  [WARNING] Check CUDA installation if GPU is available")

    # Configure 4-bit quantization if needed (for 8B model)
    if CONFIG.get("use_4bit", False) and device == "cuda":
        print(f"  -> Using 4-bit quantization (QLoRA) for memory efficiency")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            trust_remote_code=True,
            quantization_config=bnb_config,
            attn_implementation="eager",
            device_map="auto",
        )
        print(f"  -> Model loaded in 4-bit on {device} with eager attention")
    else:
        # Load model with eager attention and fp16
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            trust_remote_code=True,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="eager",
        )

        # Move to device after loading (before LoRA)
        if device == "cuda":
            model.cuda()

        print(f"  -> Model on {device} with eager attention")

    # Enable gradient checkpointing for QLoRA
    if CONFIG.get("use_4bit", False):
        print("  -> Preparing model for QLoRA training...")
        model.config.use_cache = False  # Required for gradient checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("  -> Gradient checkpointing enabled")
    else:
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Qwen3 modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"[OK] LoRA configured (r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})")

    # Test model forward pass to catch issues early
    print("\n[TEST] Testing model forward pass...")
    model.train()
    test_batch = {
        "input_ids": torch.randint(0, 50000, (1, 64)).to(device),
        "attention_mask": torch.ones(1, 64, dtype=torch.long).to(device),
        "labels": torch.randint(0, 50000, (1, 64)).to(device),
    }
    try:
        with torch.amp.autocast("cuda", enabled=False):
            test_output = model(**test_batch)
            test_loss = test_output.loss
            test_loss.backward()
        print(f"[OK] Model test passed (loss: {test_loss.item():.4f})")
        del test_batch, test_output, test_loss
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ERROR] Model forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

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

    # Set format for PyTorch
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    print(f"[OK] Tokenization complete")
except Exception as e:
    print(f"[ERROR] Tokenization failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 6. MANUAL TRAINING LOOP (Windows compatible)
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
    # Setup output directory
    output_path = Path(CONFIG["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=CONFIG["eval_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=data_collator,
    )

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    total_steps = (
        len(train_dataloader)
        * CONFIG["num_epochs"]
        // CONFIG["gradient_accumulation_steps"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    # Warmup: Pre-fetch first batch to initialize dataloader (Windows fix)
    print("[DEBUG] Warming up dataloader...")
    sys.stdout.flush()
    dataloader_iter = iter(train_dataloader)
    try:
        first_batch = next(dataloader_iter)
        print("[DEBUG] First batch loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Failed to load first batch: {e}")
        sys.exit(1)

    print("[DEBUG] Moving first batch to device...")
    sys.stdout.flush()
    # Move batch to device BEFORE entering training loop to avoid hang
    first_batch = {k: v.to(device) for k, v in first_batch.items()}
    print("[DEBUG] First batch moved to device successfully")
    sys.stdout.flush()

    print("[DEBUG] Setting model to train mode...")
    sys.stdout.flush()
    model.train()
    print("[DEBUG] Model set to train mode")
    sys.stdout.flush()

    train_start = time.time()
    global_step = 0
    total_loss = 0.0

    print("[DEBUG] Starting epoch loop...")
    sys.stdout.flush()

    for epoch in range(CONFIG["num_epochs"]):
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        sys.stdout.flush()

        # Use pre-fetched first batch (already on device)
        print("[DEBUG] Processing first batch...")
        sys.stdout.flush()
        batch = first_batch
        step = 0
        # Process first batch
        print("[DEBUG] Running forward pass...")
        sys.stdout.flush()
        outputs = model(**batch)
        print("[DEBUG] Forward pass complete")
        sys.stdout.flush()
        loss = outputs.loss / CONFIG["gradient_accumulation_steps"]
        print(f"[DEBUG] Starting backward pass (loss: {loss.item():.4f})...")
        sys.stdout.flush()
        loss.backward()
        print("[DEBUG] Backward pass complete")
        sys.stdout.flush()
        total_loss += loss.item() * CONFIG["gradient_accumulation_steps"]

        if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % CONFIG["logging_steps"] == 0:
                avg_loss = total_loss / global_step
                print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")
                sys.stdout.flush()

        # Continue with remaining batches
        for step, batch in enumerate(dataloader_iter, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / CONFIG["gradient_accumulation_steps"]

            # Backward pass
            loss.backward()

            total_loss += loss.item() * CONFIG["gradient_accumulation_steps"]

            # Gradient accumulation
            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % CONFIG["logging_steps"] == 0:
                    avg_loss = total_loss / global_step
                    print(f"Step {global_step}/{total_steps} | Loss: {avg_loss:.4f}")
                    sys.stdout.flush()

                # Save checkpoint
                if global_step % CONFIG["save_steps"] == 0:
                    checkpoint_path = output_path / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"\n[CHECKPOINT] Saved to {checkpoint_path}")
                    sys.stdout.flush()

    train_duration = time.time() - train_start
    training_success = True
    avg_train_loss = total_loss / max(global_step, 1)

    print(f"\n[OK] Training completed in {train_duration / 60:.2f} minutes")
    print(f"Average loss: {avg_train_loss:.4f}")

    # Evaluation
    print("\n[EVAL] Evaluating model...")
    model.eval()
    eval_loss = 0.0
    eval_steps = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
            eval_steps += 1

    avg_eval_loss = eval_loss / max(eval_steps, 1)
    print(f"Eval loss: {avg_eval_loss:.4f}")

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
        model.save_pretrained(str(model_path))
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
    if "optimizer" in locals():
        del optimizer
    torch.cuda.empty_cache()
    print("[OK] GPU memory cleared")
except Exception as e:
    print(f"⚠ Cleanup warning: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
total_time = time.time() - start_time

print("\n" + "=" * 70)
print(f"SEMANTIC GROUNDING CYCLE 1: COMPLETE (Qwen3-{model_size})")
print("=" * 70)
if training_success:
    print(f"\nTraining Time: {train_duration / 60:.2f} minutes")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Model Path: {CONFIG['output_dir']}/final")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Eval Loss: {avg_eval_loss:.4f}")
    print(f"\nData grounded from:")
    print(f"  • Research Corpus: 105 papers")
    print(f"  • Training Dataset: {len(train_dataset)} samples")
    print(f"\n✅ Training successful!")
else:
    print(f"\n❌ Training failed - see errors above")
    print(f"Total Time: {total_time / 60:.2f} minutes")

print("\n" + "=" * 70)

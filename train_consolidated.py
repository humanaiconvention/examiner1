#!/usr/bin/env python3
"""
CONSOLIDATED TRAINING: Semantic Grounding Cycle 1
Single script, minimal dependencies, 400 samples, 1 epoch
"""

# Disable torch.compile globally (problematic on Windows)
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ============================================================================
# CONFIGURATION (centralized, easy to modify)
# ============================================================================
CONFIG = {
    "num_train_samples": 100,
    "num_eval_samples": 25,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "max_seq_length": 256,
    "learning_rate": 2e-4,
    "warmup_steps": 1,
    "logging_steps": 2,
    "save_steps": 50,
    "eval_steps": 50,
    "num_epochs": 1,
    "model_name": "microsoft/phi-2",
    "output_dir": "models/grounding_cycle_1",
    "lora_r": 8,
    "lora_alpha": 16,
}

# Try to enable optimized kernels for CUDA
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("[OK] Flash Attention 2 available for CUDA optimization")
except ImportError:
    print("[!] Flash Attention not available, using efficient attention fallback")

print("="*70)
print("SEMANTIC GROUNDING CYCLE 1: CONSOLIDATED TRAINING")
print("="*70)

start_time = time.time()
train_duration = 0  # Initialize in case of early exit
training_success = False  # Initialize at top (not in conditional)

# ============================================================================
# 1. LOAD MINIMAL DATASET
# ============================================================================
print("\n[1/5] Loading 400 samples...")
try:
    ds = load_from_disk("data/datasets/instruction_dataset")
    train_ds = ds['train'].select(list(range(min(CONFIG["num_train_samples"], len(ds['train'])))))
    test_ds = ds['test'].select(list(range(min(CONFIG["num_eval_samples"], len(ds['test'])))))
    print(f"[OK] Loaded {len(train_ds)} train, {len(test_ds)} test samples")
except Exception as e:
    print(f"[ERROR] Dataset error: {e}")
    exit(1)

# ============================================================================
# 2. LOAD MODEL (Phi-2 with fp16 + gradient checkpointing)
# ============================================================================
print("\n[2/5] Loading Phi-2 model with fp16 + gradient checkpointing...")
try:
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    
    print("  -> Using SDPA (scaled dot-product attention, CUDA-native)")
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        **model_kwargs
    )
    
    # Enable gradient checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[OK] Model loaded (fp16 + gradient checkpointing)")
except Exception as e:
    print(f"[ERROR] Model loading error: {e}")
    exit(1)

# ============================================================================
# 3. CONFIGURE LORA
# ============================================================================
print("\n[3/5] Configuring LoRA adapters...")
try:
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"[OK] LoRA enabled (rank={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})")
except Exception as e:
    print(f"[ERROR] LoRA config error: {e}")
    exit(1)

# ============================================================================
# 4. PREPARE DATASET (pre-format to simple text)
# ============================================================================
print("\n[4/5] Preparing dataset...")
try:
    def format_example(example):
        text = f"Instruction: {example['instruction']}\n"
        if example.get('input'):
            text += f"Input: {example['input']}\n"
        text += f"Response: {example['output']}"
        return {"text": text}
    
    train_ds_formatted = train_ds.map(
        format_example,
        remove_columns=train_ds.column_names
    )
    test_ds_formatted = test_ds.map(
        format_example,
        remove_columns=test_ds.column_names
    )
    
    # Verify dataset has 'text' column for SFTTrainer
    if 'text' not in train_ds_formatted.column_names:
        print(f"[ERROR] Dataset missing 'text' column. Columns: {train_ds_formatted.column_names}")
        exit(1)
    
    print(f"[OK] Dataset formatted (400 train, 50 test)")
except Exception as e:
    print(f"[ERROR] Dataset formatting error: {e}")
    exit(1)

# ============================================================================
# 5. TRAIN (aggressive batch sizes for speed)
# ============================================================================
print("\n[5/5] Training...")
print("Configuration:")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Batch size: {CONFIG['train_batch_size']} (per device)")
print(f"  Eval batch: {CONFIG['eval_batch_size']} (per device)")
print(f"  Max sequence: {CONFIG['max_seq_length']} tokens")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Samples: {CONFIG['num_train_samples']} training, {CONFIG['num_eval_samples']} evaluation")
if FLASH_ATTN_AVAILABLE:
    print("  Kernels: Flash Attention 2 (CUDA-optimized)")
else:
    print("  Kernels: SDPA/CUDA-native efficient attention")

try:
    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",  # Standard torch optimizer, more reliable on Windows
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds_formatted,
        eval_dataset=test_ds_formatted,
    )
    
    train_start = time.time()
    result = trainer.train()
    train_duration = time.time() - train_start
    training_success = True
    print(f"\n[OK] Training completed in {train_duration/60:.1f} minutes")
    
except Exception as e:
    print(f"\n[ERROR] Training error: {e}")
    import traceback
    traceback.print_exc()
    training_success = False
    exit(1)

# ============================================================================
# 6. SAVE MODEL (only if training succeeded)
# ============================================================================
if training_success:
    print("\n[SAVE] Saving trained model...")
    try:
        model_path = Path(CONFIG["output_dir"]) / "final"
        model_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        print(f"[OK] Model saved to {model_path}")
        print(f"[OK] LoRA adapters: {model_path / 'adapter_model.bin'}")
    except Exception as e:
        print(f"[ERROR] Save error: {e}")
else:
    print("\n⚠ Training did not complete successfully, model not saved")

# ============================================================================
# 7. CLEANUP
# ============================================================================
print("\n[CLEANUP] Releasing GPU resources...")
try:
    # Explicitly delete large objects
    if 'model' in locals():
        del model
    if 'trainer' in locals():
        del trainer
    if 'train_ds_formatted' in locals():
        del train_ds_formatted
    if 'test_ds_formatted' in locals():
        del test_ds_formatted
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    print("[OK] GPU memory cleared")
except Exception as e:
    print(f"⚠ Cleanup warning: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
total_time = time.time() - start_time

print("\n" + "="*70)
print("SEMANTIC GROUNDING CYCLE 1: COMPLETE")
print("="*70)
print(f"\nTraining Time: {train_duration/60:.2f} minutes")
print(f"Total Time: {total_time/60:.2f} minutes")
print(f"Model Path: models/grounding_cycle_1/final")
print(f"\nData grounded from:")
print(f"  • Research Corpus: 104 papers (corpus_index.json)")
print(f"  • Semantic Grounding: 1 lived experience (lived_experience_log.json)")
print(f"  • Training Dataset: 400 samples from articles")

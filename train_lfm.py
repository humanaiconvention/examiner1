#!/usr/bin/env python3
"""
EXAMINER-LFM (LITE): Semantic Grounding for Edge AI
Target: liquidai/lfm-2.5-1.2b-instruct
Optimized for <4GB VRAM training via QLoRA
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

CONFIG = {
    # Data
    "num_train_samples": 100,      # Small cycle for validation
    "num_eval_samples": 25,
    "max_seq_length": 512,         # 512 is plenty for edge instruction following
    
    # Model
    "model_name": "liquidai/lfm-2.5-1.2b-instruct",
    "use_4bit": True,              # Essential for "Lite" (fits in <4GB VRAM)
    
    # Training
    "train_batch_size": 2,         # 1.2B is small enough for batch size 2 usually
    "gradient_accumulation_steps": 2, # Effective batch = 4
    "learning_rate": 3e-4,         # Higher LR for smaller model
    "num_epochs": 1,
    "warmup_steps": 5,
    "logging_steps": 5,
    
    # LoRA (Adapter)
    "lora_r": 16,                  # Higher rank for more expressivity in small model
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    
    # Paths
    "output_dir": "Examiner1/models/examiner_lfm_lite",
    "dataset_path": "data/datasets/instruction_dataset",
}

print("=" * 70)
print(f"EXAMINER-LFM: Semantic Grounding for Edge AI")
print(f"Target: {CONFIG['model_name']} (1.2B)")
print("=" * 70)

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train():
    start_time = time.time()
    
    # 1. Load Tokenizer
    print("\n[1/5] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for training
    
    # 2. Load Model (4-bit QLoRA)
    print("\n[2/5] Loading LFM 2.5 (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        trust_remote_code=True, # Required for LFM architecture
        device_map="auto",
    )
    
    # Prepare for LoRA
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    # 3. Apply LoRA Adapters
    print("\n[3/5] Applying LoRA Adapters...")
    peft_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CONFIG["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Load & Tokenize Data
    print("\n[4/5] Preparing Grounding Data...")
    try:
        ds = load_from_disk(CONFIG["dataset_path"])
        train_ds = ds["train"].select(range(min(CONFIG["num_train_samples"], len(ds["train"]))))
        
        def tokenize_fn(examples):
            texts = [
                f"<|im_start|>user\n{inst}\n<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
                for inst, out in zip(examples["instruction"], examples["output"])
            ]
            
            result = tokenizer(
                texts, 
                padding="max_length", 
                truncation=True, 
                max_length=CONFIG["max_seq_length"]
            )
            result["labels"] = result["input_ids"].copy()
            return result

        train_dataset = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
        train_dataset.set_format("torch")
        print(f"✓ Loaded {len(train_dataset)} grounding samples")
        
    except Exception as e:
        print(f"Data load error: {e}")
        return

    # 5. Training Loop
    print("\n[5/5] Training Examiner-LFM...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["train_batch_size"], 
        shuffle=True, 
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    model.train()
    total_loss = 0
    steps = 0
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        for batch in train_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            if (steps + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Step {steps+1} | Loss: {loss.item():.4f}")
            
            steps += 1
            total_loss += loss.item()

    # Save
    print("\n[SAVE] Exporting Lite Model...")
    out_path = Path(CONFIG["output_dir"])
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"✓ Saved to {out_path}")
    print(f"✓ Final Loss: {total_loss/steps:.4f}")

if __name__ == "__main__":
    train()

"""
Inference script for fine-tuned Qwen2.5-7B model with Test-Time Training
Load base model + LoRA adapters and generate responses with optional TTS
adaptation
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    from tts_adapter import create_tts_adaptor
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger_warn = logging.getLogger(__name__)
    logger_warn.warning("TTS adapter not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_finetuned_model(
    base_model_name: str = "Qwen/Qwen2.5-7B",
    lora_path: str = "models/final/lora_adapters",
    use_4bit: bool = True
):
    """
    Load base model and merge LoRA adapters
    
    Args:
        base_model_name: Base model identifier
        lora_path: Path to LoRA adapter weights
        use_4bit: Use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")
    
    # Configure quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer. from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Merge adapters for faster inference (optional)
    # model = model.merge_and_unload()
    
    logger.info("Model loaded successfully")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate response for given instruction
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        instruction:  Instruction/question
        input_text: Optional context
        max_new_tokens:  Maximum tokens to generate
        temperature:  Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty:  Repetition penalty
        
    Returns:
        Generated response
    """
    # Format prompt
    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response: 
"""
    else:
        prompt = f"""### Instruction: 
{instruction}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response = full_output. split("### Response:")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer):
    """Interactive chat mode"""
    print("\n" + "="*50)
    print("Qwen2.5-7B Fine-tuned Model - Interactive Mode")
    print("="*50)
    print("Type 'quit' to exit\n")
    
    while True:
        instruction = input("Enter your question:  ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not instruction:
            continue
        
        print("\nGenerating response.. .\n")
        
        response = generate_response(
            model,
            tokenizer,
            instruction=instruction,
            max_new_tokens=256,
            temperature=0.7
        )
        
        print(f"Response: {response}\n")
        print("-" * 50 + "\n")


def run_example_questions(model, tokenizer):
    """Run predefined example questions"""
    examples = [
        {
            "instruction": "Summarize the key findings from machine learning research.",
            "input": ""
        },
        {
            "instruction": "What are the main challenges in deep learning?",
            "input":  ""
        },
        {
            "instruction": "Explain the concept of transfer learning.",
            "input": ""
        },
        {
            "instruction": "What methodologies are commonly used in AI research?",
            "input":  ""
        },
        {
            "instruction":  "Describe the relationship between model size and performance.",
            "input": ""
        }
    ]
    
    print("\n" + "="*50)
    print("Running Example Questions")
    print("="*50 + "\n")
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Question:  {example['instruction']}\n")
        
        response = generate_response(
            model,
            tokenizer,
            instruction=example['instruction'],
            input_text=example. get('input', ''),
            max_new_tokens=256,
            temperature=0.7
        )
        
        print(f"Response: {response}\n")
        print("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Qwen2.5-7B")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Base model directory or HF identifier"
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="models/final/lora_adapters",
        help="LoRA adapters directory"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional input context"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run example questions"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch. cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU (slow)")
    
    # Load model
    model, tokenizer = load_finetuned_model(
        base_model_name=args.model_dir,
        lora_path=args.lora_dir
    )
    
    # Run modes
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.examples:
        run_example_questions(model, tokenizer)
    elif args.question:
        response = generate_response(
            model,
            tokenizer,
            instruction=args.question,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"\nQuestion: {args.question}")
        if args.input:
            print(f"Context: {args.input}")
        print(f"\nResponse: {response}\n")
    else:
        print("Please specify --question, --interactive, or --examples")
        parser.print_help()


if __name__ == "__main__": 
    main()
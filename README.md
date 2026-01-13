# Examiner

**Semantic Grounding Framework for Fine-Tuning Language Models with QLoRA**

Examiner is an open-source framework for **semantic grounding**: integrating human-validated expert knowledge into language models to establish meaning through recursive structural relationships. **You bring your corpus and expertise**—Examiner automates the framework to extract, structure, and inject domain knowledge into AI systems.

Build semantically grounded AI for any domain: medical diagnosis, legal analysis, scientific research, or specialized expertise. Your lived experience and knowledge become the foundation for model understanding.

**Now with QLoRA support** - Train 4B-8B parameter models on consumer GPUs (8GB VRAM).

## 🎯 What is Semantic Grounding?

Semantic grounding is the process of **establishing meaning through recursive structural relationships that preserve information across system levels**. It requires integrating human-validated knowledge into language models—capturing how domain experts actually understand patterns, relationships, and context within their field.

Currently, this requires **human lived experience injection**: expert knowledge, reasoning patterns, and contextual understanding that cannot be extracted from documents alone. Examiner automates the framework; your expertise provides the grounding.

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/examiner.git
cd examiner

# Install uv (fast package manager - recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
python3 -m venv .venv
~/.local/bin/uv pip install -r requirements.txt

# Or use pip (slower)
# .venv/bin/pip install -r requirements.txt
```

### 2. Prepare Your Corpus

📖 **See [CORPUS_SETUP.md](CORPUS_SETUP.md) for detailed instructions**

```bash
# Create directories
mkdir -p data/pdfs

# Add your domain-specific PDF files to data/pdfs/
# (academic papers, technical documentation, etc.)

# Extract and index your corpus
python pdf_to_dataset.py \
  --input data/pdfs \
  --output data/datasets/corpus \
  --format json
```

**For corpus composition examples, see [EXAMPLE_CORPUS_SOURCES.md](EXAMPLE_CORPUS_SOURCES.md)**

### 3. Capture Expert Knowledge (Semantic Grounding)
This is the core of the Examiner framework. To prevent "model collapse" (semantic drift), the model needs **orthogonal input** from lived human experience that cannot be generated from its own training data.

Run the interactive dialogue system to ground the model:

```bash
python lived_experience_dialogue.py --interactive
```

This will:
1.  Ask you about real-world observations (successes, failures, limitations)
2.  Generate a `lived_experience_log.json` file (excluded from git for privacy)
3.  Force the model to acknowledge and integrate this **Ground Truth**

### 4. Generate Training Data

```bash
python prepare_training_data.py
```

This combines your corpus and lived experience into a unified training dataset.

### 5. Fine-Tune Your Model with QLoRA

**Configuration** (in `train_consolidated.py`):

```python
# Choose model size:
USE_8B_MODEL = False  # True for 8B, False for 4B
USE_QLORA = True      # Enable 4-bit quantization (recommended)
```

**Hardware Requirements:**
- **4B Model (QLoRA)**: 3-4GB VRAM (RTX 2060+)
- **8B Model (QLoRA)**: 7-8GB VRAM (RTX 3070+, may be tight)
- **CPU Training**: Possible but very slow

**Run Training:**
```bash
.venv/bin/python3 train_consolidated.py
```

**Expected Output:**
```
Training Time: ~2-5 minutes (4B model, 100 samples)
Train Loss: 12.25 → 10.53 (↓14%)
Eval Loss: 2.45
Model saved to: Examiner1/models/grounding_cycle_1/final/
```

**What Gets Trained:**
- ✅ Qwen3-4B-abliterated (uncensored) with 4-bit QLoRA
- ✅ 16.5M trainable parameters (0.4% of total)
- ✅ LoRA adapters (r=8, alpha=16)
- ✅ Semantic grounding from your lived experience + corpus

### 6. Run Inference

```bash
python inference.py \
  --model Examiner1/models/grounding_cycle_1/final \
  --query "Your domain question"
```

## 📁 Core Modules

| Module | Purpose |
|--------|---------|
| `semantic_grounding.py` | Core grounding engine & corpus management |
| `pdf_to_dataset.py` | PDF extraction & dataset creation |
| `lived_experience_dialogue.py` | Interactive knowledge capture |
| `prepare_training_data.py` | Training data generation |
| `train_consolidated.py` | **QLoRA fine-tuning pipeline (4-bit quantization)** |
| `inference.py` | Grounding queries & inference |
| `add_grounding.py` | Batch semantic grounding capture (helper) |

## 🔥 QLoRA Training Features

- **4-bit Quantization**: Train 4B-8B models on 8GB GPUs
- **Gradient Checkpointing**: Automatic memory optimization
- **Uncensored Models**: Uses abliterated Qwen3 variants
- **Fast Training**: ~2-5 minutes for 100 samples
- **Semantic Grounding**: Integrates lived experience + research corpus
- **Windows/Linux Compatible**: Cross-platform support

## 🎯 Training Configuration

Edit `train_consolidated.py` to customize:

```python
CONFIG = {
    "num_train_samples": 100,      # Number of training samples
    "num_eval_samples": 25,         # Number of evaluation samples
    "train_batch_size": 1,          # Batch size (keep at 1 for 8GB GPU)
    "gradient_accumulation_steps": 4,  # Effective batch size = 4
    "max_seq_length": 512,          # Maximum sequence length
    "learning_rate": 2e-4,          # Learning rate
    "num_epochs": 1,                # Training epochs
    "lora_r": 8,                    # LoRA rank
    "lora_alpha": 16,               # LoRA alpha
}
```

## ⚠️ Troubleshooting

**Training hangs after "First batch loaded":**
- This was fixed in the current version
- Gradient checkpointing must be enabled for QLoRA
- `model.config.use_cache = False` is required

**Out of Memory (OOM):**
- Switch to 4B model: `USE_8B_MODEL = False`
- Reduce `max_seq_length` from 512 to 256
- Reduce `num_train_samples`

**Slow training:**
- Ensure CUDA is available: `torch.cuda.is_available()` should be `True`
- Check GPU utilization: `nvidia-smi`
- Use `uv` instead of `pip` for faster package management

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 12GB+ |
| CPU RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |

**Tested on**: NVIDIA RTX 2080 (8GB VRAM) with Phi-2 (2.7B)

## 📚 Building Your Corpus

**What is a corpus?** The structural foundation for semantic grounding—documents that encode domain-specific patterns, relationships, and knowledge structures:
- Research papers that establish recursive relationships and theoretical frameworks
- Technical documentation that captures domain reasoning patterns
- Industry standards that define structural constraints
- Expert-curated materials that preserve domain context
- Reference materials that maintain information fidelity across knowledge levels

**Recommended sizes:**
- **Small** (10-20 docs): 5-10 min training, basic quality
- **Medium** (50-100 docs): 20-60 min training, good quality
- **Large** (100+ docs): 1-4 hours training, excellent quality

See [EXAMPLE_CORPUS_SOURCES.md](EXAMPLE_CORPUS_SOURCES.md) for detailed examples of corpus composition and recommended sources for papers.

## 📖 Documentation

- **[Corpus Setup Guide](CORPUS_SETUP.md)** - How to build your corpus (READ FIRST!)
- **[Example Corpus Sources](EXAMPLE_CORPUS_SOURCES.md)** - Example composition and where to find papers
- **[Local Data Protection](LOCAL_DATA_PROTECTION.md)** - Keep your data safe and private
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines for the project

## 📊 Test Results

**Configuration:**
- Model: microsoft/phi-2 (2.7B)
- Training Data: 100 samples
- Corpus: 104 academic papers
- Hardware: RTX 2080 (8GB)

**Results:**
- Loss: 3.016 → 2.657 (↓ 11.9%)
- Training Time: 0.5 minutes
- GPU Usage: 6.5GB VRAM
- ✅ Successful inference and grounding

## 🛠️ Development

### Install from Source

```bash
git clone https://github.com/yourusername/examiner.git
cd examiner
pip install -r requirements.txt
python verify_system.py
```

### Verify Installation

```bash
python verify_system.py    # Verify all components
python architecture_auditor.py  # System audit and health checks
```

## 📦 Dependencies

- **PyTorch** >= 2.0
- **Hugging Face**: transformers, datasets, peft, trl
- **Data**: PyPDF2, numpy, pandas

See [requirements.txt](requirements.txt) for complete list.

## 🔐 License

MIT License - see [LICENSE](LICENSE). Use freely in academic and commercial projects.

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ❓ FAQ

**Q: Do I need domain expertise to build a corpus?**  
A: Yes—semantic grounding requires human validation. Domain experts ensure the corpus captures recursive structural relationships and contextual meaning that models cannot infer from documents alone.

**Q: How many documents do I need?**  
A: Start with 20-30, aim for 50-100 for good results.

**Q: What models does this support?**  
A: Any Hugging Face model (Phi-2, Llama, Mistral, Qwen, etc.).

**Q: Can I use this in production?**  
A: Yes! v1.0 is production-ready.

**Q: How do I update my grounding?**  
A: Add new papers, regenerate training data, and retrain in new cycles.

## 🚀 Getting Started

1. **Read** [CORPUS_SETUP.md](CORPUS_SETUP.md) - Essential guide
2. **Review** [EXAMPLE_CORPUS_SOURCES.md](EXAMPLE_CORPUS_SOURCES.md) - See example corpora
3. **Gather** your domain documents (50-100 PDFs)
4. **Run** the quick start commands above
5. **Test** inference on your knowledge

## 📈 Roadmap

- [x] v1.0: Core framework, Phi-2 support, LoRA fine-tuning
- [ ] v1.1: Multi-corpus support, improved evaluation
- [ ] v1.2: Additional model support (Llama, Mistral)
- [ ] v2.0: Web UI, API server, cloud deployment

---

**Ready to build grounded AI for your domain?** Start with [CORPUS_SETUP.md](CORPUS_SETUP.md) 🚀

*Made with ❤️ for semantic grounding in AI systems*

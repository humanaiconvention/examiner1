# Examiner v1.0

**Semantic Grounding Framework for Fine-Tuning Language Models**

Examiner is an open-source framework for building **semantic grounding systems** that align language models with domain-specific knowledge through research corpora, lived experience dialogue, and iterative fine-tuning.

## 🎯 What is Semantic Grounding?

Semantic grounding anchors AI model responses to authoritative information sources—research papers, domain expertise, and real-world context—ensuring accuracy, consistency, and trustworthiness.

Examiner provides:
- **Corpus Integration**: Index and integrate academic papers, documents, or knowledge bases
- **Lived Experience**: Capture domain expert knowledge through guided dialogue
- **Fine-Tuning**: Train models (Phi-2, Qwen, Llama, etc.) on grounded knowledge using LoRA
- **Verification**: Test grounding quality and measure semantic alignment
- **Iteration**: Improve grounding through supervised fine-tuning cycles

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  SEMANTIC GROUNDING CYCLE                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [1] CORPUS INTEGRATION        [2] LIVED EXPERIENCE          │
│  ├─ PDF extraction             ├─ Expert dialogue           │
│  ├─ Semantic indexing           ├─ Context capture          │
│  ├─ Deduplication              ├─ Grounding logs           │
│  └─ Corpus index (JSON)         └─ Experience store         │
│                                                               │
│  [3] TRAINING DATA             [4] FINE-TUNING              │
│  ├─ Instruction generation     ├─ Model loading            │
│  ├─ Corpus sampling            ├─ LoRA configuration       │
│  ├─ Data formatting            ├─ Training loop            │
│  └─ Train/eval splits          └─ Model checkpointing      │
│                                                               │
│  [5] VERIFICATION              [6] INFERENCE                │
│  ├─ Semantic alignment         ├─ Grounding queries        │
│  ├─ Corpus coverage            ├─ Answer generation        │
│  ├─ Quality metrics            ├─ Citation tracking        │
│  └─ Improvement scores         └─ Confidence scoring       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/examiner.git
cd examiner
pip install -r requirements.txt
```

### 2. Prepare Corpus

Place PDF files in `data/pdfs/`, then extract and index:

```bash
python pdf_to_dataset.py \
  --input data/pdfs \
  --output data/datasets/corpus \
  --format json
```

### 3. Capture Lived Experience

Run guided dialogue to capture domain expert knowledge:

```bash
python lived_experience_dialogue.py \
  --output lived_experience_log.json \
  --interactive
```

### 4. Generate Training Data

Create instruction-tuning dataset from corpus + lived experience:

```bash
python prepare_training_data.py \
  --corpus data/datasets/corpus \
  --experience lived_experience_log.json \
  --output training_data_cycle_1.json \
  --samples 400
```

### 5. Fine-Tune Model

Train your model with LoRA on grounded knowledge:

```bash
python train_consolidated.py
```

Configuration in `train_consolidated.py`:
```python
CONFIG = {
    "num_train_samples": 100,
    "train_batch_size": 1,
    "max_seq_length": 256,
    "learning_rate": 2e-4,
    "model_name": "microsoft/phi-2",
    "lora_r": 8,
    "lora_alpha": 16,
    "num_epochs": 1,
}
```

### 6. Run Inference

Test your grounded model:

```bash
python inference.py \
  --model models/grounding_cycle_1/final \
  --query "What is semantic grounding?" \
  --corpus data/datasets/corpus
```

## 📁 Core Modules

| Module | Purpose |
|--------|---------|
| `semantic_grounding.py` | Core grounding engine & corpus management |
| `pdf_to_dataset.py` | PDF extraction & dataset creation |
| `lived_experience_dialogue.py` | Interactive knowledge capture |
| `prepare_training_data.py` | Training data generation |
| `train_consolidated.py` | LoRA fine-tuning pipeline |
| `inference.py` | Grounding queries & inference |
| `architecture_auditor.py` | System verification & health checks |
| `tts_adapter.py` | Text-to-speech integration (optional) |

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 12GB+ |
| CPU RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |
| GPU | RTX 2080 | RTX 3090, A100 |

**Tested on**: NVIDIA RTX 2080 (8GB VRAM) with Phi-2 (2.7B) using LoRA

## 📊 Grounding Cycle 1 Results

**Configuration**:
- Model: microsoft/phi-2 (2.7B)
- Samples: 100 training, 25 eval
- Batch Size: 1
- Sequence Length: 256 tokens
- LoRA: rank=8, alpha=16

**Training Results**:
- Initial Loss: 3.016
- Final Loss: 2.657 ✓ (Loss decreased)
- Training Time: 0.5 minutes
- GPU Utilization: ~6.5GB VRAM

**Data**:
- Research Corpus: 104 papers
- Lived Experience: 1 dialogue cycle
- Training Dataset: 400 samples

## 🔄 Semantic Grounding Workflow

### Cycle 1: Foundation (Current - v1.0)
1. ✅ Corpus integration (104 papers)
2. ✅ Lived experience capture (1 expert dialogue)
3. ✅ Training data preparation (400 samples)
4. ✅ Model fine-tuning (Phi-2 + LoRA)
5. ⏳ Verification & inference testing

### Cycle 2: Refinement (Planned)
- Enhanced corpus (200+ papers)
- Multiple expert dialogues
- 1000+ training samples
- Model evaluation on grounding quality

### Cycle N: Production
- Full domain coverage
- Continuous improvement
- Multi-expert consensus
- Real-world deployment

## 🎓 Key Concepts

### Corpus Integration
Indexes research papers and domain documents, enabling semantic similarity search and context retrieval.

### Lived Experience
Captures tacit knowledge from domain experts—patterns, heuristics, and contextual understanding not found in documents.

### LoRA Fine-Tuning
Low-Rank Adaptation efficiently updates model weights (4-8% additional parameters) while preserving base model knowledge.

### Semantic Alignment
Measures how well model responses align with corpus knowledge through:
- Semantic similarity (embedding-based)
- Citation coverage (document references)
- Factuality scoring

## 📖 Documentation

- [System Summary](SYSTEM_SUMMARY.md) - Architecture overview
- [Quick Start](QUICK_START.md) - Step-by-step setup
- [Semantic Grounding Role](SEMANTIC_GROUNDING_ROLE_README.md) - Core concepts
- [Training Guide](TRAINING_CONSOLIDATED.md) - Fine-tuning details
- [Corpus Integration](CORPUS_INTEGRATION_SUMMARY.md) - Knowledge base setup
- [Lived Experience Guide](LIVED_EXPERIENCE_USAGE.md) - Expert dialogue capture

## 🛠️ Development

### Install for Development

```bash
git clone https://github.com/yourusername/examiner.git
cd examiner
pip install -r requirements.txt
python verify_system.py  # Verify installation
```

### Project Structure

```
examiner/
├── semantic_grounding.py          # Core framework
├── pdf_to_dataset.py              # Data pipeline
├── lived_experience_dialogue.py    # Knowledge capture
├── prepare_training_data.py        # Data preparation
├── train_consolidated.py           # Training script
├── inference.py                    # Inference pipeline
├── tts_adapter.py                  # Optional TTS
├── data/                           # Data directory
├── models/                         # Model outputs
├── logs/                           # Training logs
├── requirements.txt                # Dependencies
├── LICENSE                         # MIT License
├── README.md                       # This file
└── .gitignore                      # Git exclusions
```

### Testing

```bash
# Verify GPU setup
python check_gpu.py

# Test system components
python verify_system.py

# Run architecture audit
python architecture_auditor.py
```

## 📦 Dependencies

- **torch** >= 2.0 (PyTorch)
- **transformers** (Hugging Face)
- **datasets** (Hugging Face)
- **peft** (LoRA adapters)
- **trl** (Supervised fine-tuning trainer)
- **PyPDF2** (PDF extraction)
- **numpy, pandas** (Data processing)

See [requirements.txt](requirements.txt) for complete list.

## 🔐 License

Examiner is released under the **MIT License** - see [LICENSE](LICENSE) for details.

## 📝 Citation

If you use Examiner in your research, please cite:

```bibtex
@software{examiner2026,
  title={Examiner: Semantic Grounding Framework for Language Models},
  author={Contributors},
  year={2026},
  url={https://github.com/yourusername/examiner}
}
```

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Priorities

1. **Cycle 2 Expansion**: Scale to 200+ papers, 1000+ samples
2. **Multi-Model Support**: Llama, Mistral, GPT support
3. **Evaluation Framework**: Automated grounding quality metrics
4. **Production Deployment**: Docker containers, API server
5. **Documentation**: Video tutorials, example notebooks

## ❓ FAQ

**Q: Can I use this with other models?**  
A: Yes! Examiner supports any Hugging Face model. Update `CONFIG["model_name"]` in `train_consolidated.py`.

**Q: How much storage do I need?**  
A: ~50GB minimum (model weights + datasets). More for larger corpora.

**Q: Can I run this on CPU only?**  
A: Not recommended—training would be very slow. GPU with 8GB+ VRAM strongly recommended.

**Q: How do I add custom documents?**  
A: Place PDFs in `data/pdfs/` and run `pdf_to_dataset.py`. Supports PDF, TXT, and JSON.

**Q: Can I integrate external knowledge bases?**  
A: Yes, extend `semantic_grounding.py` to load from databases, APIs, or document stores.

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/examiner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/examiner/discussions)
- **Email**: contact@example.com (optional)

## 🙏 Acknowledgments

Built with:
- Hugging Face (transformers, datasets, trl)
- Meta (PEFT for LoRA)
- Microsoft (Phi model)
- PyTorch team

## 📈 Roadmap

- [ ] v1.1: Multi-corpus support
- [ ] v1.2: Automated evaluation metrics
- [ ] v2.0: API server & web interface
- [ ] v2.1: Multi-model ensemble
- [ ] v3.0: Production deployment framework

---

**Made with ❤️ for semantic grounding in AI**

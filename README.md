# Examiner

**Semantic Grounding Framework for Fine-Tuning Language Models**

Examiner is an open-source framework for building **semantic grounding systems** that align language models with domain-specific knowledge. **You bring your corpus, expertise, and knowledge**—Examiner provides the framework to integrate them.

Build grounded AI systems for any domain: medical diagnosis, legal analysis, scientific research, or specialized expertise. Fine-tune language models on your knowledge base, ensuring responses are anchored in authoritative sources with verifiable citations.

## 🎯 What is Semantic Grounding?

Semantic grounding anchors AI model responses to authoritative information sources—research papers, domain expertise, and real-world context—ensuring accuracy, consistency, and trustworthiness.

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/examiner.git
cd examiner
pip install -r requirements.txt
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

### 3. Capture Expert Knowledge (Optional)

```bash
python lived_experience_dialogue.py \
  --output lived_experience_log.json \
  --interactive
```

### 4. Generate Training Data

```bash
python prepare_training_data.py \
  --corpus data/datasets/corpus \
  --experience lived_experience_log.json \
  --output training_data.json \
  --samples 400
```

### 5. Fine-Tune Your Model

```bash
python train_consolidated.py
```

### 6. Run Inference

```bash
python inference.py \
  --model models/grounding_cycle_1/final \
  --query "Your domain question" \
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

**Tested on**: NVIDIA RTX 2080 (8GB VRAM) with Phi-2 (2.7B)

## 📚 Building Your Corpus

**What is a corpus?** A collection of authoritative documents in your domain:
- Research papers or academic publications
- Technical documentation and manuals  
- Industry reports and standards
- Domain expertise texts
- Trusted reference material

**Recommended sizes:**
- **Small** (10-20 docs): 5-10 min training, basic quality
- **Medium** (50-100 docs): 20-60 min training, good quality
- **Large** (100+ docs): 1-4 hours training, excellent quality

**Finding papers:**
- arXiv: https://arxiv.org/search/
- Google Scholar: https://scholar.google.com/
- Connected Papers: https://www.connectedpapers.com/
- Papers with Code: https://paperswithcode.com/
- Your institution's library (with access)

See [EXAMPLE_CORPUS_SOURCES.md](EXAMPLE_CORPUS_SOURCES.md) for detailed examples of corpus composition and sources.

## 📖 Documentation

- **[Corpus Setup Guide](CORPUS_SETUP.md)** - How to build your corpus (READ FIRST!)
- **[Example Corpus Sources](EXAMPLE_CORPUS_SOURCES.md)** - Example composition and where to find papers
- **[Quick Start](QUICK_START.md)** - Step-by-step setup
- **[Training Guide](TRAINING_CONSOLIDATED.md)** - Fine-tuning details
- **[System Architecture](SYSTEM_SUMMARY.md)** - Technical overview

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
python check_gpu.py        # Check GPU setup
python verify_system.py    # Verify all components
python architecture_auditor.py  # System audit
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
A: Not required, but helpful. Start with published papers and documents—domain experts improve quality.

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

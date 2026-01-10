# Building Your Own Corpus

Examiner is designed to work with **any domain-specific corpus**. This guide shows you how to prepare your own knowledge base for semantic grounding.

## 📚 What is a Corpus?

A corpus is a collection of authoritative documents in your domain:
- Research papers (PDF)
- Technical documentation
- Domain expertise texts
- Industry reports
- Any trusted reference material

## 🚀 Step-by-Step: Build Your Corpus

### Step 1: Gather Documents

Create your corpus directory:
```bash
mkdir -p data/pdfs
# Place your PDF files here
```

**Types of documents to include:**
- Research papers (most valuable for grounding)
- Technical manuals
- Industry standards
- Expert guidelines
- Domain-specific books/chapters

### Step 2: Extract & Index

```bash
python pdf_to_dataset.py \
  --input data/pdfs \
  --output data/datasets/corpus \
  --format json
```

This creates:
- `corpus_index.json` - Searchable index of your corpus
- `extracted_text/` - Raw text from documents

### Step 3: Capture Expert Knowledge (Optional)

Add domain expertise beyond documents:

```bash
python lived_experience_dialogue.py \
  --output lived_experience_log.json \
  --interactive
```

The dialogue captures:
- Expert heuristics and best practices
- Context and nuance not in documents
- Real-world patterns and exceptions

### Step 4: Generate Training Data

Combine corpus + expertise into training data:

```bash
python prepare_training_data.py \
  --corpus data/datasets/corpus \
  --experience lived_experience_log.json \
  --output training_data.json \
  --samples 400
```

### Step 5: Fine-Tune on Your Data

```bash
python train_consolidated.py
```

The model learns to answer questions grounded in **your** corpus and expertise!

---

## 📖 Reference: What We Used

This framework was tested with:

### Corpus Composition
- **104 academic papers** from arXiv and journals
- **Domains**: Machine learning, AI safety, semantic grounding
- **Format**: PDF → Text extraction → Semantic indexing
- **Coverage**: ~250,000 tokens of domain knowledge

### Paper Sources
Examples of the types of papers used:
- "Semantic Grounding in Language Models" (research papers)
- "Knowledge Integration for NLP Systems" (technical)
- "Safe AI Alignment" (domain-specific)
- "Model Architecture and Training" (reference materials)

### Lived Experience
- **1 expert dialogue cycle** capturing domain knowledge
- Topics: Semantic grounding principles, best practices, real-world constraints
- Format: Conversational Q&A with expert responses

### Training Dataset
- **400 training samples** generated from corpus + dialogue
- **50 evaluation samples** for validation
- Format: Instruction-response pairs

### Hardware & Results
- **Model**: microsoft/phi-2 (2.7B)
- **Training**: LoRA fine-tuning (8 hours)
- **Hardware**: RTX 2080 (8GB VRAM)
- **Loss**: 3.016 → 2.657 (↓ 11.9%)

---

## 🎯 Corpus Size Recommendations

| Corpus Size | Training Time | Quality | Effort |
|-----------|--------------|---------|--------|
| **Small** (10-20 docs) | 5-10 min | Basic | Low |
| **Medium** (50-100 docs) | 20-60 min | Good | Medium |
| **Large** (100+ docs) | 1-4 hours | Excellent | High |

Start small, iterate, expand as needed.

---

## 💡 Tips for Building a Great Corpus

### 1. Quality > Quantity
Better to have 20 high-quality, relevant papers than 100 random documents.

### 2. Domain Focus
Keep documents to your specific domain. Mixed domains confuse alignment.

### 3. Diversity
Include different perspectives, methodologies, and approaches within the domain.

### 4. Recent + Classic
Mix cutting-edge papers with foundational/classic works.

### 5. Complementary Sources
Combine:
- Academic papers (rigor)
- Technical documentation (practical)
- Expert knowledge (context)

---

## 📊 Example Corpus Structures

### For ML/AI Domain
```
data/pdfs/
├── attention_mechanisms/
│   ├── attention_is_all_you_need.pdf
│   ├── linformer.pdf
│   └── flash_attention.pdf
├── llms/
│   ├── gpt_papers/
│   ├── llama_papers/
│   └── phi_papers/
├── safety/
│   ├── alignment.pdf
│   ├── interpretability.pdf
│   └── safety_frameworks.pdf
└── grounding/
    ├── knowledge_grounding.pdf
    ├── retrieval_augmented.pdf
    └── semantic_grounding.pdf
```

### For Medical Domain
```
data/pdfs/
├── diagnostics/
├── treatment_protocols/
├── clinical_guidelines/
├── research_papers/
└── safety_standards/
```

### For Legal Domain
```
data/pdfs/
├── statutes/
├── case_law/
├── regulations/
├── contracts/
└── legal_precedents/
```

---

## 🔍 Validating Your Corpus

After extraction, verify:

```bash
# Check corpus index
python -c "
import json
with open('corpus_index.json') as f:
    corpus = json.load(f)
print(f'Documents: {len(corpus[\"documents\"])}')
print(f'Total tokens: {sum(d[\"token_count\"] for d in corpus[\"documents\"])}')
print(f'Topics: {len(corpus[\"topics\"])}')
"
```

Good corpus should have:
- ✅ 50+ documents minimum
- ✅ 100K+ total tokens
- ✅ Multiple topics/sources
- ✅ Clear relevance to domain

---

## 🚀 Next Steps

1. **Gather Documents**: Collect 50-100 PDFs in your domain
2. **Extract**: Run `pdf_to_dataset.py`
3. **Verify**: Check corpus_index.json
4. **Add Expertise**: Optionally run `lived_experience_dialogue.py`
5. **Generate Data**: Run `prepare_training_data.py`
6. **Fine-Tune**: Run `train_consolidated.py`
7. **Test**: Use `inference.py` to verify grounding

---

## 📝 Legal & Ethical Considerations

- ✅ Ensure you have rights to use documents
- ✅ Respect copyright and licensing
- ✅ For academic papers, check open-access policies
- ✅ Cite sources when publicly sharing results
- ✅ Be transparent about corpus composition

---

## ❓ Corpus FAQ

**Q: Can I use documents in other languages?**  
A: Yes, but consistency matters. Keep to one language per corpus.

**Q: What if my corpus is very small (5-10 docs)?**  
A: Still useful! Start small and expand. Grounding works with limited data.

**Q: Can I mix multiple domains?**  
A: Not recommended. Multi-domain corpora reduce alignment quality. Better to build separate models per domain.

**Q: How do I update the corpus?**  
A: Add new PDFs to `data/pdfs/`, rerun `pdf_to_dataset.py`, regenerate training data, and retrain.

**Q: What format files are supported?**  
A: PDF (primary), TXT, JSON, DOCX (with converters). See `pdf_to_dataset.py`.

---

## 🎓 Learning Resources

- [Corpus Linguistics Handbook](https://example.com)
- [Semantic Indexing Methods](https://example.com)
- [Knowledge Grounding Papers](https://example.com)

---

**Your domain expertise + Examiner framework = Grounded AI! 🚀**

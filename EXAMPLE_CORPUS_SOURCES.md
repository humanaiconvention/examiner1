# Example Corpus: What This Framework Was Built With

This document lists the types and examples of papers used to build and test Examiner. Use this as inspiration for building your own domain-specific corpus.

## Corpus Overview

- **Total Documents**: 104 academic papers
- **Domains**: Machine Learning, AI Safety, Semantic Grounding, NLP
- **Format**: PDF → Text extraction → Semantic indexing  
- **Coverage**: ~250,000 tokens of domain knowledge
- **Collection Time**: ~2 weeks
- **Source Mix**: arXiv (40%), Conferences (35%), Journals (25%)

---

## Paper Categories & Sources

### 1. Semantic Grounding & Knowledge Integration (18 papers)

**Core concepts** - How language models ground understanding in external knowledge.

Typical sources:
- "Semantic Grounding in Vision-Language Models"
- "Grounding Language Models in Knowledge Graphs"
- "Retrieval-Augmented Generation for Knowledge-Grounded Dialogue"
- "Knowledge Integration in Transformer Models"
- "Semantic Role Labeling for Knowledge Extraction"

**Where to find similar papers:**
- arXiv: https://arxiv.org/search/?query=semantic+grounding (filter: ML/AI)
- Proceedings: ACL, EMNLP, NAACL
- Journals: Computational Linguistics, NLP journal

---

### 2. Large Language Models & Fine-Tuning (22 papers)

**Foundation** - Understanding LLM architectures and efficient training.

Typical sources:
- "Attention is All You Need" (Transformers)
- "LoRA: Low-Rank Adaptation of Large Language Models"
- "Efficient Fine-Tuning of Language Models" 
- "QLoRA: Quantization-Aware Low-Rank Adaptation"
- "Parameter-Efficient Transfer Learning for NLP"

**Paper examples to search for:**
- "Scaling Laws for Neural Language Models"
- "Training LLMs with Limited Memory"
- "Instruction Following in Large Language Models"
- "Multi-Task Instruction Learning"
- "Domain-Specific Fine-Tuning Strategies"

**Where to find:**
- arXiv: https://arxiv.org/search/?query=language+models+fine-tuning
- Conferences: NeurIPS, ICML, ICLR
- OpenReview: https://openreview.net/

---

### 3. Model Architecture & Efficiency (16 papers)

**Technical depth** - Efficient architectures and optimization techniques.

Typical sources:
- "Phi-2: A Surprisingly Powerful Small Language Model"
- "MobileBERT: Task-Agnostic BERT for Mobile Devices"
- "Mixture of Experts Models"
- "Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- "8-bit Quantization for Neural Networks"

**Paper examples:**
- "Gradient Checkpointing Enables Memory-Efficient Learning"
- "Group Normalization"
- "Layer Normalization vs Batch Normalization"
- "Activation Function Analysis"
- "Distributed Training Strategies"

**Where to find:**
- arXiv: https://arxiv.org/search/?query=efficient+neural+networks
- IEEE Xplore: https://ieeexplore.ieee.org/
- Papers with Code: https://paperswithcode.com/

---

### 4. AI Alignment & Safety (15 papers)

**Safety considerations** - Ensuring models behave as intended.

Typical sources:
- "Constitutional AI: Harmlessness from AI Feedback"
- "Alignment by Agreement on Intermediate Targets"
- "Scalable Oversight of AI Systems"
- "Value Learning and AI Alignment"
- "Interpretability and Safety in Large Models"

**Paper examples:**
- "Identifying and Mitigating Model Bias"
- "Adversarial Robustness in NLP"
- "Explainability in Deep Learning"
- "Model Behavior Documentation"

**Where to find:**
- arXiv: https://arxiv.org/search/?query=ai+alignment+safety
- Alignment Research Center: https://arc.pm
- DeepMind Safety: https://deepmind.com

---

### 5. Knowledge Representation & NLP (18 papers)

**Knowledge engineering** - Extracting and representing domain knowledge.

Typical sources:
- "Knowledge Graphs and Their Applications"
- "Named Entity Recognition and Extraction"
- "Semantic Role Labeling"
- "Coreference Resolution"
- "Question Answering Systems"

**Paper examples:**
- "Entity Linking in Knowledge Bases"
- "Relation Extraction Methods"
- "Discourse Analysis"
- "Document Understanding"
- "Information Extraction from PDFs"

**Where to find:**
- ACL Anthology: https://aclanthology.org/
- arXiv: https://arxiv.org/search/?query=knowledge+extraction+nlp
- SemEval Workshop: https://semeval.github.io/

---

### 6. Dialogue & Conversational AI (10 papers)

**Dialogue systems** - Building interactive, knowledgeable systems.

Typical sources:
- "Dialogue State Tracking"
- "Task-Oriented Dialogue Systems"
- "Conversational AI and Context Management"
- "Multi-Turn Dialogue Understanding"
- "Dialogue Act Classification"

**Paper examples:**
- "Building Goal-Oriented Dialogue Systems"
- "Response Generation in Dialogue"
- "Context Representation in Conversation"
- "Dialogue Evaluation Metrics"

**Where to find:**
- SIGDIAL Conference: https://www.sigdial.org/
- arXiv: https://arxiv.org/search/?query=dialogue+systems
- ACL Anthology: https://aclanthology.org/

---

### 7. Evaluation & Metrics (5 papers)

**Quality assurance** - Measuring model performance and grounding quality.

Typical sources:
- "ROUGE, BLEU, and METEOR Metrics"
- "SQuAD and Reading Comprehension Evaluation"
- "Semantic Similarity Metrics"
- "Human Evaluation of NLP Systems"
- "Benchmark Datasets and Evaluation"

**Paper examples:**
- "Evaluating Grounding Quality"
- "Automatic Evaluation of Open-Ended Responses"
- "Human-Aligned Evaluation Metrics"

**Where to find:**
- arXiv: https://arxiv.org/search/?query=nlp+evaluation+metrics
- ACL Anthology: https://aclanthology.org/

---

## How We Selected Papers

### Selection Criteria

1. **Relevance**: Direct connection to semantic grounding or fine-tuning
2. **Quality**: Published in top venues (NeurIPS, ICML, ICLR, ACL, EMNLP)
3. **Recency**: Published in last 3-5 years (with foundational exceptions)
4. **Diversity**: Multiple perspectives and approaches
5. **Completeness**: Sufficient detail for implementation understanding

### Sources Used

- **arXiv**: 40 papers (recent research, preprints)
- **Conference Proceedings**: 35 papers (ACL, EMNLP, NeurIPS, ICML)
- **Journals**: 25 papers (Computational Linguistics, IEEE TPAMI)
- **Organizational Reports**: 4 papers (OpenAI, DeepMind, Anthropic)

### Collection Process

1. Started with foundational papers (Transformers, BERT, GPT)
2. Expanded to fine-tuning literature (LoRA, QLoRA)
3. Added domain-specific papers (semantic grounding, knowledge integration)
4. Included safety/alignment papers
5. Added dialogue and conversational AI
6. Filled gaps with specialized topics

**Total Collection Time**: ~2 weeks
- Literature review: 5 days
- Paper gathering: 5 days  
- PDF organization: 2 days
- Initial processing: 1 day
- Quality check: 2 days

---

## Building Your Own Corpus

### Start Here

1. **Define your domain**: What area of expertise?
2. **Find key papers**: Start with 5-10 seminal works
3. **Expand systematically**: Use citations, related papers
4. **Quality check**: Is each paper relevant and authoritative?
5. **Extract and process**: Run through `pdf_to_dataset.py`

### Paper Search Strategies

**Google Scholar**:
```
site:scholar.google.com "semantic grounding" language models
```

**arXiv**:
```
https://arxiv.org/search/?query=your+keywords&searchtype=title
```

**Connected Papers**:
```
https://www.connectedpapers.com/ (visual citation graphs)
```

**Semantic Scholar**:
```
https://www.semanticscholar.org/
```

**Papers with Code**:
```
https://paperswithcode.com/ (papers + implementations)
```

### Community Resources

- **Reddit**: r/MachineLearning, r/LanguageModels, r/complexsystems
- **Forums**: LessWrong, AI Alignment Forum

---

## Domain-Specific Corpus Examples

### Machine Learning/AI (What We Used)
- 15-20 foundational/architecture papers
- 20-25 fine-tuning and optimization papers  
- 15-20 knowledge/grounding papers
- 10-15 safety/alignment papers
- Remaining: specialized topics

### Medical/Healthcare Domain
- Medical ontologies and terminology papers
- Clinical guidelines and protocols
- Diagnostic criteria papers
- Treatment effectiveness studies
- Safety and pharmacology references
- Imaging analysis papers

### Legal Domain
- Statutes and legislation (full text)
- Case law and precedents
- Legal commentary and analysis
- Constitutional documents
- Regulatory guidance
- Court procedures

### Finance/Economics
- Economic theory papers
- Market analysis and forecasting
- Risk management literature
- Regulatory frameworks
- Financial reporting standards
- Trading strategies and analysis

---

## Quality Indicators

Your corpus is ready when:

✅ **Quantity**: 50-100 documents minimum  
✅ **Coverage**: Spans key concepts in your domain  
✅ **Authority**: Papers from reputable sources  
✅ **Recency**: Mix of recent and foundational works  
✅ **Diversity**: Multiple perspectives and approaches  
✅ **Completeness**: Full text is available (PDFs)  
✅ **Organization**: Logically organized into folders  

---

## What NOT to Include

❌ Blog posts and medium articles (typically lower quality)  
❌ Unvetted online sources (Wikipedia, personal websites)  
❌ Duplicate papers (same content, different versions)  
❌ Outdated research (unless foundational)  
❌ Papers outside your domain (causes confusion)  
❌ Papers you can't legally use (respect copyright)  

---

## Legal & Ethical Notes

- **Copyright**: Most academic papers can be used for personal/research use
- **arXiv papers**: Generally MIT/Creative Commons licensed (check individually)
- **Paywalled papers**: Use institutional access or contact authors
- **Proper attribution**: Cite your corpus sources when publishing results
- **Transparency**: Document corpus composition for reproducibility

---

## Next Steps for Your Corpus

1. **List papers**: Create a spreadsheet of potential papers
2. **Verify access**: Ensure you can legally use them
3. **Download PDFs**: Organize into `data/pdfs/` folder
4. **Extract text**: Run `python pdf_to_dataset.py`
5. **Validate**: Check `corpus_index.json` generation
6. **Add expertise**: Optional: Run dialogue system
7. **Generate training data**: Run `prepare_training_data.py`
8. **Fine-tune**: Run `train_consolidated.py`
9. **Test**: Use `inference.py` to verify grounding

---

## Resources & Tools

**Paper Management**:
- Zotero (free, open-source)
- Mendeley (free version available)
- Papers (macOS)

**PDF Tools**:
- pdfplumber (Python - extraction)
- PyPDF2 (Python - manipulation)
- Tesseract OCR (image PDFs)

**Citation Tools**:
- Bibtex managers
- Google Scholar export
- Citation extractors

---

**Happy corpus building! 📚🚀**

Your domain knowledge + Examiner = Grounded, expert-aligned AI for your field.

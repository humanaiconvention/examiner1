"""
PDF to Instruction-Tuning Dataset Converter
Extracts text from research PDFs and formats for Qwen2.5-7B fine-tuning
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import fitz
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract and clean text from PDF research articles using PyMuPDF"""
    
    def __init__(self, min_text_length: int = 100):
        self.min_text_length = min_text_length
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file using PyMuPDF (fitz)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted and cleaned text
        """
        try:
            text_blocks = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                if text:
                    text_blocks.append(text)
            
            pdf_document.close()
            
            full_text = "\n".join(text_blocks)
            cleaned_text = self._clean_text(full_text)
            
            if len(cleaned_text) < self.min_text_length:
                logger.warning(f"Insufficient text extracted from {pdf_path}: {len(cleaned_text)} chars")
                return ""
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\nPage \d+\n', '\n', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()


class InstructionDatasetBuilder:
    """Convert extracted text to instruction-tuning format"""
    
    def __init__(self, max_context_length: int = 512):
        self.max_context_length = max_context_length
    
    def create_instruction_samples(self, text: str, document_name: str) -> List[Dict[str, str]]:
        """
        Create instruction-tuning samples from document text
        
        Generates Q&A pairs based on document content
        
        Args:
            text: Extracted document text
            document_name: Name of source document
            
        Returns: 
            List of instruction samples
        """
        samples = []
        
        # Split text into chunks (paragraphs or sections)
        chunks = self._split_into_chunks(text)
        
        for i, chunk in enumerate(chunks):
            # Skip very short chunks
            if len(chunk. split()) < 20:
                continue
            
            # Generate different types of instruction samples
            
            # 1. Summarization task
            samples.append({
                "instruction":  "Summarize the following research content in 2-3 sentences.",
                "input": chunk[: self.max_context_length],
                "output": self._extract_key_sentence(chunk),
                "source": document_name,
                "sample_type": "summarization"
            })
            
            # 2. Question answering (extract key information)
            if self._contains_findings(chunk):
                samples.append({
                    "instruction": "What are the key findings discussed in this text?",
                    "input":  chunk[:self.max_context_length],
                    "output":  self._extract_findings(chunk),
                    "source": document_name,
                    "sample_type": "qa_findings"
                })
            
            # 3. Methodology explanation
            if self._contains_methodology(chunk):
                samples.append({
                    "instruction": "Explain the methodology described in this research.",
                    "input": chunk[: self.max_context_length],
                    "output": self._extract_methodology(chunk),
                    "source": document_name,
                    "sample_type":  "qa_methodology"
                })
            
            # 4. Concept explanation
            key_terms = self._extract_key_terms(chunk)
            if key_terms:
                term = key_terms[0]
                samples.append({
                    "instruction": f"Explain the concept of '{term}' based on the provided context.",
                    "input": chunk[:self.max_context_length],
                    "output": self._extract_term_context(chunk, term),
                    "source": document_name,
                    "sample_type": "concept_explanation"
                })
        
        return samples
    
    def _split_into_chunks(self, text: str, chunk_size: int = 800) -> List[str]:
        """Split text into semantic chunks"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            
            if current_length + para_length > chunk_size and current_chunk:
                chunks. append(' '.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else: 
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _extract_key_sentence(self, text: str) -> str:
        """Extract most informative sentence as summary"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip().split()) > 5]
        
        # Simple heuristic: prefer sentences with research-related keywords
        keywords = ['results', 'findings', 'concluded', 'demonstrated', 'showed', 'indicates', 'suggests']
        
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                return sentence + '.'
        
        # Return first substantial sentence
        return sentences[0] + '.' if sentences else text[:200]
    
    def _contains_findings(self, text: str) -> bool:
        """Check if text contains research findings"""
        keywords = ['results', 'findings', 'concluded', 'demonstrated', 'observed', 'discovered']
        return any(kw in text.lower() for kw in keywords)
    
    def _extract_findings(self, text: str) -> str:
        """Extract findings-related content"""
        sentences = text.split('.')
        findings = []
        
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in ['result', 'finding', 'concluded', 'demonstrated', 'showed']):
                findings.append(sentence. strip())
        
        return '. '.join(findings[: 2]) + '.' if findings else text[:200]
    
    def _contains_methodology(self, text: str) -> bool:
        """Check if text contains methodology description"""
        keywords = ['method', 'approach', 'procedure', 'technique', 'experiment', 'analysis']
        return any(kw in text.lower() for kw in keywords)
    
    def _extract_methodology(self, text: str) -> str:
        """Extract methodology-related content"""
        sentences = text.split('.')
        methodology = []
        
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in ['method', 'approach', 'procedure', 'used', 'applied']):
                methodology.append(sentence.strip())
        
        return '. '.join(methodology[: 2]) + '.' if methodology else text[:200]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract potential key terms (capitalized phrases, technical terms)"""
        # Simple extraction: capitalized multi-word terms
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        terms = re.findall(pattern, text)
        
        # Filter common non-technical terms
        stopwords = {'The', 'This', 'These', 'Those', 'In', 'On', 'At', 'To', 'For'}
        terms = [t for t in terms if t not in stopwords]
        
        return list(set(terms))[:5]  # Return unique terms
    
    def _extract_term_context(self, text: str, term: str) -> str:
        """Extract context around a specific term"""
        sentences = text.split('.')
        
        for sentence in sentences:
            if term in sentence: 
                return sentence.strip() + '.'
        
        return text[:200]


def process_pdf_directory(
    pdf_dir: str,
    output_dir: str,
    test_split: float = 0.1,
    max_samples_per_doc: int = 50
) -> Tuple[DatasetDict, Dict]: 
    """
    Process directory of PDFs and create HuggingFace dataset
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save processed dataset
        test_split:  Fraction of data for test set
        max_samples_per_doc: Maximum samples to generate per document
        
    Returns: 
        Tuple of (DatasetDict, statistics dict)
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    extractor = PDFExtractor()
    dataset_builder = InstructionDatasetBuilder()
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files: 
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process PDFs
    all_samples = []
    doc_stats = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        logger.info(f"Processing {pdf_path. name}")
        
        # Extract text
        text = extractor.extract_text(str(pdf_path))
        
        if not text:
            logger.warning(f"Skipping {pdf_path. name}:  no text extracted")
            continue
        
        # Generate instruction samples
        samples = dataset_builder.create_instruction_samples(text, pdf_path.stem)
        
        # Limit samples per document
        samples = samples[:max_samples_per_doc]
        
        all_samples.extend(samples)
        
        # Track statistics
        doc_stats.append({
            'document': pdf_path.name,
            'text_length': len(text),
            'num_samples': len(samples),
            'word_count': len(text.split())
        })
        
        logger.info(f"  Generated {len(samples)} samples from {pdf_path.name}")
    
    if not all_samples:
        raise ValueError("No samples generated from PDFs")
    
    logger.info(f"Total samples generated: {len(all_samples)}")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(all_samples)
    
    # Split into train/test
    split_dataset = dataset.train_test_split(test_size=test_split, seed=42)
    
    # Save dataset
    dataset_path = output_dir / "instruction_dataset"
    split_dataset.save_to_disk(str(dataset_path))
    logger.info(f"Dataset saved to {dataset_path}")
    
    # Calculate statistics
    sample_types = Counter([s['sample_type'] for s in all_samples])
    
    statistics = {
        'num_documents': len(doc_stats),
        'total_samples': len(all_samples),
        'train_samples': len(split_dataset['train']),
        'test_samples': len(split_dataset['test']),
        'sample_types': dict(sample_types),
        'avg_samples_per_doc': len(all_samples) / len(doc_stats),
        'document_stats': doc_stats
    }
    
    # Save statistics
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_path}")
    logger.info(f"Dataset split:  {len(split_dataset['train'])} train, {len(split_dataset['test'])} test")
    
    return split_dataset, statistics


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to instruction-tuning dataset")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="articles",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets",
        help="Directory to save processed dataset"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data for test set (default: 0.1)"
    )
    parser.add_argument(
        "--max_samples_per_doc",
        type=int,
        default=50,
        help="Maximum samples per document (default: 50)"
    )
    
    args = parser.parse_args()
    
    try:
        dataset, stats = process_pdf_directory(
            args.pdf_dir,
            args.output_dir,
            args.test_split,
            args.max_samples_per_doc
        )
        
        print("\n" + "="*50)
        print("Dataset Preparation Complete")
        print("="*50)
        print(f"Documents processed: {stats['num_documents']}")
        print(f"Total samples:  {stats['total_samples']}")
        print(f"Train samples: {stats['train_samples']}")
        print(f"Test samples: {stats['test_samples']}")
        print(f"\nSample type distribution:")
        for sample_type, count in stats['sample_types'].items():
            print(f"  {sample_type}: {count}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
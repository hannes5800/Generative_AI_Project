from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
import re
import unicodedata


def extract_pdf(path: str | Path) -> str:
    """Extract and return all text from a PDF file"""

    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_text(text: str) -> str:
    """Clean PDF-extracted text"""
    
    # Fix common PDF ligatures and Unicode issues
    replacements = {
        'ﬁ': 'fi',      
        'ﬀ': 'ff',      
        'ﬂ': 'fl',      
        'ﬃ': 'ffi',     
        'ﬄ': 'ffl',     
        'ﬅ': 'ft',     
        '—': '-',       
        '–': '-',       
        '"': '"',         
        '"': '"',
        ''': "'",
        ''': "'",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove non-printable characters but keep basic structure
    text = ''.join(char for char in text 
                   if unicodedata.category(char)[0] != 'C' or char == '\n')
    
    # Remove email addresses (optional - remove if you want to keep them)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Remove common footer/header patterns
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n\d+\n', '\n', text)  # page numbers
    
    # Remove table of contents dots (1.1 Basic Concepts . . . . . . . . 5)
    text = re.sub(r'(\d+\.\d+)\s+(.+?)\s+\.{2,}\s*\d+', r'\1 \2', text)
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # multiple newlines → double
    text = re.sub(r' {2,}', ' ', text)      # multiple spaces → single
    text = re.sub(r' +\n', '\n', text)      # trailing spaces
    text = re.sub(r'\n +', '\n', text)      # leading spaces after newline
    
    # Remove lines that are just dots or special chars
    lines = text.split('\n')
    lines = [line for line in lines 
             if line.strip() and not re.match(r'^[\s\.\-_]+$', line)]
    text = '\n'.join(lines)
    
    return text.strip()


def load_corpus(base_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Load all PDFs/Metadata from base_dir, return list of doc dicts"""

    docs = []
    
    for file_path in Path(base_dir).rglob("*"):
        if file_path.suffix == ".pdf":
            text = extract_pdf(file_path)
        # For now, just read PDFs (other file options could be added later)
        else:
            continue
            
        # Clean text
        text = clean_text(text)
        
        # Build metadata from path
        doc_class = file_path.parent.name  # e.g. "bitcoin"
        doc_id = file_path.stem             # e.g. "btc_whitepaper"
        
        docs.append({
            "document_class": doc_class,
            "project_id": doc_id,
            "text": text,
            "source_path": str(file_path)
        })
    
    return docs

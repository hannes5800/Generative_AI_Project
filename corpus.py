### To load in corpus as dictionary call load_corpus()

from pathlib import Path
import fitz  # PyMuPDF
import re
import unicodedata

# Extracting PDF
def extract_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Cleaning PDF text
def clean_text(text):
    """Aggressively clean PDF-extracted text"""
    
    # 1. Fix common PDF ligatures and Unicode issues
    replacements = {
        'ﬁ': 'fi',      # fi ligature
        'ﬀ': 'ff',      # ff ligature
        'ﬂ': 'fl',      # fl ligature
        'ﬃ': 'ffi',     # ffi ligature
        'ﬄ': 'ffl',     # ffl ligature
        'ﬅ': 'ft',      # ft ligature
        '—': '-',        # em dash
        '–': '-',        # en dash
        '"': '"',        # fancy quotes
        '"': '"',
        ''': "'",
        ''': "'",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 2. Remove non-printable characters but keep basic structure
    text = ''.join(char for char in text 
                   if unicodedata.category(char)[0] != 'C' or char == '\n')
    
    # 3. Remove email addresses (optional - remove if you want to keep them)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # 4. Remove common footer/header patterns
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n\d+\n', '\n', text)  # page numbers
    
    # 5. Remove table of contents dots (1.1 Basic Concepts . . . . . . . . 5)
    text = re.sub(r'(\d+\.\d+)\s+(.+?)\s+\.{2,}\s*\d+', r'\1 \2', text)
    
    # 6. Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # multiple newlines → double
    text = re.sub(r' {2,}', ' ', text)      # multiple spaces → single
    text = re.sub(r' +\n', '\n', text)      # trailing spaces
    text = re.sub(r'\n +', '\n', text)      # leading spaces after newline
    
    # 7. Remove lines that are just dots or special chars
    lines = text.split('\n')
    lines = [line for line in lines 
             if line.strip() and not re.match(r'^[\s\.\-_]+$', line)]
    text = '\n'.join(lines)
    
    return text.strip()

# Main function to load in corpus as dictionary
def load_corpus(base_dir="data"):
    """Load all PDFs/Metadata from base_dir, return list of doc dicts"""
    docs = []
    
    for file_path in Path(base_dir).rglob("*"):
        if file_path.suffix == ".pdf":
            text = extract_pdf(file_path)
        # For now we go just with PDF, other file options could be added    
        else:
            continue
            
        # Clean text
        text = clean_text(text)
        
        # Build metadata from path
        project_id = file_path.parent.name  # e.g., "bitcoin"
        doc_id = file_path.stem  # e.g., "btc_whitepaper"
        
        docs.append({
            "document_class": project_id,
            "project_id": doc_id,
            "text": text,
            "source_path": str(file_path)
        })
    
    return docs
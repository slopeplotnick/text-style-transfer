import re
import os
from typing import List, Dict

def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_paragraphs(text: str) -> List[str]:
    """
    Splits text into paragraphs based on double newlines.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using simple regex.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def detect_section(text: str, current_section: str) -> str:
    """
    Simple heuristic to detect academic section headers.
    """
    # Common academic headers
    headers = {
        "abstract": r"^\s*(abstract|summary)\s*$",
        "introduction": r"^\s*(1\.?|i\.?)?\s*introduction\s*$",
        "related_work": r"^\s*(2\.?|ii\.?)?\s*(related work|background|literature review)\s*$",
        "methodology": r"^\s*(3\.?|iii\.?)?\s*(method|methodology|approach|proposed scheme)\s*$",
        "experiments": r"^\s*(4\.?|iv\.?)?\s*(experiments|results|evaluation|performance)\s*$",
        "discussion": r"^\s*(5\.?|v\.?)?\s*(discussion|analysis)\s*$",
        "conclusion": r"^\s*(6\.?|vi\.?)?\s*(conclusion|concluding remarks)\s*$",
    }
    
    first_line = text.split('\n')[0].lower().strip()
    # Check if the paragraph starts with a header
    for section, pattern in headers.items():
        if re.match(pattern, first_line):
            return section
            
    return current_section

def get_journal_name(file_path: str) -> str:
    """
    Extracts journal name from the parent directory of the file.
    e.g. /data/Nature/paper.txt -> Nature
    """
    return os.path.basename(os.path.dirname(os.path.abspath(file_path)))

def process_file_to_chunks(file_path: str, chunk_size: int = 3) -> List[Dict]:
    """
    Reads a file and processes it into chunks, preserving Paragraph, Section AND Journal context.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text)
    processed_data = []

    current_section = "general"
    journal_name = get_journal_name(file_path)

    for para in paragraphs:
        # Update section if header is detected
        current_section = detect_section(para, current_section)

        sentences = split_into_sentences(para)
        if not sentences:
            continue

        # If paragraph is just a header, skip creating chunks but keep section update
        if len(sentences) == 1 and len(sentences[0].split()) < 5:
            continue

        for i in range(0, len(sentences), chunk_size):
            chunk_sents = sentences[i:i + chunk_size]
            chunk_text = " ".join(chunk_sents)

            if len(chunk_text) < 10:
                continue

            processed_data.append({
                "chunk_text": chunk_text,
                "paragraph_text": para,
                "section_type": current_section,
                "journal": journal_name, # Key addition
                "source": os.path.basename(file_path)
            })

    return processed_data

def process_file_to_paragraphs(file_path: str) -> List[Dict]:
    """
    Reads a file and processes it into paragraphs (for paragraph-level indexing).
    Returns list of paragraph data with metadata.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text)
    processed_data = []

    current_section = "general"
    journal_name = get_journal_name(file_path)

    for para in paragraphs:
        # Update section if header is detected
        current_section = detect_section(para, current_section)

        # Skip very short paragraphs or headers
        sentences = split_into_sentences(para)
        if not sentences or len(para.strip()) < 20:
            continue

        # If paragraph is just a header, skip
        if len(sentences) == 1 and len(sentences[0].split()) < 5:
            continue

        processed_data.append({
            "paragraph_text": para,
            "section_type": current_section,
            "journal": journal_name,
            "source": os.path.basename(file_path)
        })

    return processed_data
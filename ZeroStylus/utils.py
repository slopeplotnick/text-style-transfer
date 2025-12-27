"""
Utility functions for ZeroStylus framework.
"""

import re
import numpy as np
from typing import List, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.

    Args:
        text: Input text string

    Returns:
        List of sentence strings
    """
    sentences = sent_tokenize(text)
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on double newlines.

    Args:
        text: Input text string

    Returns:
        List of paragraph strings
    """
    # Split by double newlines or more
    paragraphs = re.split(r'\n\s*\n', text)
    # Clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for embeddings.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute similarity matrix
    similarity = np.dot(normalized, normalized.T)
    return similarity


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency-based approach.

    Args:
        text: Input text
        top_k: Number of top keywords to extract

    Returns:
        List of keywords
    """
    # Simple tokenization
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Remove common stop words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you',
                  'all', 'can', 'her', 'was', 'one', 'our', 'out',
                  'this', 'that', 'with', 'from', 'have', 'has',
                  'had', 'they', 'were', 'been', 'which', 'their'}

    words = [w for w in words if w not in stop_words]

    # Count frequencies
    from collections import Counter
    word_freq = Counter(words)

    # Get top keywords
    keywords = [word for word, _ in word_freq.most_common(top_k)]
    return keywords


def chunk_text(text: str, max_length: int = 2000) -> List[str]:
    """
    Chunk long text into smaller segments while preserving sentence boundaries.

    Args:
        text: Input text
        max_length: Maximum length of each chunk

    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_length and current_chunk:
            # Start a new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1] range.

    Args:
        scores: List of scores

    Returns:
        Normalized scores
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.5] * len(scores)

    normalized = [(s - min_score) / (max_score - min_score) for s in scores]
    return normalized


def calculate_embedding_distance(emb1: np.ndarray, emb2: np.ndarray,
                                 metric: str = 'euclidean') -> float:
    """
    Calculate distance between two embeddings.

    Args:
        emb1: First embedding
        emb2: Second embedding
        metric: Distance metric ('euclidean' or 'cosine')

    Returns:
        Distance value
    """
    if metric == 'euclidean':
        return np.linalg.norm(emb1 - emb2)
    elif metric == 'cosine':
        return 1 - cosine_similarity(emb1, emb2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

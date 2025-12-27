"""
Phase 1: Hierarchical Template Acquisition

This module implements the first phase of ZeroStylus framework:
1. Sentence Pattern Extraction using DBSCAN clustering
2. Paragraph Structure Modeling through hierarchical encoding
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from config_loader import get_config
from utils import (
    split_into_sentences,
    split_into_paragraphs,
    calculate_embedding_distance,
    compute_similarity_matrix
)


class TemplateExtractor:
    """
    Extract sentence and paragraph templates from reference style texts.

    This class implements Phase 1 of the ZeroStylus framework:
    - Sentence pattern extraction using DBSCAN clustering
    - Paragraph structure modeling through embedding aggregation
    """

    def __init__(
        self,
        encoder_model: Optional[str] = None,
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: Optional[int] = None,
        paragraph_threshold: Optional[float] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the template extractor.

        Args:
            encoder_model: Name of the sentence transformer model (defaults to config)
            dbscan_eps: DBSCAN epsilon parameter for sentence clustering (defaults to config)
            dbscan_min_samples: DBSCAN minimum samples parameter (defaults to config)
            paragraph_threshold: Threshold for paragraph template uniqueness (defaults to config)
            config_path: Path to config file (optional)
        """
        # Load configuration
        config = get_config(config_path)

        # Use provided values or fall back to config
        encoder_model = encoder_model or config.encoder_model
        dbscan_eps = dbscan_eps if dbscan_eps is not None else config.dbscan_eps
        dbscan_min_samples = dbscan_min_samples if dbscan_min_samples is not None else config.dbscan_min_samples
        paragraph_threshold = paragraph_threshold if paragraph_threshold is not None else config.paragraph_threshold

        self.encoder = SentenceTransformer(encoder_model)
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.paragraph_threshold = paragraph_threshold

        # Template repositories
        self.sentence_templates = []  # Γs: List of sentence template embeddings
        self.sentence_examples = []   # Example sentences for each template
        self.paragraph_templates = [] # Γp: List of paragraph template embeddings
        self.paragraph_examples = []  # Example paragraphs for each template

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences into dense vector representations.

        Args:
            sentences: List of sentence strings

        Returns:
            Array of sentence embeddings, shape (n_sentences, embedding_dim)
        """
        embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        return embeddings

    def extract_sentence_patterns(
        self,
        documents: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract sentence patterns from reference documents using DBSCAN clustering.

        This implements Step 1.1: Sentence Pattern Extraction
        Formula: e_j = π_enc(s_j)

        Args:
            documents: List of reference text documents

        Returns:
            Tuple of (template_embeddings, template_examples)
        """
        # Step 1: Extract all sentences from documents
        all_sentences = []
        for doc in documents:
            sentences = split_into_sentences(doc)
            all_sentences.extend(sentences)

        if not all_sentences:
            return np.array([]), []

        print(f"Extracted {len(all_sentences)} sentences from {len(documents)} documents")

        # Step 2: Encode sentences using π_enc
        sentence_embeddings = self.encode_sentences(all_sentences)

        # Step 3: Apply DBSCAN clustering to identify recurrent structures
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric='cosine'
        )
        cluster_labels = clustering.fit_predict(sentence_embeddings)

        # Step 4: Extract cluster centroids as sentence templates
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise points

        templates = []
        examples = []

        for label in unique_labels:
            # Get all sentences in this cluster
            cluster_mask = cluster_labels == label
            cluster_embeddings = sentence_embeddings[cluster_mask]
            cluster_sentences = [s for s, m in zip(all_sentences, cluster_mask) if m]

            # Compute centroid as template τ_s
            centroid = np.mean(cluster_embeddings, axis=0)
            templates.append(centroid)

            # Store example sentence (closest to centroid)
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
            best_idx = np.argmin(distances)
            examples.append(cluster_sentences[best_idx])

        templates = np.array(templates) if templates else np.array([])
        print(f"Extracted {len(templates)} sentence templates from {len(unique_labels)} clusters")

        self.sentence_templates = templates
        self.sentence_examples = examples

        return templates, examples

    def extract_paragraph_patterns(
        self,
        documents: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract paragraph structure patterns through hierarchical encoding.

        This implements Step 1.2: Paragraph Structure Modeling
        Formula: e_p = π_enc([e_1, ..., e_m])

        Args:
            documents: List of reference text documents

        Returns:
            Tuple of (paragraph_templates, paragraph_examples)
        """
        # Step 1: Extract paragraphs from documents
        all_paragraphs = []
        for doc in documents:
            paragraphs = split_into_paragraphs(doc)
            all_paragraphs.extend(paragraphs)

        if not all_paragraphs:
            return np.array([]), []

        print(f"Extracted {len(all_paragraphs)} paragraphs from {len(documents)} documents")

        # Step 2: Encode each paragraph through hierarchical encoding
        paragraph_embeddings = []

        for para in all_paragraphs:
            # Get sentences in paragraph
            sentences = split_into_sentences(para)

            if not sentences:
                continue

            # Encode sentences: [e_1, ..., e_m]
            sentence_embs = self.encode_sentences(sentences)

            # Aggregate through hierarchical encoding
            # e_p = π_enc([e_1, ..., e_m])
            # We use the encoder to create a paragraph-level representation
            para_embedding = self.encoder.encode(para, convert_to_numpy=True)
            paragraph_embeddings.append(para_embedding)

        paragraph_embeddings = np.array(paragraph_embeddings)

        # Step 3: Dynamic template repository with threshold-controlled expansion
        # Add template only if: min_{τ_p ∈ Γ_p} ||e_p - τ_p|| > ε
        templates = []
        examples = []

        for i, (para_emb, para_text) in enumerate(zip(paragraph_embeddings, all_paragraphs)):
            # Check if this paragraph is sufficiently different from existing templates
            if not templates:
                # First template
                templates.append(para_emb)
                examples.append(para_text)
            else:
                # Compute minimum distance to existing templates
                distances = [
                    calculate_embedding_distance(para_emb, template, metric='euclidean')
                    for template in templates
                ]
                min_distance = min(distances)

                # Add as new template if distance exceeds threshold
                if min_distance > self.paragraph_threshold:
                    templates.append(para_emb)
                    examples.append(para_text)

        templates = np.array(templates) if templates else np.array([])
        print(f"Extracted {len(templates)} paragraph templates (threshold: {self.paragraph_threshold})")

        self.paragraph_templates = templates
        self.paragraph_examples = examples

        return templates, examples

    def extract_all_templates(self, documents: List[str]) -> Dict:
        """
        Extract both sentence and paragraph templates from reference documents.

        Args:
            documents: List of reference text documents

        Returns:
            Dictionary containing all extracted templates
        """
        print("=== Phase 1: Hierarchical Template Acquisition ===")

        # Step 1.1: Sentence Pattern Extraction
        print("\n[1.1] Extracting sentence patterns...")
        sent_templates, sent_examples = self.extract_sentence_patterns(documents)

        # Step 1.2: Paragraph Structure Modeling
        print("\n[1.2] Modeling paragraph structures...")
        para_templates, para_examples = self.extract_paragraph_patterns(documents)

        return {
            'sentence_templates': sent_templates,
            'sentence_examples': sent_examples,
            'paragraph_templates': para_templates,
            'paragraph_examples': para_examples,
            'num_sentence_templates': len(sent_templates),
            'num_paragraph_templates': len(para_templates)
        }

    def get_templates(self) -> Dict:
        """
        Get the current template repositories.

        Returns:
            Dictionary containing Γs and Γp
        """
        return {
            'Γs': self.sentence_templates,
            'Γs_examples': self.sentence_examples,
            'Γp': self.paragraph_templates,
            'Γp_examples': self.paragraph_examples
        }

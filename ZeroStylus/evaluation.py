"""
Evaluation Module for ZeroStylus

This module implements the tri-axial evaluation metrics:
- X: Style Consistency (paragraph-level embedding similarity)
- Y: Content Preservation (BLEURT + keyword retention)
- Z: Expression Quality (human preference + LLM benchmarks)

Average Score: A = (X + Y + Z) / 3
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from sentence_transformers import SentenceTransformer
from config_loader import get_config
from utils import (
    cosine_similarity,
    extract_keywords,
    normalize_scores
)


class TriaxialEvaluator:
    """
    Evaluate style transfer quality using three-dimensional metrics.

    Metrics:
    - x: Style Consistency - measures alignment with reference style
    - y: Content Preservation - measures semantic similarity to original
    - z: Expression Quality - measures naturalness and fluency
    """

    def __init__(
        self,
        encoder_model: Optional[str] = None,
        use_bleurt: Optional[bool] = None,
        llm_evaluator: Optional[Callable] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the evaluator.

        Args:
            encoder_model: Sentence transformer model for embeddings (defaults to config)
            use_bleurt: Whether to use BLEURT (defaults to config)
            llm_evaluator: Optional LLM function for quality evaluation
            config_path: Path to config.yaml file (optional)
        """
        # Load configuration
        config = get_config(config_path)

        # Use provided values or fall back to config
        encoder_model = encoder_model or config.encoder_model
        use_bleurt = use_bleurt if use_bleurt is not None else config.use_bleurt

        self.encoder = SentenceTransformer(encoder_model)
        self.use_bleurt = use_bleurt
        self.llm_evaluator = llm_evaluator

        # Store config for accessing weights
        self.config = config

        # Try to load BLEURT if requested
        self.bleurt_scorer = None
        if use_bleurt:
            try:
                from bleurt import score as bleurt_score
                self.bleurt_scorer = bleurt_score.BleurtScorer()
                print("BLEURT scorer loaded successfully")
            except ImportError:
                print("Warning: BLEURT not available, using embedding similarity instead")
                self.bleurt_scorer = None

    def evaluate_style_consistency(
        self,
        transformed_text: str,
        reference_texts: List[str]
    ) -> float:
        """
        Evaluate style consistency (X metric).

        Measures paragraph-level embedding similarity between transformed text
        and reference style texts.

        Formula: x = similarity(embedding(transformed), embedding(reference))

        Args:
            transformed_text: The style-transferred output
            reference_texts: Reference texts representing target style

        Returns:
            Style consistency score (0-10 scale)
        """
        # Encode transformed text
        transformed_emb = self.encoder.encode(
            transformed_text,
            convert_to_numpy=True
        )

        # Encode reference texts
        reference_embs = self.encoder.encode(
            reference_texts,
            convert_to_numpy=True
        )

        # Compute average similarity with all reference texts
        similarities = [
            cosine_similarity(transformed_emb, ref_emb)
            for ref_emb in reference_embs
        ]

        avg_similarity = np.mean(similarities)

        # Convert to 0-10 scale
        # Cosine similarity ranges from [-1, 1], but typically [0, 1] for text
        score = avg_similarity * 10

        return max(0, min(10, score))  # Clamp to [0, 10]

    def evaluate_content_preservation(
        self,
        transformed_text: str,
        original_text: str
    ) -> float:
        """
        Evaluate content preservation (Y metric).

        Combines BLEURT scores with keyword retention recall.

        Formula: y = α * BLEURT(transformed, original) + β * keyword_retention

        Args:
            transformed_text: The style-transferred output
            original_text: The original source text

        Returns:
            Content preservation score (0-10 scale)
        """
        # Method 1: BLEURT score (if available)
        if self.bleurt_scorer is not None:
            bleurt_score = self._compute_bleurt(transformed_text, original_text)
            bleurt_component = bleurt_score
        else:
            # Fallback: Use embedding similarity
            orig_emb = self.encoder.encode(original_text, convert_to_numpy=True)
            trans_emb = self.encoder.encode(transformed_text, convert_to_numpy=True)
            bleurt_component = cosine_similarity(orig_emb, trans_emb)

        # Method 2: Keyword retention recall
        keyword_retention = self._compute_keyword_retention(
            transformed_text,
            original_text
        )

        # Combine scores using weights from config
        combined_score = (
            self.config.bleurt_weight * bleurt_component +
            self.config.keyword_weight * keyword_retention
        )

        # Convert to 0-10 scale
        score = combined_score * 10

        return max(0, min(10, score))

    def _compute_bleurt(self, transformed_text: str, original_text: str) -> float:
        """
        Compute BLEURT score between transformed and original text.

        Args:
            transformed_text: Transformed text
            original_text: Original text

        Returns:
            BLEURT score normalized to [0, 1]
        """
        if self.bleurt_scorer is None:
            return 0.0

        try:
            scores = self.bleurt_scorer.score(
                references=[original_text],
                candidates=[transformed_text]
            )
            # BLEURT scores are typically in [-1, 1], normalize to [0, 1]
            normalized = (scores[0] + 1) / 2
            return max(0, min(1, normalized))
        except Exception as e:
            print(f"BLEURT computation failed: {e}")
            return 0.0

    def _compute_keyword_retention(
        self,
        transformed_text: str,
        original_text: str,
        top_k: Optional[int] = None
    ) -> float:
        """
        Compute keyword retention recall.

        Measures what fraction of important keywords from the original text
        are preserved in the transformed text.

        Args:
            transformed_text: Transformed text
            original_text: Original text
            top_k: Number of top keywords to extract (defaults to config)

        Returns:
            Keyword retention score [0, 1]
        """
        # Use provided value or fall back to config
        if top_k is None:
            top_k = self.config.keyword_top_k

        # Extract keywords from original and transformed texts
        original_keywords = set(extract_keywords(original_text, top_k=top_k))
        transformed_keywords = set(extract_keywords(transformed_text, top_k=top_k))

        if not original_keywords:
            return 1.0  # No keywords to preserve

        # Compute recall: how many original keywords are retained
        retained = original_keywords.intersection(transformed_keywords)
        retention_rate = len(retained) / len(original_keywords)

        return retention_rate

    def evaluate_expression_quality(
        self,
        transformed_text: str,
        reference_texts: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate expression quality (Z metric).

        Combines human preference checks with LLM benchmark standards.

        Args:
            transformed_text: The style-transferred output
            reference_texts: Optional reference texts for comparison

        Returns:
            Expression quality score (0-10 scale)
        """
        # Method 1: LLM-based evaluation (if available)
        if self.llm_evaluator is not None:
            llm_score = self._evaluate_with_llm(transformed_text, reference_texts)
            return llm_score

        # Method 2: Heuristic-based evaluation
        quality_score = self._evaluate_quality_heuristics(transformed_text)

        return quality_score

    def _evaluate_with_llm(
        self,
        transformed_text: str,
        reference_texts: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate text quality using LLM.

        Args:
            transformed_text: Text to evaluate
            reference_texts: Optional reference texts

        Returns:
            Quality score (0-10 scale)
        """
        prompt = f"""Evaluate the following text for naturalness, fluency, and grammatical correctness.

**Text to Evaluate:**
{transformed_text}

**Evaluation Criteria:**
1. Grammatical correctness (0-10)
2. Fluency and readability (0-10)
3. Naturalness of expression (0-10)

Provide ONLY a single numeric score from 0-10 representing the overall quality.

**Score:**"""

        try:
            response = self.llm_evaluator(prompt)
            # Extract numeric score
            import re
            match = re.search(r'\d+\.?\d*', response)
            if match:
                score = float(match.group())
                return max(0, min(10, score))
        except Exception as e:
            print(f"LLM evaluation failed: {e}")

        # Fallback to heuristics
        return self._evaluate_quality_heuristics(transformed_text)

    def _evaluate_quality_heuristics(self, text: str) -> float:
        """
        Evaluate text quality using heuristic metrics.

        Considers:
        - Sentence length variation
        - Vocabulary diversity
        - Basic grammatical patterns

        Args:
            text: Text to evaluate

        Returns:
            Quality score (0-10 scale)
        """
        from utils import split_into_sentences
        import re

        sentences = split_into_sentences(text)

        if not sentences:
            return 0.0

        # Metric 1: Sentence length variation (not too uniform, not too varied)
        lengths = [len(s.split()) for s in sentences]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Ideal: avg 15-25 words, std 5-10 words
        length_score = 1.0
        if avg_length < 10 or avg_length > 30:
            length_score *= 0.8
        if std_length < 3 or std_length > 15:
            length_score *= 0.9

        # Metric 2: Vocabulary diversity (type-token ratio)
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        diversity_score = min(1.0, ttr * 2)  # Scale to [0, 1]

        # Metric 3: Readability (simple check for sentence complexity)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        readability_score = 1.0
        if avg_sentence_length > 40:  # Very long sentences
            readability_score = 0.7
        elif avg_sentence_length < 5:  # Very short sentences
            readability_score = 0.8

        # Combine scores using weights from config
        combined = (
            self.config.length_variation_weight * length_score +
            self.config.vocabulary_diversity_weight * diversity_score +
            self.config.readability_weight * readability_score
        )

        # Convert to 0-10 scale
        return combined * 10

    def evaluate_all(
        self,
        transformed_text: str,
        original_text: str,
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Perform complete tri-axial evaluation.

        Args:
            transformed_text: The style-transferred output
            original_text: The original source text
            reference_texts: Reference texts representing target style

        Returns:
            Dictionary with scores for X, Y, Z, and average A
        """
        print("\n=== Tri-Axial Evaluation ===")

        # X: Style Consistency
        print("[X] Evaluating style consistency...")
        x_score = self.evaluate_style_consistency(transformed_text, reference_texts)
        print(f"    Style Consistency: {x_score:.2f}/10")

        # Y: Content Preservation
        print("[Y] Evaluating content preservation...")
        y_score = self.evaluate_content_preservation(transformed_text, original_text)
        print(f"    Content Preservation: {y_score:.2f}/10")

        # Z: Expression Quality
        print("[Z] Evaluating expression quality...")
        z_score = self.evaluate_expression_quality(transformed_text, reference_texts)
        print(f"    Expression Quality: {z_score:.2f}/10")

        # Average score
        average = (x_score + y_score + z_score) / 3
        print(f"\n[Average] Overall Score: {average:.2f}/10")

        return {
            'style_consistency': x_score,
            'content_preservation': y_score,
            'expression_quality': z_score,
            'average_score': average,
            'vector': [x_score, y_score, z_score]
        }

    def batch_evaluate(
        self,
        samples: List[Dict[str, str]],
        reference_texts: List[str]
    ) -> Dict[str, List[float]]:
        """
        Evaluate multiple samples and return aggregated results.

        Args:
            samples: List of dicts with 'transformed' and 'original' keys
            reference_texts: Reference texts for style evaluation

        Returns:
            Dictionary with lists of scores for each metric
        """
        results = {
            'style_consistency': [],
            'content_preservation': [],
            'expression_quality': [],
            'average_score': []
        }

        print(f"\nEvaluating {len(samples)} samples...")

        for i, sample in enumerate(samples):
            print(f"\n[Sample {i+1}/{len(samples)}]")

            scores = self.evaluate_all(
                transformed_text=sample['transformed'],
                original_text=sample['original'],
                reference_texts=reference_texts
            )

            results['style_consistency'].append(scores['style_consistency'])
            results['content_preservation'].append(scores['content_preservation'])
            results['expression_quality'].append(scores['expression_quality'])
            results['average_score'].append(scores['average_score'])

        # Compute overall statistics
        print("\n=== Overall Statistics ===")
        for metric, scores in results.items():
            print(f"{metric}:")
            print(f"  Mean: {np.mean(scores):.2f}")
            print(f"  Std:  {np.std(scores):.2f}")
            print(f"  Min:  {np.min(scores):.2f}")
            print(f"  Max:  {np.max(scores):.2f}")

        return results

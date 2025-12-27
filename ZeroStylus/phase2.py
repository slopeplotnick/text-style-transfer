"""
Phase 2: Template-Guided Generation

This module implements the second phase of ZeroStylus framework:
1. Multi-Granular Template Matching
2. Context-Aware Sentence Transformation
3. Paragraph-Level Coherence Enhancement
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from sentence_transformers import SentenceTransformer
from config_loader import get_config
from utils import (
    split_into_sentences,
    split_into_paragraphs,
    cosine_similarity,
    calculate_embedding_distance,
    chunk_text
)


class TemplateGuidedGenerator:
    """
    Generate style-transformed text using hierarchical template matching.

    This class implements Phase 2 of the ZeroStylus framework:
    - Multi-granular template matching at sentence and paragraph levels
    - Context-aware sentence transformation using LLMs
    - Paragraph-level coherence enhancement
    """

    def __init__(
        self,
        encoder_model: Optional[str] = None,
        llm_generator: Optional[Callable] = None,
        style_intensity: Optional[float] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the template-guided generator.

        Args:
            encoder_model: Name of the sentence transformer model (defaults to config)
            llm_generator: Function to call LLM for text generation
            style_intensity: Style transfer intensity parameter α ∈ [0, 1] (defaults to config)
            config_path: Path to config file (optional)
        """
        # Load configuration
        config = get_config(config_path)

        # Use provided values or fall back to config
        encoder_model = encoder_model or config.encoder_model
        style_intensity = style_intensity if style_intensity is not None else config.style_intensity

        self.encoder = SentenceTransformer(encoder_model)
        self.llm_generator = llm_generator
        self.style_intensity = style_intensity

        # Template repositories (set by set_templates)
        self.sentence_templates = None
        self.sentence_examples = None
        self.paragraph_templates = None
        self.paragraph_examples = None

    def set_templates(self, templates: Dict):
        """
        Set the template repositories from Phase 1.

        Args:
            templates: Dictionary containing Γs and Γp from Phase 1
        """
        self.sentence_templates = templates.get('Γs', np.array([]))
        self.sentence_examples = templates.get('Γs_examples', [])
        self.paragraph_templates = templates.get('Γp', np.array([]))
        self.paragraph_examples = templates.get('Γp_examples', [])

    def match_sentence_template(self, sentence: str) -> Tuple[int, float, str]:
        """
        Match a source sentence to the closest sentence template.

        This implements Equation (2):
        τ_s^i = arg max_{τ ∈ Γs} sim(e_src_i, τ)

        Args:
            sentence: Source sentence

        Returns:
            Tuple of (template_index, similarity_score, example_sentence)
        """
        if self.sentence_templates is None or len(self.sentence_templates) == 0:
            return -1, 0.0, ""

        # Encode source sentence
        sent_embedding = self.encoder.encode(sentence, convert_to_numpy=True)

        # Compute similarity with all templates
        similarities = [
            cosine_similarity(sent_embedding, template)
            for template in self.sentence_templates
        ]

        # Find best matching template
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_example = self.sentence_examples[best_idx] if self.sentence_examples else ""

        return best_idx, best_similarity, best_example

    def match_paragraph_template(self, paragraph: str) -> Tuple[int, float, str]:
        """
        Match a source paragraph to the closest paragraph template.

        This implements Equation (3):
        τ_p* = arg min_{τ_p ∈ Γp} ||e_src_p - τ_p||

        Args:
            paragraph: Source paragraph

        Returns:
            Tuple of (template_index, distance, example_paragraph)
        """
        if self.paragraph_templates is None or len(self.paragraph_templates) == 0:
            return -1, float('inf'), ""

        # Encode source paragraph
        para_embedding = self.encoder.encode(paragraph, convert_to_numpy=True)

        # Compute distance to all templates
        distances = [
            calculate_embedding_distance(para_embedding, template, metric='euclidean')
            for template in self.paragraph_templates
        ]

        # Find closest template
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]
        best_example = self.paragraph_examples[best_idx] if self.paragraph_examples else ""

        return best_idx, best_distance, best_example

    def transform_sentence(
        self,
        sentence: str,
        sentence_template_example: str,
        paragraph_template_example: str,
        original_content: str = ""
    ) -> str:
        """
        Transform a sentence using matched templates and LLM.

        This implements Equation (4):
        s'_j = π_gen(s_j, τ_s^j, τ_p*)

        Args:
            sentence: Source sentence to transform
            sentence_template_example: Example sentence from matched template
            paragraph_template_example: Example paragraph from matched template
            original_content: Original semantic content to preserve

        Returns:
            Transformed sentence
        """
        if self.llm_generator is None:
            return sentence

        # Construct prompt for LLM
        prompt = self._construct_transformation_prompt(
            sentence,
            sentence_template_example,
            paragraph_template_example,
            original_content
        )

        # Call LLM to generate transformed sentence
        transformed = self.llm_generator(prompt)

        return transformed

    def _construct_transformation_prompt(
        self,
        sentence: str,
        sentence_template: str,
        paragraph_template: str,
        original_content: str
    ) -> str:
        """
        Construct prompt for LLM-based sentence transformation.

        The prompt guides the LLM to:
        1. Preserve semantic content
        2. Adopt lexical patterns from sentence template
        3. Maintain structural constraints from paragraph template

        Args:
            sentence: Source sentence
            sentence_template: Example sentence showing target style
            paragraph_template: Example paragraph showing target structure
            original_content: Original content for context

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a text style transfer expert. Your task is to rewrite a sentence to match a target writing style while preserving its original meaning.

**Source Sentence:**
{sentence}

**Style Reference (Sentence Level):**
{sentence_template}

**Style Reference (Paragraph Context):**
{paragraph_template[:200]}...

**Instructions:**
1. Preserve the EXACT semantic content and meaning of the source sentence
2. Adopt the writing style, tone, and expression patterns from the sentence-level reference
3. Ensure the transformed sentence fits naturally within the paragraph context shown
4. Maintain similar sentence structure and length as the reference
5. Style intensity: {self.style_intensity:.1f} (0=no change, 1=full style transfer)

**Transformed Sentence:**"""

        return prompt

    def enhance_paragraph_coherence(
        self,
        transformed_sentences: List[str],
        paragraph_template_example: str
    ) -> str:
        """
        Enhance coherence of transformed sentences at paragraph level.

        This implements Equation (5):
        p_out = π_refine([s'_1, ..., s'_n], τ_p*)

        Args:
            transformed_sentences: List of transformed sentences
            paragraph_template_example: Example paragraph showing target structure

        Returns:
            Coherent paragraph text
        """
        if self.llm_generator is None:
            return ' '.join(transformed_sentences)

        # Construct prompt for coherence enhancement
        prompt = f"""You are refining a paragraph to improve its coherence and flow.

**Transformed Sentences:**
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(transformed_sentences))}

**Target Style Reference:**
{paragraph_template_example}

**Instructions:**
1. Adjust inter-sentence transitions to improve flow
2. Add or refine discourse markers (however, therefore, moreover, etc.)
3. Ensure referential consistency across sentences
4. Maintain the logical progression shown in the reference paragraph
5. Keep the semantic content of each sentence unchanged

**Refined Paragraph:**"""

        # Call LLM for refinement
        refined_paragraph = self.llm_generator(prompt)

        return refined_paragraph

    def transform_paragraph(self, source_paragraph: str) -> str:
        """
        Transform a complete paragraph using the full pipeline.

        This implements the complete Phase 2 process:
        1. Multi-Granular Template Matching (2.1)
        2. Context-Aware Sentence Transformation (2.2)
        3. Paragraph-Level Coherence Enhancement (2.3)

        Args:
            source_paragraph: Source paragraph to transform

        Returns:
            Transformed paragraph
        """
        print(f"\n--- Transforming paragraph ({len(source_paragraph)} chars) ---")

        # Step 2.1: Multi-Granular Template Matching
        print("[2.1] Matching templates...")

        # Match paragraph template (Equation 3)
        para_idx, para_dist, para_example = self.match_paragraph_template(source_paragraph)
        print(f"  Matched paragraph template #{para_idx} (distance: {para_dist:.3f})")

        # Split into sentences
        sentences = split_into_sentences(source_paragraph)
        print(f"  Split into {len(sentences)} sentences")

        # Step 2.2: Context-Aware Sentence Transformation
        print("[2.2] Transforming sentences...")
        transformed_sentences = []

        for i, sentence in enumerate(sentences):
            # Match sentence template (Equation 2)
            sent_idx, sent_sim, sent_example = self.match_sentence_template(sentence)

            # Transform sentence (Equation 4)
            transformed = self.transform_sentence(
                sentence,
                sent_example,
                para_example,
                source_paragraph
            )

            transformed_sentences.append(transformed)
            print(f"  [{i+1}/{len(sentences)}] Matched template #{sent_idx} (sim: {sent_sim:.3f})")

        # Step 2.3: Paragraph-Level Coherence Enhancement
        print("[2.3] Enhancing coherence...")
        final_paragraph = self.enhance_paragraph_coherence(
            transformed_sentences,
            para_example
        )

        return final_paragraph

    def transform_document(self, source_text: str, max_chunk_length: int = 2000) -> str:
        """
        Transform a long document by processing paragraphs sequentially.

        Args:
            source_text: Source document to transform
            max_chunk_length: Maximum length for context window

        Returns:
            Transformed document
        """
        print("\n=== Phase 2: Template-Guided Generation ===")

        # Split document into paragraphs
        paragraphs = split_into_paragraphs(source_text)
        print(f"Document split into {len(paragraphs)} paragraphs")

        # Transform each paragraph
        transformed_paragraphs = []

        for i, para in enumerate(paragraphs):
            print(f"\n[Paragraph {i+1}/{len(paragraphs)}]")

            # Handle long paragraphs by chunking
            if len(para) > max_chunk_length:
                chunks = chunk_text(para, max_chunk_length)
                transformed_chunks = [
                    self.transform_paragraph(chunk)
                    for chunk in chunks
                ]
                transformed_para = '\n\n'.join(transformed_chunks)
            else:
                transformed_para = self.transform_paragraph(para)

            transformed_paragraphs.append(transformed_para)

        # Join paragraphs
        final_document = '\n\n'.join(transformed_paragraphs)

        return final_document

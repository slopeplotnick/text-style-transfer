"""
ZeroStylus: Zero-shot Long Text Style Transfer Framework

A hierarchical framework for long-text style transfer using LLMs through
dual-layered sentence and paragraph structure extraction and mapping.

Paper: Implementing Long Text Style Transfer with LLMs through Dual-Layered
       Sentence and Paragraph Structure Extraction and Mapping
Authors: Yusen Wu, Xiaotie Deng (Peking University)
"""

from phase1 import TemplateExtractor
from phase2 import TemplateGuidedGenerator
from evaluation import TriaxialEvaluator
from config_loader import get_config
from typing import List, Dict, Optional, Callable

__version__ = "0.1.0"
__all__ = [
    "ZeroStylus",
    "TemplateExtractor",
    "TemplateGuidedGenerator",
    "TriaxialEvaluator"
]


class ZeroStylus:
    """
    Main interface for the ZeroStylus framework.

    This class orchestrates the two-phase process:
    - Phase 1: Hierarchical Template Acquisition
    - Phase 2: Template-Guided Generation

    Example:
        >>> stylus = ZeroStylus(llm_generator=my_llm_function)
        >>> stylus.extract_templates(reference_texts)
        >>> transformed = stylus.transform(source_text)
        >>> scores = stylus.evaluate(transformed, source_text, reference_texts)
    """

    def __init__(
        self,
        encoder_model: Optional[str] = None,
        llm_generator: Optional[Callable] = None,
        llm_evaluator: Optional[Callable] = None,
        style_intensity: Optional[float] = None,
        dbscan_eps: Optional[float] = None,
        dbscan_min_samples: Optional[int] = None,
        paragraph_threshold: Optional[float] = None,
        use_bleurt: Optional[bool] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the ZeroStylus framework.

        Args:
            encoder_model: Sentence transformer model name (defaults to config)
            llm_generator: Function to call LLM for text generation
                          Signature: f(prompt: str) -> str
            llm_evaluator: Function to call LLM for evaluation
                          Signature: f(prompt: str) -> str
            style_intensity: Style transfer intensity α ∈ [0, 1] (defaults to config)
            dbscan_eps: DBSCAN epsilon parameter for clustering (defaults to config)
            dbscan_min_samples: DBSCAN minimum samples parameter (defaults to config)
            paragraph_threshold: Threshold ε for paragraph template uniqueness (defaults to config)
            use_bleurt: Whether to use BLEURT for content evaluation (defaults to config)
            config_path: Path to config.yaml file (optional)
        """
        # Load configuration
        config = get_config(config_path)

        # Use provided values or fall back to config
        encoder_model = encoder_model or config.encoder_model
        style_intensity = style_intensity if style_intensity is not None else config.style_intensity
        dbscan_eps = dbscan_eps if dbscan_eps is not None else config.dbscan_eps
        dbscan_min_samples = dbscan_min_samples if dbscan_min_samples is not None else config.dbscan_min_samples
        paragraph_threshold = paragraph_threshold if paragraph_threshold is not None else config.paragraph_threshold
        use_bleurt = use_bleurt if use_bleurt is not None else config.use_bleurt

        # Phase 1: Template Extractor
        self.extractor = TemplateExtractor(
            encoder_model=encoder_model,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            paragraph_threshold=paragraph_threshold
        )

        # Phase 2: Template-Guided Generator
        self.generator = TemplateGuidedGenerator(
            encoder_model=encoder_model,
            llm_generator=llm_generator,
            style_intensity=style_intensity
        )

        # Evaluator
        self.evaluator = TriaxialEvaluator(
            encoder_model=encoder_model,
            use_bleurt=use_bleurt,
            llm_evaluator=llm_evaluator
        )

        # Store reference texts for evaluation
        self.reference_texts = None

    def extract_templates(self, reference_texts: List[str]) -> Dict:
        """
        Execute Phase 1: Extract hierarchical templates from reference texts.

        This builds the template repositories Γs (sentence) and Γp (paragraph).

        Args:
            reference_texts: List of reference documents showing target style

        Returns:
            Dictionary containing extracted templates and statistics
        """
        # Store reference texts
        self.reference_texts = reference_texts

        # Extract templates
        templates = self.extractor.extract_all_templates(reference_texts)

        # Set templates in generator
        self.generator.set_templates(self.extractor.get_templates())

        return templates

    def transform(
        self,
        source_text: str,
        max_chunk_length: Optional[int] = None
    ) -> str:
        """
        Execute Phase 2: Transform source text using extracted templates.

        This performs:
        1. Multi-granular template matching
        2. Context-aware sentence transformation
        3. Paragraph-level coherence enhancement

        Args:
            source_text: Source document to transform
            max_chunk_length: Maximum length for processing chunks (defaults to config)

        Returns:
            Transformed document in target style
        """
        if self.generator.sentence_templates is None:
            raise ValueError(
                "Templates not extracted. Call extract_templates() first."
            )

        # Use provided value or fall back to config
        if max_chunk_length is None:
            config = get_config()
            max_chunk_length = config.max_chunk_length

        return self.generator.transform_document(source_text, max_chunk_length)

    def evaluate(
        self,
        transformed_text: str,
        original_text: str,
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the quality of style transfer using tri-axial metrics.

        Metrics:
        - X: Style Consistency
        - Y: Content Preservation
        - Z: Expression Quality
        - A: Average Score

        Args:
            transformed_text: The style-transferred output
            original_text: The original source text
            reference_texts: Reference texts (uses stored if None)

        Returns:
            Dictionary with evaluation scores
        """
        if reference_texts is None:
            if self.reference_texts is None:
                raise ValueError("No reference texts available for evaluation")
            reference_texts = self.reference_texts

        return self.evaluator.evaluate_all(
            transformed_text,
            original_text,
            reference_texts
        )

    def batch_evaluate(
        self,
        samples: List[Dict[str, str]],
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Evaluate multiple samples in batch.

        Args:
            samples: List of dicts with 'transformed' and 'original' keys
            reference_texts: Reference texts (uses stored if None)

        Returns:
            Dictionary with lists of scores for each metric
        """
        if reference_texts is None:
            if self.reference_texts is None:
                raise ValueError("No reference texts available for evaluation")
            reference_texts = self.reference_texts

        return self.evaluator.batch_evaluate(samples, reference_texts)

    def get_templates_info(self) -> Dict:
        """
        Get information about the extracted templates.

        Returns:
            Dictionary with template statistics and examples
        """
        templates = self.extractor.get_templates()

        return {
            'num_sentence_templates': len(templates['Γs']),
            'num_paragraph_templates': len(templates['Γp']),
            'sentence_examples': templates['Γs_examples'][:5],  # First 5
            'paragraph_examples': templates['Γp_examples'][:3],  # First 3
        }

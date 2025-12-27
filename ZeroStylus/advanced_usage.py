"""
Advanced usage example with real LLM integration.

This example shows how to integrate ZeroStylus with different LLM providers
and demonstrates advanced features like custom style intensity and evaluation.
"""

import os
from typing import Optional
from zerostylus import ZeroStylus


class LLMProvider:
    """Base class for LLM providers."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("OpenAI library required: pip install openai")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a text style transfer expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Anthropic library required: pip install anthropic")

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider (via OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            self.model = model
        except ImportError:
            raise ImportError("OpenAI library required: pip install openai")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()


def load_reference_papers(directory: str) -> list[str]:
    """
    Load reference papers from a directory.

    Args:
        directory: Directory containing reference text files

    Returns:
        List of reference text strings
    """
    import glob

    texts = []
    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    return texts


def academic_style_transfer_example():
    """
    Example: Transfer informal text to academic writing style.
    """
    print("=" * 70)
    print("Academic Style Transfer Example")
    print("=" * 70)

    # Initialize LLM provider
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    llm_provider = OpenAIProvider(api_key=api_key, model="gpt-4o")

    # Initialize ZeroStylus with academic writing parameters
    stylus = ZeroStylus(
        encoder_model="all-MiniLM-L6-v2",
        llm_generator=llm_provider.generate,
        llm_evaluator=llm_provider.generate,
        style_intensity=0.9,  # High intensity for clear academic style
        dbscan_eps=0.25,      # Tighter clustering for academic patterns
        paragraph_threshold=0.15
    )

    # Reference texts (academic papers)
    reference_texts = [
        """
        The proliferation of deep learning architectures has fundamentally
        transformed the landscape of natural language processing. Contemporary
        models demonstrate unprecedented capabilities in capturing semantic
        nuances and generating coherent text across diverse domains. This
        paradigm shift necessitates rigorous evaluation frameworks that extend
        beyond traditional lexical metrics to encompass structural and stylistic
        dimensions.
        """,
        """
        We propose a novel hierarchical framework for unsupervised style transfer
        that addresses critical limitations in existing methodologies. Our approach
        operates through dual-layered template extraction, systematically capturing
        both sentence-level stylistic patterns and paragraph-level discourse structures.
        Empirical evaluations across multiple benchmarks validate the efficacy of
        this architectural design, demonstrating superior performance in preserving
        semantic content while achieving target stylistic characteristics.
        """
    ]

    # Source text (informal)
    source_text = """
        We built a cool new AI system that can write in different styles.
        It's pretty smart because it doesn't need tons of training data.
        The way it works is by looking at examples and figuring out the
        patterns. Then it uses those patterns to rewrite stuff. We tried
        it out and it worked way better than other methods we compared it to.
        """

    # Extract templates
    print("\nExtracting academic writing templates...")
    stylus.extract_templates(reference_texts)

    # Transform text
    print("\nTransforming informal text to academic style...")
    transformed = stylus.transform(source_text)

    # Display results
    print("\n" + "-" * 70)
    print("ORIGINAL (Informal):")
    print("-" * 70)
    print(source_text.strip())

    print("\n" + "-" * 70)
    print("TRANSFORMED (Academic):")
    print("-" * 70)
    print(transformed.strip())

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    scores = stylus.evaluate(transformed, source_text, reference_texts)

    print(f"\nStyle Consistency:    {scores['style_consistency']:.2f}/10")
    print(f"Content Preservation: {scores['content_preservation']:.2f}/10")
    print(f"Expression Quality:   {scores['expression_quality']:.2f}/10")
    print(f"Average Score:        {scores['average_score']:.2f}/10")


def multi_author_style_example():
    """
    Example: Transfer text to match a specific author's style.
    """
    print("\n" + "=" * 70)
    print("Multi-Author Style Transfer Example")
    print("=" * 70)

    # This example would require loading actual author texts
    # For demonstration, we'll use placeholder styles

    author_styles = {
        "Author A": [
            "The analysis reveals significant patterns...",
            "Our findings demonstrate that..."
        ],
        "Author B": [
            "We observe interesting phenomena in...",
            "The data suggests compelling evidence for..."
        ]
    }

    source = "The results show that the method works well."

    print(f"\nOriginal: {source}")
    print("\nTransformed to different author styles:")

    for author, references in author_styles.items():
        # In practice, you would run the full pipeline here
        print(f"\n{author} style:")
        print(f"  (Would transform using {len(references)} reference texts)")


def batch_processing_example():
    """
    Example: Process multiple documents in batch.
    """
    print("\n" + "=" * 70)
    print("Batch Processing Example")
    print("=" * 70)

    # Prepare batch of documents
    documents = [
        "First document to transform...",
        "Second document with different content...",
        "Third document with more information..."
    ]

    print(f"\nProcessing {len(documents)} documents...")
    print("(In practice, this would run the full pipeline for each document)")


def custom_evaluation_example():
    """
    Example: Use custom evaluation metrics.
    """
    print("\n" + "=" * 70)
    print("Custom Evaluation Example")
    print("=" * 70)

    # Custom evaluator function
    def custom_quality_evaluator(prompt: str) -> str:
        # This would call your custom evaluation logic
        return "7.5"  # Mock score

    stylus = ZeroStylus(
        llm_evaluator=custom_quality_evaluator
    )

    print("\nUsing custom evaluation function...")
    print("(Would evaluate using custom metrics)")


if __name__ == "__main__":
    # Run academic style transfer example
    academic_style_transfer_example()

    # Uncomment to run other examples:
    # multi_author_style_example()
    # batch_processing_example()
    # custom_evaluation_example()

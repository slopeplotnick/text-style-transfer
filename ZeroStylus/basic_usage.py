"""
Basic usage example for ZeroStylus framework.

This example demonstrates how to use ZeroStylus for long-text style transfer
without requiring parallel corpora or LLM fine-tuning.
"""

import os
from typing import Optional, Callable
from zerostylus import ZeroStylus
from config_loader import get_config


def create_llm_generator(api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None) -> Callable:
    """
    Create an LLM generator function using OpenAI API.

    Args:
        api_key: OpenAI API key (defaults to config)
        model: Model name to use (defaults to config)
        base_url: API base URL (defaults to config)

    Returns:
        Generator function with signature: f(prompt: str) -> str
    """
    # Load config if not provided
    config = get_config()
    api_key = api_key or config.openai_api_key
    model = model or config.openai_model
    base_url = base_url or config.openai_base_url

    try:
        from openai import OpenAI

        # Initialize client with base_url if provided
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        def generator(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful text style transfer assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.openai_temperature,
                max_tokens=config.openai_max_tokens
            )
            return response.choices[0].message.content.strip()

        return generator

    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
        return None


def mock_llm_generator(prompt: str) -> str:
    """
    Mock LLM generator for testing without API access.

    This is a placeholder that simply returns the input.
    Replace with actual LLM for real usage.
    """
    # In a real scenario, this would call an LLM API
    # For now, we just extract the source sentence from the prompt
    import re
    match = re.search(r'\*\*Source Sentence:\*\*\s*(.+?)(?:\n\n|\Z)', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Transformed text"


def main():
    """
    Main example demonstrating the complete ZeroStylus workflow.
    """
    print("=" * 60)
    print("ZeroStylus: Zero-shot Long Text Style Transfer")
    print("=" * 60)

    # Step 1: Prepare reference texts (target style examples)
    reference_texts = [
        """
        Recent advances in natural language processing have demonstrated
        remarkable capabilities in various downstream tasks. The emergence
        of large language models represents a paradigm shift in how we
        approach text generation and understanding. These models, trained
        on massive corpora, exhibit surprising zero-shot learning abilities
        that challenge our traditional assumptions about machine learning.
        """,
        """
        Text style transfer constitutes a fundamental challenge in computational
        linguistics. The objective involves transforming textual attributes while
        preserving semantic content—a task requiring sophisticated modeling of
        both surface-level features and deep structural patterns. Contemporary
        approaches leverage neural architectures to disentangle style from content,
        enabling flexible manipulation of linguistic characteristics.
        """,
        """
        In this work, we propose a novel framework that addresses key limitations
        of existing methodologies. Our approach operates through hierarchical
        template extraction, capturing both micro-level stylistic features and
        macro-level discourse structures. Experimental results demonstrate
        significant improvements over baseline methods across multiple evaluation
        dimensions.
        """
    ]

    # Step 2: Prepare source text (to be transformed)
    source_text = """
        We made a new system for changing writing styles. It works pretty well
        and doesn't need any special training. The system looks at example texts
        and learns their style. Then it changes your text to match that style.
        We tested it and it works better than other methods. It keeps the meaning
        while changing the style.
        """

    # Step 3: Initialize ZeroStylus
    # Option A: With real LLM (requires API key from config.yaml)
    # api_key = os.getenv("OPENAI_API_KEY")  # Or use key from config
    # if api_key:
    #     llm_generator = create_llm_generator(api_key)
    # else:
    #     print("Warning: No API key found, using mock generator")
    #     llm_generator = mock_llm_generator

    # Option B: With mock generator (for testing)
    llm_generator = mock_llm_generator

    # Initialize with config file (parameters will be loaded from config.yaml)
    stylus = ZeroStylus(
        llm_generator=llm_generator  # All other parameters come from config.yaml
    )

    # Step 4: Phase 1 - Extract templates from reference texts
    print("\n" + "=" * 60)
    print("PHASE 1: Hierarchical Template Acquisition")
    print("=" * 60)

    templates = stylus.extract_templates(reference_texts)

    print(f"\nExtracted Templates:")
    print(f"  - Sentence templates (Γs): {templates['num_sentence_templates']}")
    print(f"  - Paragraph templates (Γp): {templates['num_paragraph_templates']}")

    # View template examples
    template_info = stylus.get_templates_info()
    print("\nSample sentence templates:")
    for i, example in enumerate(template_info['sentence_examples'][:3], 1):
        print(f"  {i}. {example[:80]}...")

    # Step 5: Phase 2 - Transform source text
    print("\n" + "=" * 60)
    print("PHASE 2: Template-Guided Generation")
    print("=" * 60)

    transformed_text = stylus.transform(source_text)

    print("\n--- Results ---")
    print("\n[Original Text]")
    print(source_text.strip())
    print("\n[Transformed Text]")
    print(transformed_text.strip())

    # Step 6: Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION: Tri-Axial Metrics")
    print("=" * 60)

    scores = stylus.evaluate(
        transformed_text=transformed_text,
        original_text=source_text,
        reference_texts=reference_texts
    )

    print("\n--- Evaluation Results ---")
    print(f"Style Consistency (X):    {scores['style_consistency']:.2f}/10")
    print(f"Content Preservation (Y): {scores['content_preservation']:.2f}/10")
    print(f"Expression Quality (Z):   {scores['expression_quality']:.2f}/10")
    print(f"{'=' * 40}")
    print(f"Average Score (A):        {scores['average_score']:.2f}/10")

    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)


def batch_example():
    """
    Example of batch processing multiple texts.
    """
    print("\n=== Batch Processing Example ===\n")

    # Prepare multiple samples
    samples = [
        {
            'original': "This is the first text to transform.",
            'transformed': "The initial document requires stylistic modification."
        },
        {
            'original': "Here's another text we want to change.",
            'transformed': "An additional passage necessitates transformation."
        }
    ]

    reference_texts = [
        "Academic writing demands precise articulation of concepts.",
        "Scholarly discourse requires rigorous formulation of ideas."
    ]

    # Initialize evaluator
    stylus = ZeroStylus()

    # Batch evaluation
    results = stylus.batch_evaluate(samples, reference_texts)

    print(f"Evaluated {len(samples)} samples")
    print(f"Average style consistency: {sum(results['style_consistency']) / len(samples):.2f}")


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to run batch example
    # batch_example()

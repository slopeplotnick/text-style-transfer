"""
Batch Style Transfer Script for ZeroStylus

This script processes multiple texts in parallel, transforming them from
style-removed versions back to academic style using reference templates.

Usage:
    python batch_process.py
"""

import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from zerostylus import ZeroStylus
from config_loader import get_config


def clean_text(text: str) -> str:
    """
    清理文本，去除多余的空行和回车

    Args:
        text: 待清理的文本

    Returns:
        清理后的文本
    """
    # 将所有换行符替换为空格
    text = re.sub(r'\n+', ' ', text)
    # 将多个连续空白字符替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text


def create_llm_generator():
    """
    Create an LLM generator function using OpenAI API with custom configuration.

    Returns:
        Generator function with signature: f(prompt: str) -> str
    """
    config = get_config()

    try:
        from openai import OpenAI

        # Initialize OpenAI client with custom base_url if provided
        client_kwargs = {"api_key": config.openai_api_key}
        if config.openai_base_url:
            client_kwargs["base_url"] = config.openai_base_url

        client = OpenAI(**client_kwargs)

        def generator(prompt: str) -> str:
            """Generate text using OpenAI API."""
            try:
                # Build API call parameters
                api_params = {
                    "model": config.openai_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful text style transfer assistant. Transform the given text to academic style while preserving its meaning."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": config.openai_temperature
                }

                # Only add max_tokens if it's specified (not None/null)
                if config.openai_max_tokens is not None:
                    api_params["max_tokens"] = config.openai_max_tokens

                response = client.chat.completions.create(**api_params)

                # Check if choices exist
                if not response.choices:
                    print(f"    [ERROR] No choices in response!")
                    return ""

                # Get the first choice
                choice = response.choices[0]
                finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else 'N/A'

                # Check message
                if not hasattr(choice, 'message'):
                    print(f"    [ERROR] No message in choice!")
                    return ""

                message = choice.message
                result = message.content

                # Handle finish_reason: length (token limit reached)
                if finish_reason == "length":
                    if config.openai_max_tokens:
                        print(f"    [WARNING] Response truncated due to token limit!")
                        print(f"    [WARNING] Prompt length: {len(prompt)} chars, Max tokens: {config.openai_max_tokens}")
                    else:
                        print(f"    [WARNING] Response truncated (reached model's max context)")

                    # If result is empty/None with finish_reason=length, it means
                    # the prompt consumed all available tokens
                    if not result or len(result.strip()) == 0:
                        print(f"    [ERROR] No output generated - prompt too long!")
                        return ""

                if result:
                    # 清理文本，去除多余的空行和回车
                    result = clean_text(result)
                    if finish_reason == "length":
                        print(f"    [WARNING] Response may be incomplete (truncated)")
                else:
                    print(f"    [WARNING] LLM returned empty response!")
                    result = ""

                return result

            except Exception as e:
                print(f"    [ERROR] LLM API call failed: {e}")
                print(f"    [ERROR] Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                return ""

        return generator

    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
        return None


def load_reference_texts(data_dir: str) -> List[str]:
    """
    Load all reference texts from the data directory.

    Args:
        data_dir: Path to directory containing reference texts

    Returns:
        List of reference text strings
    """
    reference_texts = []
    data_path = Path(data_dir)

    # Get all txt files except those in style_removed subdirectory
    txt_files = [f for f in data_path.glob("*.txt") if f.is_file()]

    print(f"Loading {len(txt_files)} reference texts from {data_dir}")

    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    reference_texts.append(content)
                    print(f"  ✓ Loaded: {txt_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {txt_file.name}: {e}")

    print(f"\nTotal reference texts loaded: {len(reference_texts)}")
    return reference_texts


def load_source_texts(source_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load all source texts to be transformed.

    Args:
        source_dir: Path to directory containing source texts
        max_samples: Maximum number of samples to load (None = load all)

    Returns:
        List of dictionaries with 'filename', 'path', and 'content'
    """
    source_texts = []
    source_path = Path(source_dir)

    txt_files = sorted(source_path.glob("*.txt"))

    # Limit the number of files if max_samples is specified
    if max_samples is not None and max_samples > 0:
        txt_files = txt_files[:max_samples]

    print(f"\nLoading {len(txt_files)} source texts from {source_dir}")
    if max_samples:
        print(f"  (Limited to {max_samples} samples)")

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    source_texts.append({
                        'filename': txt_file.name,
                        'path': str(txt_file),
                        'content': content
                    })
                    print(f"  ✓ Loaded: {txt_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {txt_file.name}: {e}")

    print(f"\nTotal source texts loaded: {len(source_texts)}")
    return source_texts


def process_single_text(
    item: Dict[str, str],
    stylus: ZeroStylus,
    output_dir: str,
    index: int,
    total: int
) -> Dict[str, any]:
    """
    Process a single text file.

    Args:
        item: Dictionary with file information
        stylus: ZeroStylus instance
        output_dir: Output directory path
        index: Current index (for progress tracking)
        total: Total number of files

    Returns:
        Dictionary with processing results
    """
    filename = item['filename']
    content = item['content']

    start_time = time.time()

    try:
        print(f"\n[{index}/{total}] Processing: {filename}")

        # Transform text
        transformed = stylus.transform(content)

        # Generate output filename
        output_filename = filename.replace('_style_removed.txt', '_transformed.txt')
        if output_filename == filename:
            output_filename = filename.replace('.txt', '_transformed.txt')

        output_path = Path(output_dir) / output_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save transformed text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transformed)

        elapsed_time = time.time() - start_time

        print(f"  ✓ Completed in {elapsed_time:.2f}s")
        print(f"  Output: {output_path}")

        return {
            'filename': filename,
            'status': 'success',
            'output_path': str(output_path),
            'elapsed_time': elapsed_time,
            'original_length': len(content),
            'transformed_length': len(transformed)
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ✗ Error processing {filename}: {e}")

        return {
            'filename': filename,
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time
        }


def batch_process(
    reference_dir: str,
    source_dir: str,
    output_dir: str,
    max_workers: int = 10,
    max_samples: Optional[int] = None
):
    """
    Batch process multiple texts in parallel.

    Args:
        reference_dir: Directory containing reference texts
        source_dir: Directory containing source texts to transform
        output_dir: Directory to save transformed texts
        max_workers: Number of parallel workers
        max_samples: Maximum number of samples to process (None = process all)
    """
    print("=" * 70)
    print("ZeroStylus Batch Processing")
    print("=" * 70)

    # Load configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  - Model: {config.openai_model}")
    print(f"  - Base URL: {config.openai_base_url}")
    print(f"  - Encoder: {config.encoder_model}")
    print(f"  - Style Intensity: {config.style_intensity}")
    print(f"  - Parallel Workers: {max_workers}")

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  - Output Directory: {output_dir_path.absolute()}")

    # Load reference texts
    print("\n" + "=" * 70)
    print("Step 1: Loading Reference Texts")
    print("=" * 70)
    reference_texts = load_reference_texts(reference_dir)

    if not reference_texts:
        print("Error: No reference texts found!")
        return

    # Load source texts
    print("\n" + "=" * 70)
    print("Step 2: Loading Source Texts")
    print("=" * 70)
    source_texts = load_source_texts(source_dir, max_samples=max_samples)

    if not source_texts:
        print("Error: No source texts found!")
        return

    # Initialize ZeroStylus
    print("\n" + "=" * 70)
    print("Step 3: Initializing ZeroStylus")
    print("=" * 70)

    llm_generator = create_llm_generator()
    if not llm_generator:
        print("Error: Failed to create LLM generator!")
        return

    stylus = ZeroStylus(llm_generator=llm_generator)

    # Extract templates
    print("\n" + "=" * 70)
    print("Step 4: Extracting Templates from Reference Texts")
    print("=" * 70)

    template_start = time.time()
    stylus.extract_templates(reference_texts)
    template_time = time.time() - template_start

    print(f"\n✓ Template extraction completed in {template_time:.2f}s")

    template_info = stylus.get_templates_info()
    print(f"\nTemplate Statistics:")
    print(f"  - Sentence templates: {template_info['num_sentence_templates']}")
    print(f"  - Paragraph templates: {template_info['num_paragraph_templates']}")

    # Process texts in parallel
    print("\n" + "=" * 70)
    print("Step 5: Processing Source Texts")
    print("=" * 70)

    total_texts = len(source_texts)
    results = []

    process_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(
                process_single_text,
                item,
                stylus,
                output_dir,
                idx + 1,
                total_texts
            ): item
            for idx, item in enumerate(source_texts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

    process_time = time.time() - process_start

    # Generate summary
    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"\nTotal files: {total_texts}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"\nTotal processing time: {process_time:.2f}s")
    print(f"Average time per file: {process_time / total_texts:.2f}s")

    # Save detailed results
    results_file = Path(output_dir) / f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': config.openai_model,
                'base_url': config.openai_base_url,
                'style_intensity': config.style_intensity,
                'max_workers': max_workers
            },
            'template_extraction_time': template_time,
            'processing_time': process_time,
            'total_files': total_texts,
            'success_count': success_count,
            'error_count': error_count,
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {results_file}")

    # Show errors if any
    if error_count > 0:
        print("\n" + "=" * 70)
        print("Errors:")
        print("=" * 70)
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r['filename']}: {r['error']}")

    print("\n" + "=" * 70)
    print("✓ Batch processing completed!")
    print("=" * 70)


def main():
    """Main entry point for batch processing."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch style transfer with ZeroStylus')
    parser.add_argument('--reference_dir', type=str, help='Directory containing reference texts')
    parser.add_argument('--source_dir', type=str, help='Directory containing source texts')
    parser.add_argument('--output_dir', type=str, help='Output directory for transformed texts')
    parser.add_argument('--max_workers', type=int, help='Number of parallel workers')
    parser.add_argument('--max_samples', type=int, help='Maximum number of samples to process')

    args = parser.parse_args()

    # Load configuration
    config = get_config()

    # Use command line arguments if provided, otherwise fall back to config
    reference_dir = args.reference_dir if args.reference_dir else config.reference_texts_dir
    source_dir = args.source_dir if args.source_dir else config.source_texts_dir
    output_dir = args.output_dir if args.output_dir else config.batch_output_dir
    max_workers = args.max_workers if args.max_workers else config.num_workers

    print(f"\nLoading paths from configuration:")
    print(f"  Reference texts: {reference_dir}")
    print(f"  Source texts: {source_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max workers: {max_workers}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    print()

    # Run batch processing
    batch_process(
        reference_dir=reference_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

"""
Main Pipeline for ZeroStylus Framework

This script provides a complete workflow:
1. Load reference and source texts
2. Extract style templates
3. Transform texts in parallel
4. Evaluate transformation quality
5. Generate comprehensive reports

Usage:
    python main_pipeline.py
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from zerostylus import ZeroStylus
from evaluation import TriaxialEvaluator
from config_loader import get_config
from thread_safe_logger import get_logger


def create_llm_generator():
    """
    Create an LLM generator function using OpenAI API with custom configuration.

    Returns:
        Generator function with signature: f(prompt: str) -> str
    """
    config = get_config()
    logger = get_logger()

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
                    logger.error("No choices in response!")
                    return ""

                # Get the first choice
                choice = response.choices[0]
                finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else 'N/A'

                # Check message
                if not hasattr(choice, 'message'):
                    logger.error("No message in choice!")
                    return ""

                message = choice.message
                result = message.content

                # Handle finish_reason: length (token limit reached)
                if finish_reason == "length":
                    if config.openai_max_tokens:
                        logger.warning(f"Response truncated due to token limit! Prompt length: {len(prompt)} chars, Max tokens: {config.openai_max_tokens}")
                    else:
                        logger.warning("Response truncated (reached model's max context)")

                    # If result is empty/None with finish_reason=length, it means
                    # the prompt consumed all available tokens
                    if not result or len(result.strip()) == 0:
                        logger.error("No output generated - prompt too long!")
                        return ""

                if result:
                    result = result.strip()
                    if finish_reason == "length":
                        logger.warning("Response may be incomplete (truncated)")
                else:
                    logger.warning("LLM returned empty response!")
                    result = ""

                return result

            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                logger.error(f"Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                return ""

        return generator

    except ImportError:
        logger.error("OpenAI library not installed. Install with: pip install openai")
        return None


def load_reference_texts(data_dir: str, logger) -> List[str]:
    """
    Load all reference texts from the data directory.

    Args:
        data_dir: Path to directory containing reference texts
        logger: Logger instance

    Returns:
        List of reference text strings
    """
    reference_texts = []
    data_path = Path(data_dir)

    # Get all txt files except those in style_removed subdirectory
    txt_files = [f for f in data_path.glob("*.txt") if f.is_file()]

    logger.info(f"Loading {len(txt_files)} reference texts from {data_dir}")

    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    reference_texts.append(content)
                    logger.info(f"✓ Loaded: {txt_file.name}")
        except Exception as e:
            logger.error(f"Error loading {txt_file.name}: {e}")

    logger.info(f"Total reference texts loaded: {len(reference_texts)}")
    return reference_texts


def load_source_texts(source_dir: str, logger) -> List[Dict[str, str]]:
    """
    Load all source texts to be transformed.

    Args:
        source_dir: Path to directory containing source texts
        logger: Logger instance

    Returns:
        List of dictionaries with 'filename', 'path', and 'content'
    """
    source_texts = []
    source_path = Path(source_dir)

    txt_files = sorted(source_path.glob("*.txt"))
    logger.info(f"Loading {len(txt_files)} source texts from {source_dir}")

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
                    logger.info(f"✓ Loaded: {txt_file.name}")
        except Exception as e:
            logger.error(f"Error loading {txt_file.name}: {e}")

    logger.info(f"Total source texts loaded: {len(source_texts)}")
    return source_texts


def process_single_text(
    item: Dict[str, str],
    stylus: ZeroStylus,
    output_dir: str,
    index: int,
    total: int,
    logger
) -> Dict[str, any]:
    """
    Process a single text file.

    Args:
        item: Dictionary with file information
        stylus: ZeroStylus instance
        output_dir: Output directory path
        index: Current index (for progress tracking)
        total: Total number of files
        logger: Logger instance

    Returns:
        Dictionary with processing results
    """
    filename = item['filename']
    content = item['content']

    # Set thread context for logging
    logger.set_task_context(f"Task-{index}", filename)

    start_time = time.time()

    try:
        logger.progress(index, total, f"Processing: {filename}")

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

        logger.success(f"Completed in {elapsed_time:.2f}s - {filename}")

        return {
            'filename': filename,
            'status': 'success',
            'output_path': str(output_path),
            'elapsed_time': elapsed_time,
            'original_length': len(content),
            'transformed_length': len(transformed),
            'original_text': content,
            'transformed_text': transformed
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing {filename}: {e}")

        return {
            'filename': filename,
            'status': 'error',
            'error': str(e),
            'elapsed_time': elapsed_time
        }


def evaluate_results(
    results: List[Dict],
    reference_texts: List[str],
    llm_generator,
    logger
) -> Dict:
    """
    Evaluate transformation results using tri-axial metrics.

    Args:
        results: List of processing results
        reference_texts: Reference texts for style evaluation
        llm_generator: LLM generator function
        logger: Logger instance

    Returns:
        Dictionary with evaluation results
    """
    logger.section("Step 6: Evaluating Transformation Quality")

    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']

    if not successful_results:
        logger.warning("No successful results to evaluate")
        return {}

    # Initialize evaluator
    evaluator = TriaxialEvaluator(llm_evaluator=llm_generator)

    # Prepare samples for batch evaluation
    samples = []
    for result in successful_results:
        samples.append({
            'transformed': result['transformed_text'],
            'original': result['original_text'],
            'filename': result['filename']
        })

    logger.info(f"Evaluating {len(samples)} transformed texts...")

    # Evaluate each sample
    evaluation_results = []

    for i, sample in enumerate(samples):
        logger.progress(i + 1, len(samples), f"Evaluating: {sample['filename']}")

        scores = evaluator.evaluate_all(
            transformed_text=sample['transformed'],
            original_text=sample['original'],
            reference_texts=reference_texts
        )

        scores['filename'] = sample['filename']
        evaluation_results.append(scores)

        logger.info(f"  Style: {scores['style_consistency']:.2f}, "
                   f"Content: {scores['content_preservation']:.2f}, "
                   f"Quality: {scores['expression_quality']:.2f}, "
                   f"Average: {scores['average_score']:.2f}")

    # Compute overall statistics
    logger.section("Evaluation Statistics")

    metrics = ['style_consistency', 'content_preservation', 'expression_quality', 'average_score']
    statistics = {}

    for metric in metrics:
        scores = [r[metric] for r in evaluation_results]
        stats = {
            'mean': sum(scores) / len(scores) if scores else 0,
            'min': min(scores) if scores else 0,
            'max': max(scores) if scores else 0,
            'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5 if scores else 0
        }
        statistics[metric] = stats

        logger.info(f"{metric.replace('_', ' ').title()}:")
        logger.info(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, "
                   f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")

    return {
        'individual_scores': evaluation_results,
        'statistics': statistics
    }


def main_pipeline(
    reference_dir: str,
    source_dir: str,
    output_dir: str,
    max_workers: int = 10,
    enable_evaluation: bool = True
):
    """
    Run the complete ZeroStylus pipeline.

    Args:
        reference_dir: Directory containing reference texts
        source_dir: Directory containing source texts to transform
        output_dir: Directory to save transformed texts
        max_workers: Number of parallel workers
        enable_evaluation: Whether to run evaluation after transformation
    """
    # Initialize logger
    log_file = Path(output_dir) / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = get_logger(str(log_file))

    logger.section("ZeroStylus Main Pipeline")

    # Load configuration
    config = get_config()
    logger.info(f"Configuration:")
    logger.info(f"  - Model: {config.openai_model}")
    logger.info(f"  - Base URL: {config.openai_base_url}")
    logger.info(f"  - Encoder: {config.encoder_model}")
    logger.info(f"  - Style Intensity: {config.style_intensity}")
    logger.info(f"  - Parallel Workers: {max_workers}")
    logger.info(f"  - Enable Evaluation: {enable_evaluation}")

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"  - Output Directory: {output_dir_path.absolute()}")

    # Step 1: Load reference texts
    logger.section("Step 1: Loading Reference Texts")
    reference_texts = load_reference_texts(reference_dir, logger)

    if not reference_texts:
        logger.error("No reference texts found!")
        return

    # Step 2: Load source texts
    logger.section("Step 2: Loading Source Texts")
    source_texts = load_source_texts(source_dir, logger)

    if not source_texts:
        logger.error("No source texts found!")
        return

    # Step 3: Initialize ZeroStylus
    logger.section("Step 3: Initializing ZeroStylus")

    llm_generator = create_llm_generator()
    if not llm_generator:
        logger.error("Failed to create LLM generator!")
        return

    stylus = ZeroStylus(llm_generator=llm_generator)

    # Step 4: Extract templates
    logger.section("Step 4: Extracting Templates from Reference Texts")

    template_start = time.time()
    stylus.extract_templates(reference_texts)
    template_time = time.time() - template_start

    logger.success(f"Template extraction completed in {template_time:.2f}s")

    template_info = stylus.get_templates_info()
    logger.info(f"Template Statistics:")
    logger.info(f"  - Sentence templates: {template_info['num_sentence_templates']}")
    logger.info(f"  - Paragraph templates: {template_info['num_paragraph_templates']}")

    # Step 5: Process texts in parallel
    logger.section("Step 5: Processing Source Texts")

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
                total_texts,
                logger
            ): item
            for idx, item in enumerate(source_texts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

    process_time = time.time() - process_start

    # Generate transformation summary
    logger.section("Transformation Summary")

    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')

    logger.info(f"Total files: {total_texts}")
    logger.info(f"  ✓ Success: {success_count}")
    logger.info(f"  ✗ Errors: {error_count}")
    logger.info(f"Total processing time: {process_time:.2f}s")
    logger.info(f"Average time per file: {process_time / total_texts:.2f}s")

    # Step 6: Evaluate results (if enabled)
    evaluation_results = {}
    if enable_evaluation and success_count > 0:
        evaluation_results = evaluate_results(
            results,
            reference_texts,
            llm_generator,
            logger
        )

    # Save comprehensive results
    logger.section("Saving Results")

    results_file = Path(output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Remove large text content from results for JSON
    json_results = []
    for r in results:
        json_result = r.copy()
        # Keep text references but truncate for JSON
        if 'original_text' in json_result:
            json_result['original_text_preview'] = json_result['original_text'][:200] + '...'
            del json_result['original_text']
        if 'transformed_text' in json_result:
            json_result['transformed_text_preview'] = json_result['transformed_text'][:200] + '...'
            del json_result['transformed_text']
        json_results.append(json_result)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': config.openai_model,
                'base_url': config.openai_base_url,
                'style_intensity': config.style_intensity,
                'max_workers': max_workers,
                'enable_evaluation': enable_evaluation
            },
            'template_extraction_time': template_time,
            'processing_time': process_time,
            'total_files': total_texts,
            'success_count': success_count,
            'error_count': error_count,
            'transformation_results': json_results,
            'evaluation_results': evaluation_results
        }, f, indent=2, ensure_ascii=False)

    logger.success(f"Results saved to: {results_file}")

    # Show errors if any
    if error_count > 0:
        logger.section("Errors")
        for r in results:
            if r['status'] == 'error':
                logger.error(f"{r['filename']}: {r['error']}")

    logger.section("✓ Pipeline Completed Successfully!")


def main():
    """Main entry point for the pipeline."""
    # Load configuration
    config = get_config()

    # Get all paths from config
    reference_dir = config.reference_texts_dir
    source_dir = config.source_texts_dir
    output_dir = config.batch_output_dir
    max_workers = config.num_workers

    print(f"\nLoading paths from configuration:")
    print(f"  Reference texts: {reference_dir}")
    print(f"  Source texts: {source_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max workers: {max_workers}\n")

    # Run main pipeline
    main_pipeline(
        reference_dir=reference_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        enable_evaluation=True  # Enable evaluation by default
    )


if __name__ == "__main__":
    main()

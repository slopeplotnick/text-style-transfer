"""
Evaluate a single baseline method and generate reports
"""
import os
import sys
import argparse
import json

# Add unified_test_framework to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_test_framework.config import TestConfig
from unified_test_framework.data_loader import DataLoader
from unified_test_framework.extended_evaluator import ExtendedStyleEvaluator
from unified_test_framework.report_generator import ReportGenerator


def evaluate_baseline(model_name: str, transferred_dir: str, max_samples: int = None):
    """
    Evaluate a baseline method

    Args:
        model_name: Name of the model (e.g., 'Zero-Shot', 'Few-Shot')
        transferred_dir: Directory containing transferred texts
        max_samples: Maximum number of samples to evaluate
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING BASELINE: {model_name}")
    print(f"{'='*80}")

    # Initialize components
    TestConfig.ensure_dirs()

    data_loader = DataLoader(
        reference_dir=TestConfig.REFERENCE_DIR,
        source_dir=TestConfig.SOURCE_DIR,
        metadata_file=TestConfig.METADATA_FILE
    )

    report_gen = ReportGenerator(TestConfig.RESULTS_DIR)

    # Load data
    print("\nLoading data...")
    samples = data_loader.load_base_samples(max_samples)

    # Load transferred texts
    samples = data_loader.load_transferred_texts(
        samples, transferred_dir, naming_pattern="auto"
    )

    # Filter valid samples
    samples = data_loader.filter_valid_samples(samples)

    if not samples:
        print(f"Error: No valid samples found for {model_name}")
        return

    # Extract texts
    references, sources, transferred = data_loader.extract_texts(samples)

    print(f"\nEvaluating with {len(samples)} samples")

    # Save samples summary
    summary_file = os.path.join(
        TestConfig.RESULTS_DIR,
        f"{model_name.lower().replace('-', '_')}_samples_summary.json"
    )
    data_loader.save_samples_summary(samples, summary_file)

    # Initialize evaluator
    evaluator = ExtendedStyleEvaluator(
        enable_deeptransfer_metrics=True,
        content_encoder_model=TestConfig.CONTENT_ENCODER_MODEL,
        style_encoder_model=TestConfig.STYLE_ENCODER_MODEL,
        use_gpu=TestConfig.USE_GPU,
        batch_size=TestConfig.ENCODING_BATCH_SIZE
    )

    # Run evaluation
    results = evaluator.evaluate_all_extended(
        source_texts=sources,
        reference_texts=references,
        transferred_texts=transferred,
        train_classifier=True,
        content_weight=TestConfig.CONTENT_WEIGHT,
        style_weight=TestConfig.STYLE_WEIGHT
    )

    # Print results
    evaluator.print_results_extended(results)

    # Save results using the same format as other models
    report_gen.save_individual_model_report(model_name, results)

    print(f"\nResults saved to: {TestConfig.RESULTS_DIR}/{model_name.lower().replace('-', '_')}_results.json")

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETED: {model_name}")
    print(f"{'='*80}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline method"
    )

    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Name of the baseline model (e.g., "Zero-Shot", "Few-Shot")'
    )

    parser.add_argument(
        '--transferred-dir',
        type=str,
        required=True,
        help='Directory containing transferred texts'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_baseline(
        model_name=args.model_name,
        transferred_dir=args.transferred_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

"""
Main Test Runner for Unified Style Transfer Evaluation
Tests CAT-LLM, ZeroStylus, and DeepTransfer with unified metrics
"""
import os
import sys
import argparse
from typing import Optional, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .config import TestConfig
from .data_loader import DataLoader, load_model_data
from .extended_evaluator import ExtendedStyleEvaluator
from .report_generator import ReportGenerator


class TestRunner:
    """Main test runner for evaluating style transfer models"""

    def __init__(self, config: TestConfig = TestConfig):
        """
        Initialize test runner.

        Args:
            config: Test configuration
        """
        self.config = config
        config.ensure_dirs()

        # Initialize report generator
        self.report_gen = ReportGenerator(config.RESULTS_DIR)

        # Data loader
        self.data_loader = DataLoader(
            reference_dir=config.REFERENCE_DIR,
            source_dir=config.SOURCE_DIR,
            metadata_file=config.METADATA_FILE
        )

    def test_single_model(
        self,
        model_name: str,
        transferred_dir: str,
        max_samples: Optional[int] = None,
        naming_pattern: str = "auto",
        enable_deeptransfer_metrics: bool = True,
        skip_catllm: bool = False,
        skip_zerostylus: bool = False,
        skip_deeptransfer: bool = False
    ) -> Dict:
        """
        Test a single model.

        Args:
            model_name: Name of the model (e.g., 'CAT-LLM', 'ZeroStylus', 'DeepTransfer')
            transferred_dir: Directory containing transferred texts
            max_samples: Maximum number of samples to test
            naming_pattern: File naming pattern for transferred texts
            enable_deeptransfer_metrics: Whether to enable DeepTransfer metrics
            skip_catllm: Skip CAT-LLM metrics
            skip_zerostylus: Skip ZeroStylus metrics
            skip_deeptransfer: Skip DeepTransfer metrics

        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*80)
        print(f"TESTING MODEL: {model_name}")
        print("="*80)

        # Load data
        print("\nLoading data...")
        samples = self.data_loader.load_base_samples(max_samples)

        # Load transferred texts
        samples = self.data_loader.load_transferred_texts(
            samples, transferred_dir, naming_pattern
        )

        # Filter valid samples
        samples = self.data_loader.filter_valid_samples(samples)

        if not samples:
            print(f"Error: No valid samples found for {model_name}")
            return {}

        # Extract texts
        references, sources, transferred = self.data_loader.extract_texts(samples)

        print(f"\nTesting with {len(samples)} samples")

        # Save samples summary
        summary_file = os.path.join(
            self.config.RESULTS_DIR,
            f"{model_name.lower()}_samples_summary.json"
        )
        self.data_loader.save_samples_summary(samples, summary_file)

        # Initialize evaluator
        evaluator = ExtendedStyleEvaluator(
            enable_deeptransfer_metrics=enable_deeptransfer_metrics,
            content_encoder_model=self.config.CONTENT_ENCODER_MODEL,
            style_encoder_model=self.config.STYLE_ENCODER_MODEL,
            use_gpu=self.config.USE_GPU,
            batch_size=self.config.ENCODING_BATCH_SIZE
        )

        # Run evaluation
        results = evaluator.evaluate_all_extended(
            source_texts=sources,
            reference_texts=references,
            transferred_texts=transferred,
            train_classifier=True,
            content_weight=self.config.CONTENT_WEIGHT,
            style_weight=self.config.STYLE_WEIGHT,
            skip_catllm=skip_catllm,
            skip_zerostylus=skip_zerostylus,
            skip_deeptransfer=skip_deeptransfer
        )

        # Print results
        evaluator.print_results_extended(results)

        # Save individual model report
        self.report_gen.save_individual_model_report(model_name, results)

        return results

    def test_all_models(
        self,
        models_config: Dict[str, Dict],
        max_samples: Optional[int] = None,
        enable_deeptransfer_metrics: bool = True
    ) -> Dict[str, Dict]:
        """
        Test all models.

        Args:
            models_config: Dictionary mapping model names to their config
                Example: {
                    'CAT-LLM': {
                        'transferred_dir': '/path/to/catllm/output',
                        'naming_pattern': 'auto'
                    },
                    ...
                }
            max_samples: Maximum number of samples to test
            enable_deeptransfer_metrics: Whether to enable DeepTransfer metrics

        Returns:
            Dictionary mapping model names to their results
        """
        all_results = {}

        for model_name, model_config in models_config.items():
            try:
                results = self.test_single_model(
                    model_name=model_name,
                    transferred_dir=model_config['transferred_dir'],
                    max_samples=max_samples,
                    naming_pattern=model_config.get('naming_pattern', 'auto'),
                    enable_deeptransfer_metrics=enable_deeptransfer_metrics,
                    skip_catllm=model_config.get('skip_catllm', False),
                    skip_zerostylus=model_config.get('skip_zerostylus', False),
                    skip_deeptransfer=model_config.get('skip_deeptransfer', False)
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"\nError testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[model_name] = {}

        # Generate comparison reports
        if len(all_results) > 1:
            print("\n" + "="*80)
            print("GENERATING COMPARISON REPORTS")
            print("="*80)
            self.report_gen.generate_all_reports(all_results)

        return all_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Unified Style Transfer Testing Framework"
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['catllm', 'zerostylus', 'deeptransfer', 'all'],
        default='all',
        help='Which model(s) to test'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to test (None = all)'
    )

    parser.add_argument(
        '--catllm-dir',
        type=str,
        default=None,
        help='CAT-LLM transferred texts directory'
    )

    parser.add_argument(
        '--zerostylus-dir',
        type=str,
        default=None,
        help='ZeroStylus transferred texts directory'
    )

    parser.add_argument(
        '--deeptransfer-dir',
        type=str,
        default=None,
        help='DeepTransfer transferred texts directory'
    )

    parser.add_argument(
        '--no-dt-metrics',
        action='store_true',
        help='Disable DeepTransfer-specific metrics'
    )

    parser.add_argument(
        '--skip-catllm',
        action='store_true',
        help='Skip CAT-LLM metrics'
    )

    parser.add_argument(
        '--skip-zerostylus',
        action='store_true',
        help='Skip ZeroStylus metrics'
    )

    parser.add_argument(
        '--skip-deeptransfer',
        action='store_true',
        help='Skip DeepTransfer metrics'
    )

    args = parser.parse_args()

    # Print configuration
    TestConfig.print_config()

    # Initialize runner
    runner = TestRunner()

    # Configure models to test
    models_config = {}

    if args.model in ['catllm', 'all']:
        catllm_dir = args.catllm_dir or TestConfig.CATLLM_OUTPUT_DIR
        if catllm_dir and os.path.exists(catllm_dir):
            models_config['CAT-LLM'] = {
                'transferred_dir': catllm_dir,
                'naming_pattern': 'auto',
                'skip_catllm': args.skip_catllm,
                'skip_zerostylus': args.skip_zerostylus,
                'skip_deeptransfer': args.skip_deeptransfer
            }

    if args.model in ['zerostylus', 'all']:
        zerostylus_dir = args.zerostylus_dir or TestConfig.ZEROSTYLUS_OUTPUT_DIR
        if zerostylus_dir and os.path.exists(zerostylus_dir):
            models_config['ZeroStylus'] = {
                'transferred_dir': zerostylus_dir,
                'naming_pattern': 'auto',
                'skip_catllm': args.skip_catllm,
                'skip_zerostylus': args.skip_zerostylus,
                'skip_deeptransfer': args.skip_deeptransfer
            }

    if args.model in ['deeptransfer', 'all']:
        deeptransfer_dir = args.deeptransfer_dir or TestConfig.DEEPTRANSFER_OUTPUT_DIR
        if deeptransfer_dir and os.path.exists(deeptransfer_dir):
            models_config['DeepTransfer'] = {
                'transferred_dir': deeptransfer_dir,
                'naming_pattern': 'auto',
                'skip_catllm': args.skip_catllm,
                'skip_zerostylus': args.skip_zerostylus,
                'skip_deeptransfer': args.skip_deeptransfer
            }

    if not models_config:
        print("\nError: No valid model directories found!")
        print("Please specify model directories with --catllm-dir, --zerostylus-dir, or --deeptransfer-dir")
        print("Or update the paths in config.py")
        return

    # Run tests
    results = runner.test_all_models(
        models_config=models_config,
        max_samples=args.max_samples,
        enable_deeptransfer_metrics=not args.no_dt_metrics
    )

    print("\n" + "="*80)
    print("TESTING COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {TestConfig.RESULTS_DIR}")


if __name__ == "__main__":
    main()

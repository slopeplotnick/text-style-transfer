"""
Quick Test Script for Unified Style Transfer Evaluation
Uses 2 samples for quick testing
"""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_test_framework.config import TestConfig
from unified_test_framework.test_runner import TestRunner


def setup_api_config():
    """Setup API configuration for LLM calls"""
    # Set environment variables for API access
    os.environ['OPENAI_API_BASE'] = 'https://newapi.deepwisdom.ai/v1'
    os.environ['OPENAI_API_KEY'] = 'sk-YF5cLsbs1SI47T8ilS4BCIU7wYo7l2UN43dqONwb0WbvyXxA'
    os.environ['OPENAI_MODEL'] = 'gpt-4o'

    print("API Configuration:")
    print(f"  Base URL: {os.environ['OPENAI_API_BASE']}")
    print(f"  Model: {os.environ['OPENAI_MODEL']}")


def main():
    """Run quick test with 2 samples"""
    print("="*80)
    print("QUICK TEST - UNIFIED STYLE TRANSFER EVALUATION")
    print("Testing with 2 samples")
    print("="*80)

    # Setup API configuration
    setup_api_config()

    # Update config for quick test
    TestConfig.MAX_SAMPLES = 2
    TestConfig.ENCODING_BATCH_SIZE = 2

    # Print configuration
    TestConfig.print_config()

    # Initialize test runner
    runner = TestRunner(TestConfig)

    # Define models to test
    # You can modify these paths based on where your model outputs are
    models_config = {}

    # Example: Test with mock transferred texts (you should replace with actual paths)
    # For now, let's create a simple test that uses reference texts as transferred
    # This is just for testing the framework

    # Option 1: Test a single model
    print("\n" + "="*80)
    print("OPTION: Test with custom transferred directory")
    print("="*80)

    # You can specify a custom directory here
    custom_transferred_dir = input("\nEnter transferred texts directory (or press Enter to skip): ").strip()

    if custom_transferred_dir and os.path.exists(custom_transferred_dir):
        model_name = input("Enter model name (e.g., CAT-LLM, ZeroStylus, DeepTransfer): ").strip() or "TestModel"

        models_config[model_name] = {
            'transferred_dir': custom_transferred_dir,
            'naming_pattern': 'auto'
        }
    else:
        print("\nNo custom directory provided or directory not found.")
        print("Creating a demo test using reference texts as transferred texts...")
        print("(This is just to test the framework, not for real evaluation)")

        # For demo purposes, use reference directory as transferred
        models_config['DemoTest'] = {
            'transferred_dir': TestConfig.REFERENCE_DIR,
            'naming_pattern': 'ref_XXXX',
            'skip_deeptransfer': True  # Skip DT metrics for quick demo
        }

    # Run tests
    if models_config:
        try:
            results = runner.test_all_models(
                models_config=models_config,
                max_samples=2,  # Only 2 samples for quick test
                enable_deeptransfer_metrics=True
            )

            print("\n" + "="*80)
            print("QUICK TEST COMPLETED")
            print("="*80)
            print(f"\nResults saved to: {TestConfig.RESULTS_DIR}")
            print("\nGenerated files:")
            print(f"  - Individual model reports: {TestConfig.RESULTS_DIR}/<model_name>_results.json")
            print(f"  - Comparison report (JSON): {TestConfig.RESULTS_DIR}/evaluation_results.json")
            print(f"  - Comparison report (Markdown): {TestConfig.RESULTS_DIR}/evaluation_report.md")
            print(f"  - Comparison report (CSV): {TestConfig.RESULTS_DIR}/evaluation_results.csv")

        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo models configured for testing.")


if __name__ == "__main__":
    main()

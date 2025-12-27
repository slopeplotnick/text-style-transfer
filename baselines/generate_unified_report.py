"""
Generate unified evaluation report including all models (CAT-LLM, ZeroStylus, DeepTransfer, Zero-Shot, Few-Shot)
"""
import os
import sys
import json
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_test_framework.config import TestConfig
from unified_test_framework.report_generator import ReportGenerator


def load_all_results():
    """Load all available evaluation results from results directory"""
    results_dir = TestConfig.RESULTS_DIR
    all_results = {}

    # Mapping of result files to model names
    result_files = {
        'cat-llm_results.json': 'CAT-LLM',
        'zerostylus_results.json': 'ZeroStylus',
        'deeptransfer_results.json': 'DeepTransfer',
        'zero_shot_results.json': 'Zero-Shot',
        'few_shot_results.json': 'Few-Shot'
    }

    for filename, model_name in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Normalize data structure: extract metrics from 'results' key if present
                if 'results' in data and isinstance(data['results'], dict):
                    # Data has nested structure, extract the metrics
                    all_results[model_name] = data['results']
                else:
                    # Data is already flat, use as is
                    all_results[model_name] = data

                print(f"✓ Loaded results for {model_name}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
        else:
            print(f"- No results found for {model_name}")

    return all_results


def main():
    """Generate unified evaluation report"""
    print("\n" + "="*80)
    print("GENERATING UNIFIED EVALUATION REPORT")
    print("="*80)

    # Load all results
    all_results = load_all_results()

    if not all_results:
        print("\nError: No results found to generate report!")
        return

    print(f"\nFound results for {len(all_results)} models")

    # Initialize report generator
    report_gen = ReportGenerator(TestConfig.RESULTS_DIR)

    # Generate all comparison reports
    print("\nGenerating comparison reports...")
    report_gen.generate_all_reports(all_results)

    print("\n" + "="*80)
    print("UNIFIED REPORT GENERATION COMPLETED")
    print("="*80)
    print(f"\nReports saved to: {TestConfig.RESULTS_DIR}")
    print("  - evaluation_report.md")
    print("  - evaluation_results.json")
    print("  - evaluation_results.csv")


if __name__ == "__main__":
    main()

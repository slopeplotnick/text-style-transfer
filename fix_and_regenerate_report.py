#!/usr/bin/env python
"""
Fix evaluation_results.json structure and regenerate report
"""
import json
import sys
import os

# Add unified_test_framework to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unified_test_framework'))

from report_generator import ReportGenerator


def fix_evaluation_results():
    """Fix the data structure for Zero-Shot and Few-Shot models"""

    # Load the current results
    results_file = "unified_test_framework/results/evaluation_results.json"

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Fix Zero-Shot and Few-Shot data structure
    for model_name in ['Zero-Shot', 'Few-Shot']:
        if model_name in data['results']:
            model_data = data['results'][model_name]

            # If data is not nested under 'results', wrap it
            if 'results' not in model_data or not isinstance(model_data.get('results'), dict):
                # Wrap the metrics in 'results' field
                data['results'][model_name] = {
                    'model': model_name,
                    'generated_at': data.get('generated_at', ''),
                    'results': model_data
                }
                print(f"Fixed data structure for {model_name}")

    # Save the fixed data
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFixed evaluation results saved to: {results_file}")

    return data['results']


def regenerate_reports(results):
    """Regenerate all reports with fixed data"""

    # Extract just the results for each model (without metadata)
    model_results = {}
    for model_name, model_data in results.items():
        if 'results' in model_data and isinstance(model_data['results'], dict):
            model_results[model_name] = model_data['results']
        else:
            model_results[model_name] = model_data

    # Initialize report generator
    output_dir = "unified_test_framework/results"
    report_gen = ReportGenerator(output_dir)

    # Generate all reports
    print("\nRegenerating reports...")
    report_gen.generate_all_reports(model_results)

    print("\nAll reports regenerated successfully!")


def main():
    """Main function"""
    print("="*80)
    print("FIXING EVALUATION RESULTS AND REGENERATING REPORTS")
    print("="*80)

    # Fix the data structure
    results = fix_evaluation_results()

    # Regenerate reports
    regenerate_reports(results)

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()

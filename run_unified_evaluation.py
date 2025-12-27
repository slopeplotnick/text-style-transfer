#!/usr/bin/env python
"""
Script to run unified evaluation on CAT-LLM and ZeroStylus models.
This script should be run on the server with access to the data directories.
"""

import sys
import os
from unified_evaluation import UnifiedStyleEvaluator, evaluate_model


def main():
    """Main evaluation runner."""
    print("\n" + "="*80)
    print("UNIFIED STYLE TRANSFER EVALUATION - RUNNER")
    print("="*80)

    # Check if running on server (basic check for data directories)
    reference_dir = "/root/datasets/text_style_transfer/ipm_abstracts"
    if not os.path.exists(reference_dir):
        print("\nError: Data directory not found!")
        print("Expected directory: " + reference_dir)
        print("\nThis script should be run on the server with access to:")
        print("  - /root/datasets/text_style_transfer/ipm_abstracts")
        print("  - /root/datasets/text_style_transfer/ipm_abstracts/style_removed")
        print("  - /root/rzy/tst/CAT-LLM/output-gpt-5-2025-08-07-20251030_090303/transferred")
        print("  - /root/rzy/tst/ZeroStylus/outputs/batch_transformed")
        sys.exit(1)

    # Initialize evaluator
    print("\nInitializing unified evaluator...")
    evaluator = UnifiedStyleEvaluator()

    # Define common paths
    source_dir = "/root/datasets/text_style_transfer/ipm_abstracts/style_removed"

    # Create output directory
    output_dir = "evaluation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Evaluate CAT-LLM
    print("\n" + "="*80)
    print("PHASE 1: Evaluating CAT-LLM")
    print("="*80)

    catllm_transferred_dir = "/root/rzy/tst/CAT-LLM/output-gpt-5-2025-08-07-20251030_090303/transferred"
    catllm_output_file = os.path.join(output_dir, "catllm_results.json")

    catllm_results = evaluate_model(
        model_name="CAT-LLM",
        reference_dir=reference_dir,
        source_dir=source_dir,
        transferred_dir=catllm_transferred_dir,
        output_file=catllm_output_file,
        evaluator=evaluator
    )

    # Evaluate ZeroStylus
    print("\n" + "="*80)
    print("PHASE 2: Evaluating ZeroStylus")
    print("="*80)

    zerostylus_transferred_dir = "/root/rzy/tst/ZeroStylus/outputs/batch_transformed"
    zerostylus_output_file = os.path.join(output_dir, "zerostylus_results.json")

    zerostylus_results = evaluate_model(
        model_name="ZeroStylus",
        reference_dir=reference_dir,
        source_dir=source_dir,
        transferred_dir=zerostylus_transferred_dir,
        output_file=zerostylus_output_file,
        evaluator=evaluator
    )

    # Generate comparison report
    if catllm_results and zerostylus_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        comparison = {
            "CAT-LLM": {
                "overall_score": catllm_results.get('catllm_overall_score', 'N/A'),
                "style_accuracy": catllm_results.get('style_transfer_accuracy', 'N/A'),
                "bleu4_content": catllm_results.get('bleu-4_content', 'N/A'),
                "bleu4_style": catllm_results.get('bleu-4_style', 'N/A'),
                "bert_f1_content": catllm_results.get('bert_f1_content', 'N/A'),
                "bert_f1_style": catllm_results.get('bert_f1_style', 'N/A'),
                "perplexity": catllm_results.get('perplexity_mean', 'N/A')
            },
            "ZeroStylus": {
                "average_score": zerostylus_results.get('zerostylus_average_score', 'N/A'),
                "style_consistency": zerostylus_results.get('style_consistency_mean', 'N/A'),
                "content_preservation": zerostylus_results.get('content_preservation_zs_mean', 'N/A'),
                "expression_quality": zerostylus_results.get('expression_quality_mean', 'N/A'),
                "keyword_retention": zerostylus_results.get('keyword_retention_mean', 'N/A'),
                # Include CAT-LLM metrics that were also computed for ZeroStylus
                "bleu4_content": zerostylus_results.get('bleu-4_content', 'N/A'),
                "bleu4_style": zerostylus_results.get('bleu-4_style', 'N/A'),
                "style_accuracy": zerostylus_results.get('style_transfer_accuracy', 'N/A')
            }
        }

        # Print comparison
        print("\n[CAT-LLM Metrics]")
        for metric, value in comparison["CAT-LLM"].items():
            print(f"  {metric}: {value}")

        print("\n[ZeroStylus Metrics]")
        for metric, value in comparison["ZeroStylus"].items():
            print(f"  {metric}: {value}")

        # Save comparison
        import json
        comparison_file = os.path.join(output_dir, "comparison_summary.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\nComparison saved to: {comparison_file}")

        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {output_dir}/")
        print(f"  - {catllm_output_file}")
        print(f"  - {zerostylus_output_file}")
        print(f"  - {comparison_file}")
    else:
        print("\nError: Could not complete evaluation for both models")
        sys.exit(1)


if __name__ == "__main__":
    main()

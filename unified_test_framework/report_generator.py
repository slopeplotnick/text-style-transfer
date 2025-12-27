"""
Report Generator for Unified Style Transfer Testing
Generates comparison reports in multiple formats (JSON, Markdown, CSV)
"""
import os
import json
import csv
from typing import Dict, List, Optional
from datetime import datetime


class ReportGenerator:
    """Generates evaluation reports in multiple formats"""

    def __init__(self, output_dir: str):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json_report(
        self,
        results: Dict[str, Dict],
        filename: str = "evaluation_results.json"
    ):
        """
        Save results as JSON.

        Args:
            results: Dictionary mapping model names to their results
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)

        # Add metadata
        report = {
            'generated_at': datetime.now().isoformat(),
            'models': list(results.keys()),
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"JSON report saved to: {output_path}")

    def save_markdown_report(
        self,
        results: Dict[str, Dict],
        filename: str = "evaluation_report.md"
    ):
        """
        Save results as Markdown.

        Args:
            results: Dictionary mapping model names to their results
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Style Transfer Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Models Evaluated:** {', '.join(results.keys())}\n\n")
            f.write("---\n\n")

            # Overall Comparison
            f.write("## Overall Comparison\n\n")
            self._write_overall_comparison_table(f, results)

            # Individual Model Results
            for model_name, model_results in results.items():
                f.write(f"\n## {model_name} Detailed Results\n\n")
                self._write_model_details(f, model_results)

            # Key Metrics Comparison
            f.write("\n## Key Metrics Comparison\n\n")
            self._write_key_metrics_comparison(f, results)

        print(f"Markdown report saved to: {output_path}")

    def _write_overall_comparison_table(self, f, results: Dict[str, Dict]):
        """Write overall comparison table to markdown file"""
        # Collect key metrics
        key_metrics = [
            ('CAT-LLM Overall', 'catllm_overall_score'),
            ('ZeroStylus Average', 'zerostylus_average_score'),
            ('Style Accuracy', 'style_transfer_accuracy'),
            ('BLEU-4 Content', 'bleu-4_content'),
            ('BLEU-4 Style', 'bleu-4_style'),
            ('BERTScore F1 Content', 'bert_f1_content'),
            ('BERTScore F1 Style', 'bert_f1_style'),
            ('Perplexity', 'perplexity_mean'),
            ('DT Content Preservation', 'dt_content_preservation_mean'),
            ('DT Style Similarity', 'dt_style_similarity_mean'),
            ('DT Disentanglement', 'dt_disentanglement_score'),
        ]

        # Write table header
        f.write("| Metric | " + " | ".join(results.keys()) + " |\n")
        f.write("|--------|" + "|".join(["--------"] * len(results)) + "|\n")

        # Write metrics
        for metric_name, metric_key in key_metrics:
            row = f"| {metric_name} |"
            for model_name in results.keys():
                value = results[model_name].get(metric_key, 'N/A')
                if isinstance(value, float):
                    row += f" {value:.4f} |"
                else:
                    row += f" {value} |"
            f.write(row + "\n")

        f.write("\n")

    def _write_model_details(self, f, model_results: Dict):
        """Write detailed results for a single model"""
        # Group metrics by category
        categories = {
            'CAT-LLM Metrics': [
                'bleu-1_content', 'bleu-2_content', 'bleu-3_content', 'bleu-4_content',
                'bleu-1_style', 'bleu-2_style', 'bleu-3_style', 'bleu-4_style',
                'bert_precision_content', 'bert_recall_content', 'bert_f1_content',
                'bert_precision_style', 'bert_recall_style', 'bert_f1_style',
                'style_transfer_accuracy', 'perplexity_mean', 'catllm_overall_score'
            ],
            'ZeroStylus Metrics': [
                'style_consistency_mean', 'content_preservation_zs_mean',
                'expression_quality_mean', 'zerostylus_average_score'
            ],
            'DeepTransfer Metrics': [
                'dt_content_preservation_mean', 'dt_style_similarity_mean',
                'dt_disentanglement_score', 'dt_alignment_score_mean',
                'dt_diversity_pairwise_dist_mean'
            ]
        }

        for category, metrics in categories.items():
            # Check if any metric in this category exists
            if not any(m in model_results for m in metrics):
                continue

            f.write(f"### {category}\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")

            for metric_key in metrics:
                if metric_key in model_results:
                    value = model_results[metric_key]
                    metric_display = metric_key.replace('_', ' ').title()
                    if isinstance(value, float):
                        f.write(f"| {metric_display} | {value:.4f} |\n")
                    else:
                        f.write(f"| {metric_display} | {value} |\n")

            f.write("\n")

    def _write_key_metrics_comparison(self, f, results: Dict[str, Dict]):
        """Write key metrics comparison charts"""
        # Content Preservation Metrics
        f.write("### Content Preservation\n\n")
        f.write("| Model | BLEU-4 | BERTScore F1 | DT Content |\n")
        f.write("|-------|--------|--------------|------------|\n")

        for model_name in results.keys():
            bleu4 = results[model_name].get('bleu-4_content', 'N/A')
            bert_f1 = results[model_name].get('bert_f1_content', 'N/A')
            dt_content = results[model_name].get('dt_content_preservation_mean', 'N/A')

            bleu4_str = f"{bleu4:.4f}" if isinstance(bleu4, float) else bleu4
            bert_str = f"{bert_f1:.4f}" if isinstance(bert_f1, float) else bert_f1
            dt_str = f"{dt_content:.4f}" if isinstance(dt_content, float) else dt_content

            f.write(f"| {model_name} | {bleu4_str} | {bert_str} | {dt_str} |\n")

        f.write("\n")

        # Style Transfer Metrics
        f.write("### Style Transfer\n\n")
        f.write("| Model | BLEU-4 Style | Style Accuracy | DT Style Sim |\n")
        f.write("|-------|--------------|----------------|---------------|\n")

        for model_name in results.keys():
            bleu4 = results[model_name].get('bleu-4_style', 'N/A')
            style_acc = results[model_name].get('style_transfer_accuracy', 'N/A')
            dt_style = results[model_name].get('dt_style_similarity_mean', 'N/A')

            bleu4_str = f"{bleu4:.4f}" if isinstance(bleu4, float) else bleu4
            acc_str = f"{style_acc:.4f}" if isinstance(style_acc, float) else style_acc
            dt_str = f"{dt_style:.4f}" if isinstance(dt_style, float) else dt_style

            f.write(f"| {model_name} | {bleu4_str} | {acc_str} | {dt_str} |\n")

        f.write("\n")

    def save_csv_report(
        self,
        results: Dict[str, Dict],
        filename: str = "evaluation_results.csv"
    ):
        """
        Save results as CSV.

        Args:
            results: Dictionary mapping model names to their results
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)

        # Collect all unique metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())

        all_metrics = sorted(all_metrics)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Metric'] + list(results.keys()))

            # Rows
            for metric in all_metrics:
                row = [metric]
                for model_name in results.keys():
                    value = results[model_name].get(metric, 'N/A')
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                writer.writerow(row)

        print(f"CSV report saved to: {output_path}")

    def generate_all_reports(
        self,
        results: Dict[str, Dict],
        json_filename: str = "evaluation_results.json",
        markdown_filename: str = "evaluation_report.md",
        csv_filename: str = "evaluation_results.csv"
    ):
        """
        Generate all report formats.

        Args:
            results: Dictionary mapping model names to their results
            json_filename: JSON output filename
            markdown_filename: Markdown output filename
            csv_filename: CSV output filename
        """
        print("\nGenerating evaluation reports...")

        self.save_json_report(results, json_filename)
        self.save_markdown_report(results, markdown_filename)
        self.save_csv_report(results, csv_filename)

        print(f"\nAll reports saved to: {self.output_dir}")

    def save_individual_model_report(
        self,
        model_name: str,
        results: Dict,
        filename: Optional[str] = None
    ):
        """
        Save report for a single model.

        Args:
            model_name: Name of the model
            results: Results dictionary for the model
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            # Convert model name to snake_case for filename (e.g., "Zero-Shot" -> "zero_shot")
            filename = f"{model_name.lower().replace('-', '_')}_results.json"

        output_path = os.path.join(self.output_dir, filename)

        report = {
            'model': model_name,
            'generated_at': datetime.now().isoformat(),
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Individual report for {model_name} saved to: {output_path}")

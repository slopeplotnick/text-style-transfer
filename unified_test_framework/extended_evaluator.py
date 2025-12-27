"""
Extended Evaluator for Unified Style Transfer Testing
Combines metrics from CAT-LLM, ZeroStylus, and DeepTransfer
"""
import sys
import os

# Add parent directory to path to import unified_evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_evaluation import UnifiedStyleEvaluator
from typing import Dict, List, Optional
from .deeptransfer_metrics import DualEncoderMetrics


class ExtendedStyleEvaluator(UnifiedStyleEvaluator):
    """
    Extended evaluator that includes all metrics from:
    - CAT-LLM (BLEU, BERTScore, Style Accuracy, Perplexity)
    - ZeroStylus (Style Consistency, Content Preservation, Expression Quality)
    - DeepTransfer (Dual-Encoder based metrics)
    """

    def __init__(
        self,
        enable_deeptransfer_metrics: bool = True,
        content_encoder_model: str = "allenai/specter2_base",
        style_encoder_model: str = "AnnaWegmann/Style-Embedding",
        use_gpu: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize extended evaluator.

        Args:
            enable_deeptransfer_metrics: Whether to compute DeepTransfer metrics
            content_encoder_model: Model for content encoding
            style_encoder_model: Model for style encoding
            use_gpu: Whether to use GPU for encoding
            batch_size: Batch size for encoding
        """
        # Initialize base evaluator
        super().__init__()

        # Initialize DeepTransfer metrics
        self.enable_deeptransfer_metrics = enable_deeptransfer_metrics
        self.dt_metrics = None

        if enable_deeptransfer_metrics:
            print("\nInitializing DeepTransfer metrics...")
            self.dt_metrics = DualEncoderMetrics(
                content_model_name=content_encoder_model,
                style_model_name=style_encoder_model,
                use_gpu=use_gpu,
                batch_size=batch_size
            )

    def evaluate_all_extended(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        transferred_texts: List[str],
        train_classifier: bool = True,
        classifier_model_path: Optional[str] = None,
        output_file: Optional[str] = None,
        content_weight: float = 0.5,
        style_weight: float = 0.5,
        skip_catllm: bool = False,
        skip_zerostylus: bool = False,
        skip_deeptransfer: bool = False
    ) -> Dict[str, any]:
        """
        Complete evaluation with all metrics from all three systems.

        Args:
            source_texts: Source texts (de-styled)
            reference_texts: Reference texts (target style)
            transferred_texts: Transferred texts (model output)
            train_classifier: Whether to train new fastText classifier
            classifier_model_path: Path to pre-trained classifier
            output_file: Output file path for results
            content_weight: Weight for content in DeepTransfer alignment
            style_weight: Weight for style in DeepTransfer alignment
            skip_catllm: Skip CAT-LLM metrics
            skip_zerostylus: Skip ZeroStylus metrics
            skip_deeptransfer: Skip DeepTransfer metrics

        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*80)
        print("EXTENDED UNIFIED STYLE TRANSFER EVALUATION")
        print("="*80)
        print(f"\nEvaluating {len(transferred_texts)} samples")
        print(f"CAT-LLM metrics: {'Enabled' if not skip_catllm else 'Disabled'}")
        print(f"ZeroStylus metrics: {'Enabled' if not skip_zerostylus else 'Disabled'}")
        print(f"DeepTransfer metrics: {'Enabled' if not skip_deeptransfer and self.enable_deeptransfer_metrics else 'Disabled'}")

        results = {}

        # ========== CAT-LLM Metrics ==========
        if not skip_catllm:
            print("\n" + "="*80)
            print("CAT-LLM METRICS")
            print("="*80)

            # 0. Train or load classifier
            if train_classifier:
                print("\n[Step 0: Training Style Classifier]")
                self.train_style_classifier(source_texts, reference_texts)
            elif classifier_model_path:
                print("\n[Step 0: Loading Style Classifier]")
                import fasttext
                self.classifier = fasttext.load_model(classifier_model_path)
                self.classifier_path = classifier_model_path

            # 1. BLEU - Content Preservation
            print("\n[CAT-LLM 1: BLEU for Content Preservation]")
            bleu_content = self.calculate_bleu(source_texts, transferred_texts)
            for key, value in bleu_content.items():
                results[f'{key}_content'] = value

            # 2. BLEU - Style Similarity
            print("\n[CAT-LLM 2: BLEU for Style Similarity]")
            bleu_style = self.calculate_bleu(reference_texts, transferred_texts)
            for key, value in bleu_style.items():
                results[f'{key}_style'] = value

            # 3. BERTScore - Content Preservation
            print("\n[CAT-LLM 3: BERTScore for Content Preservation]")
            bert_content = self.calculate_bert_score(source_texts, transferred_texts)
            for key, value in bert_content.items():
                results[f'{key}_content'] = value

            # 4. BERTScore - Style Similarity
            print("\n[CAT-LLM 4: BERTScore for Style Similarity]")
            bert_style = self.calculate_bert_score(reference_texts, transferred_texts)
            for key, value in bert_style.items():
                results[f'{key}_style'] = value

            # 5. Style Transfer Accuracy
            print("\n[CAT-LLM 5: Style Transfer Accuracy]")
            style_acc = self.calculate_style_transfer_accuracy(transferred_texts)
            results.update(style_acc)

            # 6. Perplexity
            print("\n[CAT-LLM 6: Perplexity]")
            ppl_scores = self.calculate_perplexity(transferred_texts)
            results.update(ppl_scores)

            # CAT-LLM Overall Score
            if bleu_content and bleu_style and style_acc:
                import numpy as np
                bleu_content_avg = np.mean([v for k, v in bleu_content.items()])
                bleu_style_avg = np.mean([v for k, v in bleu_style.items()])
                style_acc_val = style_acc['style_transfer_accuracy']

                if ppl_scores and 'perplexity_mean' in ppl_scores:
                    ppl_mean = ppl_scores['perplexity_mean']
                    fluency_score = np.exp(-ppl_mean / 100.0)
                    results['fluency_score'] = round(fluency_score, 4)

                    results['catllm_overall_score'] = round(
                        (bleu_content_avg * bleu_style_avg * style_acc_val * fluency_score) ** (1/4),
                        4
                    )
                else:
                    results['catllm_overall_score'] = round(
                        (bleu_content_avg * bleu_style_avg * style_acc_val) ** (1/3),
                        4
                    )

        # ========== ZeroStylus Metrics ==========
        if not skip_zerostylus:
            print("\n" + "="*80)
            print("ZEROSTYLUS METRICS")
            print("="*80)

            # 7. Style Consistency
            print("\n[ZeroStylus 7: Style Consistency]")
            style_consistency = self.calculate_style_consistency(
                transferred_texts, reference_texts
            )
            results.update(style_consistency)

            # 8. Content Preservation (ZeroStylus method)
            print("\n[ZeroStylus 8: Content Preservation]")
            content_pres_zs = self.calculate_content_preservation_zerostylus(
                transferred_texts, source_texts
            )
            results.update(content_pres_zs)

            # 9. Expression Quality
            print("\n[ZeroStylus 9: Expression Quality]")
            expr_quality = self.calculate_expression_quality(transferred_texts)
            results.update(expr_quality)

            # ZeroStylus Average Score
            if style_consistency and content_pres_zs and expr_quality:
                import numpy as np
                x_score = style_consistency['style_consistency_mean']
                y_score = content_pres_zs['content_preservation_zs_mean']
                z_score = expr_quality['expression_quality_mean']

                results['zerostylus_average_score'] = round(
                    (x_score + y_score + z_score) / 3,
                    4
                )

        # ========== DeepTransfer Metrics ==========
        if not skip_deeptransfer and self.enable_deeptransfer_metrics and self.dt_metrics:
            print("\n" + "="*80)
            print("DEEPTRANSFER METRICS")
            print("="*80)

            dt_results = self.dt_metrics.evaluate_all(
                source_texts=source_texts,
                reference_texts=reference_texts,
                transferred_texts=transferred_texts,
                content_weight=content_weight,
                style_weight=style_weight
            )
            results.update(dt_results)

        print("\n" + "="*80)
        print("EVALUATION COMPLETED")
        print("="*80)

        # Save results
        if output_file:
            self.save_results(results, output_file)

        return results

    def print_results_extended(self, results: Dict):
        """Print evaluation results including DeepTransfer metrics."""
        # Print base results (CAT-LLM and ZeroStylus)
        self.print_results(results)

        # Print DeepTransfer metrics
        if any(key.startswith('dt_') for key in results.keys()):
            print("\n" + "-"*80)
            print("DEEPTRANSFER METRICS")
            print("-"*80)

            if 'dt_content_preservation_mean' in results:
                print("\n[Content Preservation (Dual-Encoder)]")
                print(f"  Mean: {results['dt_content_preservation_mean']:.4f}")
                print(f"  Std: {results.get('dt_content_preservation_std', 0):.4f}")
                print(f"  Min: {results.get('dt_content_preservation_min', 0):.4f}")
                print(f"  Max: {results.get('dt_content_preservation_max', 0):.4f}")

            if 'dt_style_similarity_mean' in results:
                print("\n[Style Similarity (Dual-Encoder)]")
                print(f"  Mean: {results['dt_style_similarity_mean']:.4f}")
                print(f"  Std: {results.get('dt_style_similarity_std', 0):.4f}")
                print(f"  Min: {results.get('dt_style_similarity_min', 0):.4f}")
                print(f"  Max: {results.get('dt_style_similarity_max', 0):.4f}")

            if 'dt_disentanglement_score' in results:
                print("\n[Content-Style Disentanglement]")
                print(f"  Disentanglement Score: {results['dt_disentanglement_score']:.4f}")
                print(f"  Content Preservation: {results.get('dt_disentangle_content_preservation_mean', 0):.4f}")
                print(f"  Style Transfer: {results.get('dt_disentangle_style_transfer_mean', 0):.4f}")
                print(f"  Content Diff: {results.get('dt_disentangle_content_diff_mean', 0):.4f}")

            if 'dt_alignment_score_mean' in results:
                print("\n[Dual-Space Alignment]")
                print(f"  Score: {results['dt_alignment_score_mean']:.4f}")
                print(f"  Std: {results.get('dt_alignment_score_std', 0):.4f}")
                print(f"  Content Weight: {results.get('dt_alignment_content_weight', 0.5)}")
                print(f"  Style Weight: {results.get('dt_alignment_style_weight', 0.5)}")

            if 'dt_diversity_pairwise_dist_mean' in results:
                print("\n[Embedding Diversity]")
                print(f"  Pairwise Distance: {results['dt_diversity_pairwise_dist_mean']:.4f}")
                print(f"  Embedding Variance: {results.get('dt_diversity_embedding_variance', 0):.4f}")

            print("\n" + "="*80 + "\n")

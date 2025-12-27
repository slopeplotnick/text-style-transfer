"""
Unified Style Transfer Evaluation Module
Combines metrics from both CAT-LLM and ZeroStylus evaluation systems.

Metrics included:
1. BLEU scores (content and style)
2. BERTScore (content and style)
3. Style Transfer Accuracy (fastText classifier)
4. Perplexity (gpt2-large)
5. Style Consistency (embedding similarity)
6. Content Preservation (BLEURT + keyword retention)
7. Expression Quality (heuristics)
"""

import os
import json
import math
import glob
import tempfile
from typing import Dict, List, Optional, Callable
import numpy as np


class UnifiedStyleEvaluator:
    """Unified evaluator combining all metrics from both systems."""

    def __init__(self):
        """Initialize the unified evaluator."""
        self.metrics = {}
        self.classifier = None
        self.classifier_path = None
        self.encoder = None
        self.bleurt_scorer = None

    def _ensure_nltk_resources(self):
        """Ensure NLTK resources are downloaded."""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt resource...")
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                print("Downloading NLTK punkt_tab resource...")
                nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK resources: {e}")

    def _load_sentence_transformer(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load sentence transformer for embedding-based metrics."""
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading SentenceTransformer: {model_name}...")
                self.encoder = SentenceTransformer(model_name)
                print("SentenceTransformer loaded successfully")
            except ImportError:
                print("Warning: sentence-transformers not installed.")
                print("Install with: pip install sentence-transformers")

    def _load_bleurt(self):
        """Load BLEURT scorer if available."""
        if self.bleurt_scorer is None:
            try:
                from bleurt import score as bleurt_score
                print("Loading BLEURT scorer...")
                self.bleurt_scorer = bleurt_score.BleurtScorer()
                print("BLEURT scorer loaded successfully")
            except ImportError:
                print("Warning: BLEURT not available, will use embedding similarity instead")
                self.bleurt_scorer = None

    # ========== CAT-LLM Metrics ==========

    def calculate_bleu(
        self,
        references: List[str],
        hypotheses: List[str],
        max_n: int = 4
    ) -> Dict[str, float]:
        """Calculate BLEU scores using nltk."""
        try:
            self._ensure_nltk_resources()
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            smoothing = SmoothingFunction().method1
            bleu_scores = {f'bleu-{i}': [] for i in range(1, max_n + 1)}

            for ref, hyp in zip(references, hypotheses):
                ref_tokens = word_tokenize(ref.lower())
                hyp_tokens = word_tokenize(hyp.lower())

                for n in range(1, max_n + 1):
                    weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
                    score = sentence_bleu(
                        [ref_tokens],
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=smoothing
                    )
                    bleu_scores[f'bleu-{n}'].append(score)

            avg_bleu = {
                key: round(np.mean(values), 4)
                for key, values in bleu_scores.items()
            }

            return avg_bleu

        except ImportError:
            print("Warning: NLTK not installed. Skipping BLEU calculation.")
            print("Install with: pip install nltk")
            return {}

    def calculate_bert_score(
        self,
        references: List[str],
        hypotheses: List[str],
        model_type: str = "bert-base-uncased"
    ) -> Dict[str, float]:
        """Calculate BERTScore."""
        try:
            from bert_score import score

            P, R, F1 = score(
                hypotheses,
                references,
                model_type=model_type,
                verbose=False
            )

            return {
                'bert_precision': round(P.mean().item(), 4),
                'bert_recall': round(R.mean().item(), 4),
                'bert_f1': round(F1.mean().item(), 4)
            }

        except ImportError:
            print("Warning: bert-score not installed. Skipping BERTScore calculation.")
            print("Install with: pip install bert-score")
            return {}

    def train_style_classifier(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        model_path: Optional[str] = None,
        epoch: int = 25,
        lr: float = 0.1,
        wordNgrams: int = 2
    ) -> str:
        """Train fastText binary classifier for style classification."""
        try:
            import fasttext
        except ImportError:
            print("Error: fasttext not installed.")
            print("Install with: pip install fasttext")
            raise

        print("Training fastText style classifier...")

        training_data = []

        for text in source_texts:
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            if cleaned_text:
                training_data.append(f"__label__source {cleaned_text}")

        for text in reference_texts:
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            if cleaned_text:
                training_data.append(f"__label__target {cleaned_text}")

        print(f"Training data: {len(training_data)} samples")
        print(f"  - Source texts: {len(source_texts)}")
        print(f"  - Reference texts: {len(reference_texts)}")

        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            suffix='.txt',
            delete=False
        ) as f:
            train_file = f.name
            for line in training_data:
                f.write(line + '\n')

        try:
            model = fasttext.train_supervised(
                input=train_file,
                epoch=epoch,
                lr=lr,
                wordNgrams=wordNgrams,
                verbose=2
            )

            if model_path is None:
                model_path = os.path.join(
                    tempfile.gettempdir(),
                    'style_classifier.bin'
                )

            model.save_model(model_path)
            self.classifier = model
            self.classifier_path = model_path

            print(f"Classifier trained and saved to: {model_path}")

            return model_path

        finally:
            if os.path.exists(train_file):
                os.remove(train_file)

    def calculate_style_transfer_accuracy(
        self,
        transferred_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate style transfer accuracy using fastText classifier."""
        if self.classifier is None:
            print("Warning: Classifier not trained or loaded. Cannot calculate style accuracy.")
            return {
                'style_transfer_accuracy': 0.0,
                'style_transfer_std': 0.0
            }

        print("Calculating style transfer accuracy with fastText classifier...")

        correct_count = 0
        predictions = []

        for text in transferred_texts:
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()

            if not cleaned_text:
                predictions.append(0)
                continue

            prediction = self.classifier.predict(cleaned_text)
            label = prediction[0][0]

            if label == '__label__target':
                correct_count += 1
                predictions.append(1)
            else:
                predictions.append(0)

        accuracy = correct_count / len(transferred_texts) if transferred_texts else 0
        std = np.std(predictions)

        print(f"  Classified as target style: {correct_count}/{len(transferred_texts)}")
        print(f"  Accuracy: {accuracy:.4f}")

        return {
            'style_transfer_accuracy': round(accuracy, 4),
            'style_transfer_std': round(std, 4)
        }

    def calculate_perplexity(
        self,
        texts: List[str],
        model_name: str = "gpt2-large"
    ) -> Dict[str, float]:
        """Calculate perplexity using gpt2-large."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading {model_name} for perplexity calculation...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            print(f"Model loaded on device: {device}")

            perplexities = []

            with torch.no_grad():
                for i, text in enumerate(texts):
                    if (i + 1) % 10 == 0:
                        print(f"  Processing {i + 1}/{len(texts)}...")

                    encodings = tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(device)

                    outputs = model(**encodings, labels=encodings['input_ids'])
                    loss = outputs.loss

                    ppl = math.exp(loss.item())
                    perplexities.append(ppl)

            print(f"Perplexity calculation completed")

            return {
                'perplexity_mean': round(np.mean(perplexities), 4),
                'perplexity_std': round(np.std(perplexities), 4)
            }

        except ImportError:
            print("Warning: transformers not installed. Skipping perplexity calculation.")
            print("Install with: pip install transformers torch")
            return {}
        except Exception as e:
            print(f"Warning: Error calculating perplexity: {e}")
            return {}

    # ========== ZeroStylus Metrics ==========

    def calculate_style_consistency(
        self,
        transformed_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate style consistency using embedding similarity."""
        self._load_sentence_transformer()

        if self.encoder is None:
            print("Warning: SentenceTransformer not available. Skipping style consistency.")
            return {}

        print("Calculating style consistency...")

        # Encode all texts
        transformed_embs = self.encoder.encode(
            transformed_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        reference_embs = self.encoder.encode(
            reference_texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Calculate average similarity for each transformed text
        scores = []
        for trans_emb in transformed_embs:
            similarities = [
                self._cosine_similarity(trans_emb, ref_emb)
                for ref_emb in reference_embs
            ]
            avg_similarity = np.mean(similarities)
            # Convert to 0-10 scale
            score = avg_similarity * 10
            scores.append(max(0, min(10, score)))

        return {
            'style_consistency_mean': round(np.mean(scores), 4),
            'style_consistency_std': round(np.std(scores), 4)
        }

    def calculate_content_preservation_zerostylus(
        self,
        transformed_texts: List[str],
        original_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate content preservation using BLEURT/embedding + keyword retention."""
        self._load_sentence_transformer()
        self._load_bleurt()

        print("Calculating content preservation (ZeroStylus method)...")

        bleurt_scores = []
        keyword_scores = []

        for trans, orig in zip(transformed_texts, original_texts):
            # BLEURT or embedding similarity
            if self.bleurt_scorer is not None:
                bleurt_score = self._compute_bleurt(trans, orig)
            elif self.encoder is not None:
                orig_emb = self.encoder.encode(orig, convert_to_numpy=True)
                trans_emb = self.encoder.encode(trans, convert_to_numpy=True)
                bleurt_score = self._cosine_similarity(orig_emb, trans_emb)
            else:
                bleurt_score = 0.0

            bleurt_scores.append(bleurt_score)

            # Keyword retention
            keyword_score = self._compute_keyword_retention(trans, orig)
            keyword_scores.append(keyword_score)

        # Combine with weights (0.7 BLEURT, 0.3 keyword)
        combined_scores = [
            0.7 * b + 0.3 * k
            for b, k in zip(bleurt_scores, keyword_scores)
        ]

        # Convert to 0-10 scale
        final_scores = [score * 10 for score in combined_scores]

        return {
            'content_preservation_zs_mean': round(np.mean(final_scores), 4),
            'content_preservation_zs_std': round(np.std(final_scores), 4),
            'bleurt_component_mean': round(np.mean(bleurt_scores), 4),
            'keyword_retention_mean': round(np.mean(keyword_scores), 4)
        }

    def calculate_expression_quality(
        self,
        transformed_texts: List[str]
    ) -> Dict[str, float]:
        """Calculate expression quality using heuristic metrics."""
        print("Calculating expression quality...")

        scores = []

        for text in transformed_texts:
            score = self._evaluate_quality_heuristics(text)
            scores.append(score)

        return {
            'expression_quality_mean': round(np.mean(scores), 4),
            'expression_quality_std': round(np.std(scores), 4)
        }

    # ========== Helper Methods ==========

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _compute_bleurt(self, transformed_text: str, original_text: str) -> float:
        """Compute BLEURT score."""
        if self.bleurt_scorer is None:
            return 0.0

        try:
            scores = self.bleurt_scorer.score(
                references=[original_text],
                candidates=[transformed_text]
            )
            normalized = (scores[0] + 1) / 2
            return max(0, min(1, normalized))
        except Exception as e:
            print(f"BLEURT computation failed: {e}")
            return 0.0

    def _compute_keyword_retention(
        self,
        transformed_text: str,
        original_text: str,
        top_k: int = 10
    ) -> float:
        """Compute keyword retention recall."""
        from collections import Counter
        import re

        # Simple keyword extraction using TF
        def extract_keywords(text, top_k):
            words = re.findall(r'\b\w+\b', text.lower())
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                         'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                         'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                         'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them'}
            words = [w for w in words if w not in stop_words and len(w) > 2]
            counter = Counter(words)
            return [word for word, _ in counter.most_common(top_k)]

        original_keywords = set(extract_keywords(original_text, top_k))
        transformed_keywords = set(extract_keywords(transformed_text, top_k))

        if not original_keywords:
            return 1.0

        retained = original_keywords.intersection(transformed_keywords)
        retention_rate = len(retained) / len(original_keywords)

        return retention_rate

    def _evaluate_quality_heuristics(self, text: str) -> float:
        """Evaluate text quality using heuristic metrics."""
        import re

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Metric 1: Sentence length variation
        lengths = [len(s.split()) for s in sentences]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)

        length_score = 1.0
        if avg_length < 10 or avg_length > 30:
            length_score *= 0.8
        if std_length < 3 or std_length > 15:
            length_score *= 0.9

        # Metric 2: Vocabulary diversity
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        diversity_score = min(1.0, ttr * 2)

        # Metric 3: Readability
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        readability_score = 1.0
        if avg_sentence_length > 40:
            readability_score = 0.7
        elif avg_sentence_length < 5:
            readability_score = 0.8

        # Combine scores (equal weights)
        combined = (length_score + diversity_score + readability_score) / 3

        # Convert to 0-10 scale
        return combined * 10

    # ========== Unified Evaluation ==========

    def evaluate_all(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        transferred_texts: List[str],
        train_classifier: bool = True,
        classifier_model_path: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Complete unified evaluation with all metrics.

        Args:
            source_texts: Source texts (style-removed)
            reference_texts: Reference texts (original journal articles)
            transferred_texts: Transferred texts (output from style transfer model)
            train_classifier: Whether to train new classifier
            classifier_model_path: Path to pre-trained classifier
            output_file: Output file path

        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*80)
        print("UNIFIED STYLE TRANSFER EVALUATION")
        print("="*80)

        results = {}

        # 0. Train or load classifier
        if train_classifier:
            print("\n[Step 0: Training Style Classifier]")
            self.train_style_classifier(source_texts, reference_texts)
        elif classifier_model_path:
            print("\n[Step 0: Loading Style Classifier]")
            import fasttext
            self.classifier = fasttext.load_model(classifier_model_path)
            self.classifier_path = classifier_model_path

        # === CAT-LLM Metrics ===

        # 1. BLEU - Content Preservation
        print("\n[CAT-LLM 1: BLEU for Content Preservation]")
        print("  Comparing: source vs transferred")
        bleu_content = self.calculate_bleu(source_texts, transferred_texts)
        for key, value in bleu_content.items():
            results[f'{key}_content'] = value

        # 2. BLEU - Style Similarity
        print("\n[CAT-LLM 2: BLEU for Style Similarity]")
        print("  Comparing: reference vs transferred")
        bleu_style = self.calculate_bleu(reference_texts, transferred_texts)
        for key, value in bleu_style.items():
            results[f'{key}_style'] = value

        # 3. BERTScore - Content Preservation
        print("\n[CAT-LLM 3: BERTScore for Content Preservation]")
        print("  Comparing: source vs transferred")
        bert_content = self.calculate_bert_score(source_texts, transferred_texts)
        for key, value in bert_content.items():
            results[f'{key}_content'] = value

        # 4. BERTScore - Style Similarity
        print("\n[CAT-LLM 4: BERTScore for Style Similarity]")
        print("  Comparing: reference vs transferred")
        bert_style = self.calculate_bert_score(reference_texts, transferred_texts)
        for key, value in bert_style.items():
            results[f'{key}_style'] = value

        # 5. Style Transfer Accuracy
        print("\n[CAT-LLM 5: Style Transfer Accuracy]")
        style_acc = self.calculate_style_transfer_accuracy(transferred_texts)
        results.update(style_acc)

        # 6. Perplexity
        print("\n[CAT-LLM 6: Perplexity]")
        ppl_scores = self.calculate_perplexity(transferred_texts, model_name="gpt2-large")
        results.update(ppl_scores)

        # === ZeroStylus Metrics ===

        # 7. Style Consistency
        print("\n[ZeroStylus 7: Style Consistency]")
        style_consistency = self.calculate_style_consistency(transferred_texts, reference_texts)
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

        # === Composite Scores ===

        # CAT-LLM Overall Score
        if bleu_content and bleu_style and style_acc:
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

        # ZeroStylus Average Score
        if style_consistency and content_pres_zs and expr_quality:
            x_score = style_consistency['style_consistency_mean']
            y_score = content_pres_zs['content_preservation_zs_mean']
            z_score = expr_quality['expression_quality_mean']

            results['zerostylus_average_score'] = round(
                (x_score + y_score + z_score) / 3,
                4
            )

        print("\n" + "="*80)
        print("EVALUATION COMPLETED")
        print("="*80)

        # Save results
        if output_file:
            self.save_results(results, output_file)

        return results

    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_file}")

    def print_results(self, results: Dict):
        """Print evaluation results in a formatted manner."""
        print("\n" + "="*80)
        print("UNIFIED EVALUATION RESULTS")
        print("="*80)

        # CAT-LLM Metrics
        print("\n" + "-"*80)
        print("CAT-LLM METRICS")
        print("-"*80)

        print("\n[Content Preservation - BLEU]")
        for key in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
            result_key = f'{key}_content'
            if result_key in results:
                print(f"  {key}: {results[result_key]:.4f}")

        print("\n[Style Similarity - BLEU]")
        for key in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
            result_key = f'{key}_style'
            if result_key in results:
                print(f"  {key}: {results[result_key]:.4f}")

        if 'bert_f1_content' in results:
            print("\n[Content Preservation - BERTScore]")
            print(f"  Precision: {results.get('bert_precision_content', 0):.4f}")
            print(f"  Recall: {results.get('bert_recall_content', 0):.4f}")
            print(f"  F1: {results.get('bert_f1_content', 0):.4f}")

        if 'bert_f1_style' in results:
            print("\n[Style Similarity - BERTScore]")
            print(f"  Precision: {results.get('bert_precision_style', 0):.4f}")
            print(f"  Recall: {results.get('bert_recall_style', 0):.4f}")
            print(f"  F1: {results.get('bert_f1_style', 0):.4f}")

        print("\n[Style Transfer Accuracy]")
        print(f"  Accuracy: {results.get('style_transfer_accuracy', 0):.4f}")

        if 'perplexity_mean' in results:
            print("\n[Fluency - Perplexity]")
            print(f"  Mean: {results.get('perplexity_mean', 0):.4f}")
            print(f"  Std: {results.get('perplexity_std', 0):.4f}")
            if 'fluency_score' in results:
                print(f"  Fluency Score: {results.get('fluency_score', 0):.4f}")

        if 'catllm_overall_score' in results:
            print("\n[CAT-LLM Overall Score]")
            print(f"  Score: {results['catllm_overall_score']:.4f}")

        # ZeroStylus Metrics
        print("\n" + "-"*80)
        print("ZEROSTYLUS METRICS")
        print("-"*80)

        if 'style_consistency_mean' in results:
            print("\n[Style Consistency]")
            print(f"  Mean: {results['style_consistency_mean']:.4f}")
            print(f"  Std: {results.get('style_consistency_std', 0):.4f}")

        if 'content_preservation_zs_mean' in results:
            print("\n[Content Preservation]")
            print(f"  Mean: {results['content_preservation_zs_mean']:.4f}")
            print(f"  Std: {results.get('content_preservation_zs_std', 0):.4f}")
            print(f"  - BLEURT Component: {results.get('bleurt_component_mean', 0):.4f}")
            print(f"  - Keyword Retention: {results.get('keyword_retention_mean', 0):.4f}")

        if 'expression_quality_mean' in results:
            print("\n[Expression Quality]")
            print(f"  Mean: {results['expression_quality_mean']:.4f}")
            print(f"  Std: {results.get('expression_quality_std', 0):.4f}")

        if 'zerostylus_average_score' in results:
            print("\n[ZeroStylus Average Score]")
            print(f"  Score: {results['zerostylus_average_score']:.4f}")

        print("\n" + "="*80 + "\n")


def load_txt_files(directory: str) -> List[str]:
    """Load all .txt files from a directory."""
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    if not txt_files:
        print(f"Warning: No .txt files found in {directory}")
        return []

    texts = []
    for file_path in sorted(txt_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Loaded {len(texts)} texts from {directory}")
    return texts


def evaluate_model(
    model_name: str,
    reference_dir: str,
    source_dir: str,
    transferred_dir: str,
    output_file: str,
    evaluator: UnifiedStyleEvaluator
):
    """Evaluate a single model."""
    print("\n" + "="*80)
    print(f"EVALUATING: {model_name}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    reference_texts = load_txt_files(reference_dir)
    source_texts = load_txt_files(source_dir)
    transferred_texts = load_txt_files(transferred_dir)

    # Check data
    if not reference_texts or not source_texts or not transferred_texts:
        print(f"Error: Missing data for {model_name}")
        return None

    min_len = min(len(reference_texts), len(source_texts), len(transferred_texts))
    print(f"\nUsing {min_len} samples (minimum of all sets)")

    reference_texts = reference_texts[:min_len]
    source_texts = source_texts[:min_len]
    transferred_texts = transferred_texts[:min_len]

    # Evaluate
    results = evaluator.evaluate_all(
        source_texts=source_texts,
        reference_texts=reference_texts,
        transferred_texts=transferred_texts,
        train_classifier=True,
        output_file=output_file
    )

    # Print results
    evaluator.print_results(results)

    return results


def main():
    """Main evaluation script."""
    # Initialize evaluator
    evaluator = UnifiedStyleEvaluator()

    # Define paths
    reference_dir = "/root/datasets/text_style_transfer/ipm_abstracts"
    source_dir = "/root/datasets/text_style_transfer/ipm_abstracts/style_removed"

    # Evaluate CAT-LLM
    catllm_transferred_dir = "/root/rzy/tst/CAT-LLM/output-gpt-5-2025-08-07-20251030_090303/transferred"
    catllm_output_file = "evaluation_results_catllm.json"

    catllm_results = evaluate_model(
        model_name="CAT-LLM",
        reference_dir=reference_dir,
        source_dir=source_dir,
        transferred_dir=catllm_transferred_dir,
        output_file=catllm_output_file,
        evaluator=evaluator
    )

    # Evaluate ZeroStylus
    zerostylus_transferred_dir = "/root/rzy/tst/ZeroStylus/outputs/batch_transformed"
    zerostylus_output_file = "evaluation_results_zerostylus.json"

    zerostylus_results = evaluate_model(
        model_name="ZeroStylus",
        reference_dir=reference_dir,
        source_dir=source_dir,
        transferred_dir=zerostylus_transferred_dir,
        output_file=zerostylus_output_file,
        evaluator=evaluator
    )

    # Compare results
    if catllm_results and zerostylus_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        print("\n[CAT-LLM]")
        print(f"  Overall Score: {catllm_results.get('catllm_overall_score', 'N/A')}")
        print(f"  Style Accuracy: {catllm_results.get('style_transfer_accuracy', 'N/A')}")
        print(f"  BLEU-4 Content: {catllm_results.get('bleu-4_content', 'N/A')}")
        print(f"  BLEU-4 Style: {catllm_results.get('bleu-4_style', 'N/A')}")

        print("\n[ZeroStylus]")
        print(f"  Average Score: {zerostylus_results.get('zerostylus_average_score', 'N/A')}")
        print(f"  Style Consistency: {zerostylus_results.get('style_consistency_mean', 'N/A')}")
        print(f"  Content Preservation: {zerostylus_results.get('content_preservation_zs_mean', 'N/A')}")
        print(f"  Expression Quality: {zerostylus_results.get('expression_quality_mean', 'N/A')}")

        print("\n" + "="*80)


if __name__ == "__main__":
    main()

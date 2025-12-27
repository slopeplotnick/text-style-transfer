"""
Journal Style Transfer Evaluation V2
改进版评估模块:
1. 使用fastText二分类器计算风格迁移准确率
2. 使用gpt2-large计算困惑度
3. 保持BLEU和BERTScore不变
"""

import os
import json
import math
import tempfile
from typing import Dict, List, Tuple, Optional
import numpy as np


class JournalStyleEvaluator:
    """学术期刊风格转换评估器 V2"""

    def __init__(self):
        """初始化评估器"""
        self.metrics = {}
        self.classifier = None
        self.classifier_path = None

    def _ensure_nltk_resources(self):
        """确保NLTK资源已下载"""
        try:
            import nltk
            # 尝试使用word_tokenize，如果失败则下载资源
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

    def calculate_bleu(
        self,
        references: List[str],
        hypotheses: List[str],
        max_n: int = 4
    ) -> Dict[str, float]:
        """
        计算BLEU分数(使用nltk实现)

        Args:
            references: 参考文本列表
            hypotheses: 生成文本列表
            max_n: 最大n-gram

        Returns:
            BLEU分数字典
        """
        try:
            # 确保NLTK资源已下载
            self._ensure_nltk_resources()

            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from nltk.tokenize import word_tokenize

            smoothing = SmoothingFunction().method1

            bleu_scores = {f'bleu-{i}': [] for i in range(1, max_n + 1)}

            for ref, hyp in zip(references, hypotheses):
                # 分词
                ref_tokens = word_tokenize(ref.lower())
                hyp_tokens = word_tokenize(hyp.lower())

                # 计算不同n-gram的BLEU
                for n in range(1, max_n + 1):
                    weights = tuple([1.0/n] * n + [0.0] * (max_n - n))
                    score = sentence_bleu(
                        [ref_tokens],
                        hyp_tokens,
                        weights=weights,
                        smoothing_function=smoothing
                    )
                    bleu_scores[f'bleu-{n}'].append(score)

            # 计算平均值
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
        """
        计算BERTScore

        Args:
            references: 参考文本列表
            hypotheses: 生成文本列表
            model_type: BERT模型类型

        Returns:
            BERTScore字典
        """
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
        """
        训练fastText二分类器用于风格分类
        source_texts标记为 __label__source
        reference_texts标记为 __label__target

        Args:
            source_texts: 源文本列表(去风格后的文本)
            reference_texts: 参考文本列表(原始期刊文章)
            model_path: 模型保存路径(如果为None,使用临时文件)
            epoch: 训练轮数
            lr: 学习率
            wordNgrams: n-gram大小

        Returns:
            模型保存路径
        """
        try:
            import fasttext
        except ImportError:
            print("Error: fasttext not installed.")
            print("Install with: pip install fasttext")
            raise

        print("Training fastText style classifier...")

        # 准备训练数据
        training_data = []

        # 添加source文本(标签: __label__source)
        for text in source_texts:
            # 清理文本，去除换行符
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            if cleaned_text:
                training_data.append(f"__label__source {cleaned_text}")

        # 添加reference文本(标签: __label__target)
        for text in reference_texts:
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()
            if cleaned_text:
                training_data.append(f"__label__target {cleaned_text}")

        print(f"Training data: {len(training_data)} samples")
        print(f"  - Source texts: {len(source_texts)}")
        print(f"  - Reference texts: {len(reference_texts)}")

        # 创建临时训练文件
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
            # 训练模型
            model = fasttext.train_supervised(
                input=train_file,
                epoch=epoch,
                lr=lr,
                wordNgrams=wordNgrams,
                verbose=2
            )

            # 保存模型
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
            # 清理临时训练文件
            if os.path.exists(train_file):
                os.remove(train_file)

    def load_style_classifier(self, model_path: str):
        """
        加载已训练的fastText分类器

        Args:
            model_path: 模型文件路径
        """
        try:
            import fasttext
        except ImportError:
            print("Error: fasttext not installed.")
            print("Install with: pip install fasttext")
            raise

        print(f"Loading classifier from: {model_path}")
        self.classifier = fasttext.load_model(model_path)
        self.classifier_path = model_path
        print("Classifier loaded successfully")

    def calculate_style_transfer_accuracy(
        self,
        transferred_texts: List[str]
    ) -> Dict[str, float]:
        """
        使用fastText分类器计算风格迁移准确率
        统计有多少transferred_texts被分类为 __label__target

        Args:
            transferred_texts: 转换后的文本列表

        Returns:
            风格迁移准确率字典
        """
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
            # 清理文本
            cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()

            if not cleaned_text:
                predictions.append(0)
                continue

            # 预测
            prediction = self.classifier.predict(cleaned_text)
            label = prediction[0][0]  # 获取预测标签

            # 如果分类为target风格，则认为风格转换成功
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
        """
        计算困惑度(流畅度指标)
        改进: 使用gpt2-large模型

        Args:
            texts: 文本列表
            model_name: 语言模型名称(默认: gpt2-large)

        Returns:
            困惑度字典
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading {model_name} for perplexity calculation...")

            # 加载模型
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

                    # 编码
                    encodings = tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512
                    ).to(device)

                    # 计算loss
                    outputs = model(**encodings, labels=encodings['input_ids'])
                    loss = outputs.loss

                    # 计算perplexity
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

    def evaluate(
        self,
        source_texts: List[str],
        reference_texts: List[str],
        transferred_texts: List[str],
        train_classifier: bool = True,
        classifier_model_path: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, any]:
        """
        完整评估流程

        Args:
            source_texts: 源文本列表(去风格后的文本)
            reference_texts: 参考文本列表(原始期刊文章)
            transferred_texts: 转换后的文本列表
            train_classifier: 是否训练新的分类器(如果False,需要提供classifier_model_path)
            classifier_model_path: 已训练的分类器路径
            output_file: 输出文件路径

        Returns:
            评估结果字典
        """
        print("Starting evaluation with V2 evaluator...")
        results = {}

        # 0. 训练或加载分类器
        if train_classifier:
            print("\n[Step 0: Training Style Classifier]")
            self.train_style_classifier(source_texts, reference_texts)
        elif classifier_model_path:
            print("\n[Step 0: Loading Style Classifier]")
            self.load_style_classifier(classifier_model_path)
        else:
            print("Warning: No classifier available. Style accuracy will be 0.")

        # 1. 内容保留度评估 (BLEU: source vs transferred)
        print("\n[Step 1: Calculating BLEU scores for content preservation]")
        print("  Comparing: source_texts vs transferred_texts")
        bleu_content_scores = self.calculate_bleu(source_texts, transferred_texts)
        # 重命名键名以区分内容保留度
        for key, value in bleu_content_scores.items():
            results[f'{key}_content'] = value

        # 2. 风格相似度评估 (BLEU: reference vs transferred)
        print("\n[Step 2: Calculating BLEU scores for style similarity]")
        print("  Comparing: reference_texts vs transferred_texts")
        bleu_style_scores = self.calculate_bleu(reference_texts, transferred_texts)
        # 重命名键名以区分风格相似度
        for key, value in bleu_style_scores.items():
            results[f'{key}_style'] = value

        # 3. 内容保留度评估 (BERTScore: source vs transferred)
        print("\n[Step 3: Calculating BERTScore for content preservation]")
        print("  Comparing: source_texts vs transferred_texts")
        bert_content_scores = self.calculate_bert_score(source_texts, transferred_texts)
        # 重命名键名以区分内容保留度
        for key, value in bert_content_scores.items():
            results[f'{key}_content'] = value

        # 4. 风格相似度评估 (BERTScore: reference vs transferred)
        print("\n[Step 4: Calculating BERTScore for style similarity]")
        print("  Comparing: reference_texts vs transferred_texts")
        bert_style_scores = self.calculate_bert_score(reference_texts, transferred_texts)
        # 重命名键名以区分风格相似度
        for key, value in bert_style_scores.items():
            results[f'{key}_style'] = value

        # 5. 风格迁移准确率 (使用fastText分类器)
        print("\n[Step 5: Calculating Style Transfer Accuracy]")
        style_scores = self.calculate_style_transfer_accuracy(transferred_texts)
        results.update(style_scores)

        # 6. 流畅度评估 (Perplexity with gpt2-large)
        print("\n[Step 6: Calculating Perplexity with gpt2-large]")
        ppl_scores = self.calculate_perplexity(transferred_texts, model_name="gpt2-large")
        results.update(ppl_scores)

        # 7. 综合指标
        if bleu_content_scores and bleu_style_scores and style_scores:
            # 计算内容保留度的平均BLEU
            bleu_content_avg = np.mean([v for k, v in bleu_content_scores.items()])
            # 计算风格相似度的平均BLEU
            bleu_style_avg = np.mean([v for k, v in bleu_style_scores.items()])
            # 风格准确率
            style_acc = style_scores['style_transfer_accuracy']

            # 归一化困惑度 (越低越好 -> 越高越好)
            # 使用公式: fluency_score = exp(-perplexity/100)
            # 这样perplexity=0时score=1, perplexity=100时score≈0.37
            if ppl_scores and 'perplexity_mean' in ppl_scores:
                ppl_mean = ppl_scores['perplexity_mean']
                fluency_score = np.exp(-ppl_mean / 100.0)
                results['fluency_score'] = round(fluency_score, 4)

                # 综合得分: 包含流畅度的四维几何平均数
                # (内容保留 × 风格相似 × 风格准确 × 流畅度) ^ (1/4)
                results['overall_score'] = round(
                    (bleu_content_avg * bleu_style_avg * style_acc * fluency_score) ** (1/4),
                    4
                )
            else:
                # 如果没有困惑度，使用三维几何平均数
                results['overall_score'] = round(
                    (bleu_content_avg * bleu_style_avg * style_acc) ** (1/3),
                    4
                )

        print("\nEvaluation completed!")

        # 保存结果
        if output_file:
            self.save_results(results, output_file)

        return results

    def save_results(self, results: Dict, output_file: str):
        """
        保存评估结果

        Args:
            results: 评估结果
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_file}")

    def print_results(self, results: Dict):
        """
        打印评估结果

        Args:
            results: 评估结果
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (V2)")
        print("=" * 60)

        # 1. 内容保留度 - BLEU (source vs transferred)
        print("\n[Content Preservation - BLEU]")
        print("  (source_texts vs transferred_texts)")
        for key in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
            result_key = f'{key}_content'
            if result_key in results:
                print(f"    {key}: {results[result_key]:.4f}")

        # 2. 风格相似度 - BLEU (reference vs transferred)
        print("\n[Style Similarity - BLEU]")
        print("  (reference_texts vs transferred_texts)")
        for key in ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
            result_key = f'{key}_style'
            if result_key in results:
                print(f"    {key}: {results[result_key]:.4f}")

        # 3. 内容保留度 - BERTScore (source vs transferred)
        if 'bert_f1_content' in results:
            print("\n[Content Preservation - BERTScore]")
            print("  (source_texts vs transferred_texts)")
            print(f"    Precision: {results.get('bert_precision_content', 0):.4f}")
            print(f"    Recall: {results.get('bert_recall_content', 0):.4f}")
            print(f"    F1: {results.get('bert_f1_content', 0):.4f}")

        # 4. 风格相似度 - BERTScore (reference vs transferred)
        if 'bert_f1_style' in results:
            print("\n[Style Similarity - BERTScore]")
            print("  (reference_texts vs transferred_texts)")
            print(f"    Precision: {results.get('bert_precision_style', 0):.4f}")
            print(f"    Recall: {results.get('bert_recall_style', 0):.4f}")
            print(f"    F1: {results.get('bert_f1_style', 0):.4f}")

        # 5. 风格迁移准确率 (fastText)
        print("\n[Style Transfer Accuracy - fastText Classifier]")
        print(f"  Accuracy: {results.get('style_transfer_accuracy', 0):.4f}")
        print(f"  Std: {results.get('style_transfer_std', 0):.4f}")

        # 6. 流畅度 (gpt2-large)
        if 'perplexity_mean' in results:
            print("\n[Fluency - Perplexity (gpt2-large)]")
            print(f"  Mean: {results.get('perplexity_mean', 0):.4f}")
            print(f"  Std: {results.get('perplexity_std', 0):.4f}")
            if 'fluency_score' in results:
                print(f"  Fluency Score (normalized): {results.get('fluency_score', 0):.4f}")

        # 7. 综合分数
        if 'overall_score' in results:
            print("\n[Overall Score]")
            print(f"  Score: {results.get('overall_score', 0):.4f}")
            if 'fluency_score' in results:
                print("  (Geometric mean of content_bleu × style_bleu × style_accuracy × fluency)")
            else:
                print("  (Geometric mean of content_bleu × style_bleu × style_accuracy)")

        print("=" * 60 + "\n")


def main():
    """主函数"""
    # 示例使用
    evaluator = JournalStyleEvaluator()

    # 示例数据
    source_texts = [
        "This is a simple text that needs to be transformed.",
        "Another example of source text for evaluation."
    ]

    reference_texts = [
        "This research demonstrates a comprehensive approach that systematically addresses the fundamental challenges.",
        "Furthermore, this investigation reveals significant insights regarding the methodological considerations."
    ]

    transferred_texts = [
        "This paper demonstrates a fundamental approach that requires systematic transformation.",
        "Moreover, this study presents substantial findings concerning the comprehensive evaluation procedures."
    ]

    # 执行评估
    results = evaluator.evaluate(
        source_texts=source_texts,
        reference_texts=reference_texts,
        transferred_texts=transferred_texts,
        train_classifier=True,
        output_file='evaluation_results_v2.json'
    )

    # 打印结果
    evaluator.print_results(results)


if __name__ == "__main__":
    main()

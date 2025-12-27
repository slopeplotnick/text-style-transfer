"""
Async Main Pipeline for Journal Style Transfer
异步版本主流程：支持并发处理，大幅提升效率
"""

import os
import argparse
import json
import glob
import time
from typing import List, Dict, Optional

from journal_style_define import JournalStyleAnalyzer
from style_removal_async import AsyncStyleRemover
from journal_style_transfer_async import AsyncJournalStyleTransfer
from journal_style_evaluation import JournalStyleEvaluator


class AsyncJournalStylePipeline:
    """异步学术期刊风格转换完整流程"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        output_dir: str = "./output_async",
        max_concurrent: int = 10  # 最大并发数
    ):
        """
        初始化异步流程

        Args:
            api_url: API地址
            api_key: API密钥
            model_name: 模型名称
            output_dir: 输出目录
            max_concurrent: 最大并发请求数
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent

        # 创建输出目录结构
        self.style_removed_dir = os.path.join(output_dir, "style_removed")
        self.transferred_dir = os.path.join(output_dir, "transferred")
        self.eval_dir = os.path.join(output_dir, "evaluation")

        for dir_path in [self.style_removed_dir, self.transferred_dir, self.eval_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 初始化各个模块
        self.style_analyzer = JournalStyleAnalyzer()

        # 异步处理器
        self.async_style_remover = AsyncStyleRemover(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            max_concurrent=max_concurrent
        )

        self.async_style_transfer = None  # 稍后初始化（需要风格描述）

        self.evaluator = JournalStyleEvaluator()

        # 风格描述
        self.target_style_desc = None

    def run_full_pipeline(
        self,
        data_dir: str,
        skip_style_removal: bool = False,
        skip_transfer: bool = False,
        skip_evaluation: bool = False
    ):
        """
        运行完整流程（异步版本）

        Args:
            data_dir: 数据目录
            skip_style_removal: 是否跳过风格去除
            skip_transfer: 是否跳过风格转换
            skip_evaluation: 是否跳过评估
        """
        print("=" * 80)
        print("ASYNC JOURNAL STYLE TRANSFER PIPELINE")
        print("=" * 80)
        print(f"Max concurrent requests: {self.max_concurrent}")
        print("=" * 80)

        pipeline_start_time = time.time()

        # Step 1: 分析data目录生成聚合风格描述
        print("\n[Step 1/5] Analyzing all articles to generate aggregated style description...")
        step_start = time.time()

        self.target_style_desc = self.style_analyzer.analyze_all_texts(data_dir)

        # 保存风格描述
        style_desc_file = os.path.join(self.output_dir, "target_style_description.txt")
        with open(style_desc_file, 'w', encoding='utf-8') as f:
            f.write(self.target_style_desc)

        step_time = time.time() - step_start
        print(f"[Step 1/5] Completed in {step_time:.2f}s")
        print(f"Style description saved to: {style_desc_file}")

        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        print(f"\nFound {len(txt_files)} files to process")

        # Step 2: 异步批量风格去除
        if not skip_style_removal:
            print("\n[Step 2/5] Async batch removing style from all articles...")
            step_start = time.time()

            removal_stats = self.async_style_remover.batch_process(
                input_dir=data_dir,
                output_dir=self.style_removed_dir
            )

            step_time = time.time() - step_start
            print(f"[Step 2/5] Completed in {step_time:.2f}s")
            print(f"Style removal completed for {len(removal_stats)} files")

            # 保存统计
            stats_file = os.path.join(self.output_dir, "removal_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(removal_stats, f, indent=2, ensure_ascii=False)

        else:
            print("\n[Step 2/5] Skipping style removal (using existing files)")

        # Step 3: 异步批量风格转换
        if not skip_transfer:
            print("\n[Step 3/5] Async batch transferring style for all articles...")
            step_start = time.time()

            # 初始化风格转换器（使用聚合的风格描述）
            self.async_style_transfer = AsyncJournalStyleTransfer(
                api_url=self.api_url,
                api_key=self.api_key,
                model_name=self.model_name,
                target_style=self.target_style_desc,
                max_concurrent=self.max_concurrent
            )

            # 批量异步处理
            transfer_stats = self.async_style_transfer.batch_process(
                input_dir=self.style_removed_dir,
                output_dir=self.transferred_dir
            )

            step_time = time.time() - step_start
            print(f"[Step 3/5] Completed in {step_time:.2f}s")
            print(f"Style transfer completed for {len(transfer_stats)} files")

            # 保存统计
            stats_file = os.path.join(self.output_dir, "transfer_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(transfer_stats, f, indent=2, ensure_ascii=False)

        else:
            print("\n[Step 3/5] Skipping style transfer (using existing files)")

        # Step 4: 收集所有数据用于评估
        if not skip_evaluation:
            print("\n[Step 4/5] Collecting data for evaluation...")
            step_start = time.time()

            evaluation_data = self._collect_evaluation_data(data_dir)

            step_time = time.time() - step_start
            print(f"[Step 4/5] Completed in {step_time:.2f}s")
            print(f"Collected {len(evaluation_data)} samples for evaluation")

            # Step 5: 聚合评估
            print("\n[Step 5/5] Evaluating style transfer quality...")
            step_start = time.time()

            eval_results = self._run_aggregate_evaluation(evaluation_data)

            step_time = time.time() - step_start
            print(f"[Step 5/5] Completed in {step_time:.2f}s")

            # 保存评估结果
            eval_results_file = os.path.join(self.eval_dir, "aggregate_evaluation_results.json")
            with open(eval_results_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)

            print(f"Evaluation results saved to: {eval_results_file}")

            # 打印结果
            self.evaluator.print_results(eval_results)
        else:
            print("\n[Step 4/5] Skipping evaluation (as requested)")
            print("[Step 5/5] Skipping evaluation (as requested)")

        # 总结
        pipeline_time = time.time() - pipeline_start_time

        print("\n" + "=" * 80)
        print("ASYNC PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"Total pipeline time: {pipeline_time:.2f}s ({pipeline_time/60:.2f} minutes)")
        print(f"All results saved to: {self.output_dir}")
        print("=" * 80)

    def _collect_evaluation_data(self, data_dir: str) -> List[Dict]:
        """
        收集评估数据

        Args:
            data_dir: 原始数据目录

        Returns:
            评估数据列表
        """
        evaluation_data = []

        # 获取所有原始文件
        original_files = glob.glob(os.path.join(data_dir, "*.txt"))

        for original_file in original_files:
            basename = os.path.basename(original_file).replace('.txt', '')

            # 对应的文件路径
            source_file = os.path.join(self.style_removed_dir, f"{basename}_style_removed.txt")
            transferred_file = os.path.join(self.transferred_dir, f"{basename}_transferred.txt")

            # 检查文件是否存在
            if not os.path.exists(source_file):
                print(f"Warning: Source file not found for {basename}")
                continue

            if not os.path.exists(transferred_file):
                print(f"Warning: Transferred file not found for {basename}")
                continue

            # 读取文本
            try:
                with open(original_file, 'r', encoding='utf-8') as f:
                    reference_text = f.read()

                with open(source_file, 'r', encoding='utf-8') as f:
                    source_text = f.read()

                with open(transferred_file, 'r', encoding='utf-8') as f:
                    transferred_text = f.read()

                evaluation_data.append({
                    'id': basename,
                    'reference': reference_text,
                    'source': source_text,
                    'transferred': transferred_text
                })

            except Exception as e:
                print(f"Error reading files for {basename}: {e}")
                continue

        return evaluation_data

    def _run_aggregate_evaluation(self, evaluation_data: List[Dict]) -> Dict:
        """
        对整个数据集进行聚合评估

        Args:
            evaluation_data: 评估数据列表

        Returns:
            聚合评估结果
        """
        if not evaluation_data:
            print("Warning: No evaluation data available")
            return {}

        # 提取所有文本
        source_texts = []
        reference_texts = []
        transferred_texts = []

        for item in evaluation_data:
            # 按段落分割
            source_paragraphs = [p.strip() for p in item['source'].split('\n\n') if p.strip()]
            reference_paragraphs = [p.strip() for p in item['reference'].split('\n\n') if p.strip()]
            transferred_paragraphs = [p.strip() for p in item['transferred'].split('\n\n') if p.strip()]

            # 只取最小长度,确保对齐
            min_len = min(len(source_paragraphs), len(reference_paragraphs), len(transferred_paragraphs))

            source_texts.extend(source_paragraphs[:min_len])
            reference_texts.extend(reference_paragraphs[:min_len])
            transferred_texts.extend(transferred_paragraphs[:min_len])

        print(f"Total paragraphs for evaluation: {len(transferred_texts)}")

        # 执行聚合评估 (使用V2评估器,会自动训练fastText分类器)
        eval_results = self.evaluator.evaluate(
            source_texts=source_texts,
            reference_texts=reference_texts,
            transferred_texts=transferred_texts,
            train_classifier=True  # 训练新的分类器
        )

        # 添加数据集信息
        eval_results['dataset_info'] = {
            'num_articles': len(evaluation_data),
            'num_paragraphs': len(transferred_texts),
            'articles': [item['id'] for item in evaluation_data]
        }

        return eval_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Async Journal Style Transfer Pipeline (10x faster!)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Data directory containing journal articles'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output_async',
        help='Output directory'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=os.getenv('OPENAI_API_URL', 'https://yunwu.zeabur.app/v1/chat/completions'),
        help='API URL'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENAI_API_KEY', ''),
        help='API key (required, set via --api-key or OPENAI_API_KEY env var)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-2025-08-07',
        help='Model name'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent requests (default: 10)'
    )

    parser.add_argument(
        '--skip-style-removal',
        action='store_true',
        help='Skip style removal step'
    )

    parser.add_argument(
        '--skip-transfer',
        action='store_true',
        help='Skip transfer step and only evaluate'
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step (useful for batch processing)'
    )

    args = parser.parse_args()

    if not args.api_key and (not args.skip_style_removal or not args.skip_transfer):
        print("Error: API key is required for style removal and transfer")
        print("Please set OPENAI_API_KEY environment variable or use --api-key argument")
        print("Or use --skip-style-removal and --skip-transfer to only evaluate existing files")
        return

    # 创建异步pipeline
    pipeline = AsyncJournalStylePipeline(
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent
    )

    # 运行pipeline
    pipeline.run_full_pipeline(
        data_dir=args.data_dir,
        skip_style_removal=args.skip_style_removal,
        skip_transfer=args.skip_transfer,
        skip_evaluation=args.skip_evaluation
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Async Journal Style Transfer Pipeline")
        print("\nUsage:")
        print("  python main_pipeline_async.py --data-dir <data_directory> [options]")
        print("\nOptions:")
        print("  --data-dir PATH           Data directory with journal articles (required)")
        print("  --output-dir PATH         Output directory (default: ./output_async)")
        print("  --api-url URL             API URL")
        print("  --api-key KEY             API key")
        print("  --model NAME              Model name (default: gpt-5-2025-08-07)")
        print("  --max-concurrent N        Maximum concurrent requests (default: 10)")
        print("  --skip-style-removal      Skip style removal step")
        print("  --skip-transfer           Skip transfer and only evaluate")
        print("  --skip-evaluation         Skip evaluation step (useful for batch processing)")
        print("\nEnvironment variables:")
        print("  OPENAI_API_URL           API URL")
        print("  OPENAI_API_KEY           API key")
        print("\nExample:")
        print("  python main_pipeline_async.py --data-dir ./data --max-concurrent 10")
    else:
        main()

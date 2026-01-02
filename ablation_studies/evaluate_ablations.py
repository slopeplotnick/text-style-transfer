"""
Evaluate All Ablation Experiments
使用unified_test_framework评估所有消融实验结果
计算完整的指标体系（CAT-LLM + ZeroStylus + DeepTransfer）

用法：
    python evaluate_ablations.py
    python evaluate_ablations.py --ablation dt_no_specter  # 只评估单个
"""
import os
import sys
import argparse
import json
from datetime import datetime

# 添加路径
sys.path.insert(0, '/root/rzy/tst')
sys.path.insert(0, '/root/rzy/tst/unified_test_framework')

from unified_test_framework.test_runner import TestRunner
from unified_test_framework.config import TestConfig

# 消融实验输出目录映射
ABLATION_OUTPUT_DIRS = {
    'DT-NoSPECTER': '/root/rzy/tst/output/ablation_dt_no_specter',
    'DT-Sentence': '/root/rzy/tst/output/ablation_dt_sentence',
    'DT-NoRerank': '/root/rzy/tst/output/ablation_dt_no_rerank',
}

# 内部名称映射
INTERNAL_NAMES = {
    'dt_no_specter': 'DT-NoSPECTER',
    'dt_sentence': 'DT-Sentence',
    'dt_no_rerank': 'DT-NoRerank',
}


def evaluate_single_ablation(model_name: str, output_dir: str, runner: TestRunner = None) -> dict:
    """
    评估单个消融实验

    Args:
        model_name: 模型名称（用于报告）
        output_dir: 转换结果输出目录
        runner: TestRunner实例（可选，用于复用）

    Returns:
        评估结果字典
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ABLATION: {model_name}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")

    if not os.path.exists(output_dir):
        print(f"Warning: Output directory not found: {output_dir}")
        return None

    # 检查是否有输出文件
    transferred_files = [f for f in os.listdir(output_dir) if f.endswith('.txt') and f.startswith('transferred_')]
    if not transferred_files:
        print(f"Warning: No transferred files found in {output_dir}")
        return None

    print(f"Found {len(transferred_files)} transferred files")

    # 初始化runner（如果没有传入）
    if runner is None:
        runner = TestRunner()

    # 使用完整的评估（CAT-LLM + ZeroStylus + DeepTransfer指标）
    results = runner.test_single_model(
        model_name=model_name,
        transferred_dir=output_dir,
        naming_pattern='auto',
        enable_deeptransfer_metrics=True,  # 启用DeepTransfer指标
        skip_catllm=False,       # 不跳过CAT-LLM指标
        skip_zerostylus=False,   # 不跳过ZeroStylus指标
        skip_deeptransfer=False  # 不跳过DeepTransfer指标
    )

    return results


def evaluate_all_ablations():
    """
    评估所有消融实验
    使用完整的指标体系（60+个指标）
    """
    print(f"\n{'#'*80}")
    print(f"# EVALUATING ALL ABLATION EXPERIMENTS")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    print(f"\nAblations to evaluate:")
    for name, dir_path in ABLATION_OUTPUT_DIRS.items():
        exists = "✓" if os.path.exists(dir_path) else "✗"
        print(f"  {exists} {name}: {dir_path}")

    # 初始化TestRunner（复用以提高效率）
    print("\nInitializing TestRunner...")
    runner = TestRunner()

    all_results = {}
    successful = 0
    failed = 0

    for model_name, output_dir in ABLATION_OUTPUT_DIRS.items():
        try:
            results = evaluate_single_ablation(model_name, output_dir, runner)
            if results:
                all_results[model_name] = results
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 打印汇总
    print(f"\n{'#'*80}")
    print(f"# ABLATION EVALUATION COMPLETED")
    print(f"{'#'*80}")
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if all_results:
        print(f"\nKey metrics comparison:")
        print("-" * 100)
        print(f"{'Model':<15} {'CAT-LLM Overall':>15} {'DT Style Sim':>15} {'DT Content':>15} {'Style Acc':>12}")
        print("-" * 100)

        for model_name, results in all_results.items():
            catllm_overall = results.get('catllm_overall_score', 'N/A')
            dt_style = results.get('dt_style_similarity_mean', 'N/A')
            dt_content = results.get('dt_content_preservation_mean', 'N/A')
            style_acc = results.get('style_transfer_accuracy', 'N/A')

            catllm_str = f"{catllm_overall:.4f}" if isinstance(catllm_overall, float) else str(catllm_overall)
            dt_style_str = f"{dt_style:.4f}" if isinstance(dt_style, float) else str(dt_style)
            dt_content_str = f"{dt_content:.4f}" if isinstance(dt_content, float) else str(dt_content)
            style_acc_str = f"{style_acc:.4f}" if isinstance(style_acc, float) else str(style_acc)

            print(f"{model_name:<15} {catllm_str:>15} {dt_style_str:>15} {dt_content_str:>15} {style_acc_str:>12}")

        print("-" * 100)

    print(f"\nResults saved to: {TestConfig.RESULTS_DIR}")
    print(f"{'#'*80}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation experiments")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=list(INTERNAL_NAMES.keys()),
        default=None,
        help="Specific ablation to evaluate (default: all)"
    )

    args = parser.parse_args()

    if args.ablation:
        # 评估单个消融
        model_name = INTERNAL_NAMES[args.ablation]
        output_dir = ABLATION_OUTPUT_DIRS[model_name]
        evaluate_single_ablation(model_name, output_dir)
    else:
        # 评估所有消融
        evaluate_all_ablations()


if __name__ == "__main__":
    main()

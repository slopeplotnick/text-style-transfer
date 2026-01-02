"""
Run All Ablation Experiments
依次运行所有消融实验

用法：
    python run_all_ablations.py --max-samples 500
"""
import os
import sys
import argparse
import time
import json
import subprocess
from datetime import datetime

# =============================================================================
# 配置代理（用于下载HuggingFace模型）
# =============================================================================
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 添加当前目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# 所有消融实验
ABLATIONS = ['dt_no_specter', 'dt_sentence', 'dt_no_rerank']

# 默认路径
DEFAULT_SOURCE_DIR = "/root/datasets/arxiv_style_transfer/source"
DEFAULT_DATA_DIR = "/root/datasets/arxiv_style_transfer/reference"


def run_all_ablations(
    source_dir: str,
    data_dir: str,
    max_samples: int = None,
    skip_existing: bool = False,
    start_from: str = None
):
    """
    运行所有消融实验

    Args:
        source_dir: 源文件目录
        data_dir: 参考数据目录
        max_samples: 最大样本数
        skip_existing: 是否跳过已完成的实验
        start_from: 从指定的消融实验开始运行
    """
    # 确定要运行的消融实验列表
    ablations_to_run = ABLATIONS.copy()
    if start_from:
        if start_from not in ABLATIONS:
            print(f"Error: Unknown ablation '{start_from}'. Available: {ABLATIONS}")
            return
        start_idx = ABLATIONS.index(start_from)
        ablations_to_run = ABLATIONS[start_idx:]
        print(f"Starting from ablation: {start_from} (skipping {ABLATIONS[:start_idx]})")

    print(f"\n{'#'*80}")
    print(f"# RUNNING ALL ABLATION EXPERIMENTS")
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")
    print(f"\nAblations to run: {ablations_to_run}")
    print(f"Source directory: {source_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Skip existing: {skip_existing}")

    all_start_time = time.time()
    results = {}

    for i, ablation in enumerate(ablations_to_run):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(ablations_to_run)}] Running ablation: {ablation}")
        print(f"{'='*80}")

        # 检查是否已完成
        output_dir = f"/root/rzy/tst/output/ablation_{ablation}"
        stats_file = os.path.join(output_dir, 'ablation_stats.json')

        if skip_existing and os.path.exists(stats_file):
            print(f"Skipping {ablation} - already completed")
            with open(stats_file, 'r') as f:
                results[ablation] = json.load(f)
            continue

        # 运行消融实验
        ablation_start = time.time()

        try:
            # 使用subprocess运行，以确保每次都是干净的Python环境
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, 'run_ablation.py'),
                '--ablation', ablation,
                '--source-dir', source_dir,
                '--data-dir', data_dir,
            ]

            if max_samples:
                cmd.extend(['--max-samples', str(max_samples)])

            # 运行子进程
            process = subprocess.run(
                cmd,
                cwd=SCRIPT_DIR,
                capture_output=False,  # 直接输出到终端
                text=True
            )

            if process.returncode != 0:
                raise RuntimeError(f"Ablation {ablation} failed with return code {process.returncode}")

            # 读取结果
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    results[ablation] = json.load(f)
            else:
                results[ablation] = {
                    'status': 'completed',
                    'elapsed_time': time.time() - ablation_start
                }

        except Exception as e:
            print(f"\nError running ablation {ablation}: {e}")
            results[ablation] = {
                'status': 'failed',
                'error': str(e),
                'elapsed_time': time.time() - ablation_start
            }

    # 计算总时间
    total_time = time.time() - all_start_time

    # 汇总结果
    summary = {
        'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'ablations': results
    }

    # 保存汇总结果
    summary_file = os.path.join(SCRIPT_DIR, 'all_ablations_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f"\n{'#'*80}")
    print(f"# ALL ABLATION EXPERIMENTS COMPLETED")
    print(f"{'#'*80}")
    print(f"\nSummary:")
    for ablation, result in results.items():
        status = result.get('status', 'completed')
        if status == 'failed':
            print(f"  - {ablation}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            successful = result.get('successful', 'N/A')
            total_files = result.get('total_files', 'N/A')
            print(f"  - {ablation}: {successful}/{total_files} files processed")

    print(f"\nTotal time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Summary saved to: {summary_file}")
    print(f"{'#'*80}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run all ablation experiments")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing source texts"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing reference data (for building index)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of samples to process (default: 500)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ablations that have already been completed"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=ABLATIONS,
        default=None,
        help=f"Start from a specific ablation (choices: {ABLATIONS})"
    )

    args = parser.parse_args()

    run_all_ablations(
        source_dir=args.source_dir,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        skip_existing=args.skip_existing,
        start_from=args.start_from
    )


if __name__ == "__main__":
    main()

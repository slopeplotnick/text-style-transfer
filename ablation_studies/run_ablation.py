"""
Run Single Ablation Experiment
运行单个消融实验

用法：
    python run_ablation.py --ablation dt_no_specter --max-samples 500
    python run_ablation.py --ablation dt_sentence --max-samples 500
    python run_ablation.py --ablation dt_no_rerank --max-samples 500
"""
import os
import sys
import asyncio
import argparse
import time
import json
from pathlib import Path

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

# 支持的消融实验
ABLATIONS = ['dt_no_specter', 'dt_sentence', 'dt_no_rerank']

# 默认路径
DEFAULT_SOURCE_DIR = "/root/datasets/arxiv_style_transfer/source"
DEFAULT_DATA_DIR = "/root/datasets/arxiv_style_transfer/reference"


def get_config_class(ablation_name: str):
    """动态获取消融配置类"""
    if ablation_name == "dt_no_specter":
        from configs.dt_no_specter import Config
    elif ablation_name == "dt_sentence":
        from configs.dt_sentence import Config
    elif ablation_name == "dt_no_rerank":
        from configs.dt_no_rerank import Config
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}. Supported: {ABLATIONS}")
    return Config


def inject_config(Config):
    """
    将消融配置注入到DeepTransfer的config模块
    这样当DeepTransfer的pipeline导入Config时，会使用我们的消融配置
    """
    # 创建一个模块对象
    config_module = type(sys)('config')
    config_module.Config = Config

    # 注入到sys.modules
    sys.modules['config'] = config_module

    return Config


def build_index_if_needed(Config, data_dir: str):
    """如果索引不存在则构建"""
    index_path = Config.get_index_path()

    if os.path.exists(index_path):
        print(f"Index already exists: {index_path}")
        return True

    print(f"\n{'='*80}")
    print(f"Building index for ablation: {Config.ABLATION_NAME}")
    print(f"Index path: {index_path}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*80}")

    # 确保目录存在
    Config.ensure_dirs()

    # 设置INDEX_FILE为完整路径
    Config.INDEX_FILE = index_path

    # 导入并构建索引
    sys.path.insert(0, '/root/rzy/tst/DeepTransfer')
    from pipeline import DeepTransferPipeline

    pipeline = DeepTransferPipeline()
    pipeline.build_index(data_dir)

    print(f"\n✓ Index built successfully: {index_path}")
    return True


async def run_batch_transfer(Config, source_dir: str, output_dir: str, max_samples: int):
    """运行批量转换"""
    print(f"\n{'='*80}")
    print(f"Running batch transfer for ablation: {Config.ABLATION_NAME}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples}")
    print(f"{'='*80}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设置INDEX_FILE为完整路径
    Config.INDEX_FILE = Config.get_index_path()

    # 导入异步处理模块
    sys.path.insert(0, '/root/rzy/tst/DeepTransfer')

    # 重新加载模块以使用新配置
    import importlib
    if 'pipeline_async' in sys.modules:
        del sys.modules['pipeline_async']
    if 'batch_transfer_async' in sys.modules:
        del sys.modules['batch_transfer_async']

    from batch_transfer_async import batch_process_async

    # 运行批量处理
    results = await batch_process_async(
        source_dir=source_dir,
        output_dir=output_dir,
        max_samples=max_samples,
        max_concurrent=20,
        pattern="src_*.txt",
        resume=True  # 启用断点续传
    )

    return results


def rename_outputs_to_transferred(output_dir: str):
    """
    将输出文件从 ref_xxxx.txt 重命名为 transferred_xxxx.txt
    以与其他方法的输出格式保持一致
    """
    import glob

    ref_files = glob.glob(os.path.join(output_dir, "ref_*.txt"))
    renamed_count = 0

    for ref_file in ref_files:
        basename = os.path.basename(ref_file)
        # ref_0000.txt -> transferred_0000.txt
        new_basename = basename.replace("ref_", "transferred_")
        new_path = os.path.join(output_dir, new_basename)

        try:
            os.rename(ref_file, new_path)
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {ref_file}: {e}")

    print(f"Renamed {renamed_count} files from ref_xxxx.txt to transferred_xxxx.txt")


def run_ablation(
    ablation_name: str,
    source_dir: str,
    data_dir: str,
    max_samples: int = None,
    skip_build: bool = False
):
    """
    运行单个消融实验

    Args:
        ablation_name: 消融实验名称
        source_dir: 源文件目录
        data_dir: 参考数据目录（用于构建索引）
        max_samples: 最大样本数
        skip_build: 是否跳过索引构建
    """
    print(f"\n{'#'*80}")
    print(f"# ABLATION EXPERIMENT: {ablation_name}")
    print(f"{'#'*80}")

    start_time = time.time()

    # 1. 获取并注入配置
    Config = get_config_class(ablation_name)
    inject_config(Config)

    # 2. 获取输出目录
    output_dir = Config.get_output_dir()

    # 3. 构建索引（如果需要）
    if not skip_build:
        build_index_if_needed(Config, data_dir)

    # 4. 运行批量转换
    results = asyncio.run(run_batch_transfer(Config, source_dir, output_dir, max_samples))

    # 5. 重命名输出文件 (ref_xxxx.txt -> transferred_xxxx.txt)
    rename_outputs_to_transferred(output_dir)

    # 6. 计算统计
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r.get('success', False))

    # 7. 保存统计信息
    stats = {
        'ablation_name': ablation_name,
        'total_files': len(results),
        'successful': success_count,
        'failed': len(results) - success_count,
        'total_time_seconds': total_time,
        'config': {
            'CONTENT_ENCODER_MODEL': Config.CONTENT_ENCODER_MODEL,
            'STYLE_ENCODER_MODEL': Config.STYLE_ENCODER_MODEL,
            'CHUNK_SIZE': Config.CHUNK_SIZE,
            'ENABLE_STYLE_RERANKING': Config.ENABLE_STYLE_RERANKING,
        }
    }

    stats_file = os.path.join(output_dir, 'ablation_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"ABLATION {ablation_name} COMPLETED")
    print(f"{'='*80}")
    print(f"  Total files: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results) - success_count}")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"  Output: {output_dir}")
    print(f"  Stats: {stats_file}")
    print(f"{'='*80}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run single ablation experiment")
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=ABLATIONS,
        help=f"Ablation experiment to run. Choices: {ABLATIONS}"
    )
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
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip index building (use existing index)"
    )

    args = parser.parse_args()

    run_ablation(
        ablation_name=args.ablation,
        source_dir=args.source_dir,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        skip_build=args.skip_build
    )


if __name__ == "__main__":
    main()

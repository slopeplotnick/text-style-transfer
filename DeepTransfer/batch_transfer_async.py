"""
DeepTransfer Async Batch Processing Script
异步批处理：支持并发处理多个文件，大幅提升效率
"""
import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline_async import AsyncDeepTransferPipeline
from config import Config

try:
    import aiofiles
except ImportError:
    print("Warning: aiofiles not installed. Install with: pip install aiofiles")
    aiofiles = None


async def process_file_async(
    pipeline: AsyncDeepTransferPipeline,
    source_file: Path,
    output_dir: str,
    file_id: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """
    异步处理单个文件

    Args:
        pipeline: Pipeline 实例
        source_file: 源文件路径
        output_dir: 输出目录
        file_id: 文件标识（用于日志）
        semaphore: 并发控制信号量

    Returns:
        处理统计信息
    """
    async with semaphore:
        start_time = time.time()

        try:
            # Read source text (异步)
            if aiofiles:
                async with aiofiles.open(source_file, 'r', encoding='utf-8') as f:
                    source_text = await f.read()
            else:
                # Fallback to sync read in executor
                loop = asyncio.get_event_loop()
                source_text = await loop.run_in_executor(
                    None,
                    lambda: open(source_file, 'r', encoding='utf-8').read()
                )

            if not source_text.strip():
                print(f"[{file_id}] Warning: Empty file, skipping")
                return {
                    'file': source_file.name,
                    'success': False,
                    'error': 'Empty file',
                    'processing_time': 0
                }

            # Perform style transfer (异步)
            transferred_text = await pipeline.transfer_style_async(source_text)

            # Save result (异步)
            base_name = source_file.stem.replace('src_', 'ref_')
            output_file = os.path.join(output_dir, f"{base_name}.txt")

            if aiofiles:
                async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                    await f.write(transferred_text)
            else:
                # Fallback to sync write in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: open(output_file, 'w', encoding='utf-8').write(transferred_text)
                )

            elapsed_time = time.time() - start_time

            print(f"[{file_id}] ✓ Saved to {output_file} ({elapsed_time:.2f}s)")

            return {
                'file': source_file.name,
                'success': True,
                'output_file': output_file,
                'processing_time': elapsed_time
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[{file_id}] ✗ Error: {e}")
            import traceback
            traceback.print_exc()

            return {
                'file': source_file.name,
                'success': False,
                'error': str(e),
                'processing_time': elapsed_time
            }


async def batch_process_async(
    source_dir: str,
    output_dir: str,
    max_samples: int = None,
    max_concurrent: int = 10,
    use_journal_filter: bool = False,
    use_section_filter: bool = False,
    pattern: str = "src_*.txt",
    batch_size: int = 20  # 新增：每批处理的文件数
) -> List[Dict]:
    """
    异步批处理多个文件

    Args:
        source_dir: 源文件目录
        output_dir: 输出目录
        max_samples: 最大处理样本数
        max_concurrent: 最大并发数
        use_journal_filter: 是否使用期刊过滤
        use_section_filter: 是否使用章节过滤
        pattern: 文件匹配模式
        batch_size: 每批处理的文件数（避免任务爆炸）

    Returns:
        处理统计信息列表
    """
    print("=" * 80)
    print("DeepTransfer Async Batch Processing")
    print("=" * 80)

    # Initialize pipeline
    print(f"\nInitializing async pipeline...")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Batch size: {batch_size} files/batch")
    print(f"  Journal filter: {use_journal_filter}")
    print(f"  Section filter: {use_section_filter}")

    pipeline = AsyncDeepTransferPipeline(
        use_journal_filter=use_journal_filter,
        use_section_filter=use_section_filter,
        max_concurrent=max_concurrent
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all source files
    source_files = sorted(Path(source_dir).glob(pattern))

    if max_samples:
        source_files = source_files[:max_samples]

    if not source_files:
        print(f"\nError: No files matching '{pattern}' found in {source_dir}")
        return []

    print(f"\nFound {len(source_files)} files to process")
    print(f"Output directory: {output_dir}")
    print(f"\nStarting async batch processing...")
    print("=" * 80)

    start_time = time.time()

    # ===== 优化：使用 Semaphore 控制并发 + 取消批次串行化 =====
    semaphore = asyncio.Semaphore(max_concurrent)

    # 创建所有任务（一次性），但通过 semaphore 控制并发数
    tasks = []
    for i, source_file in enumerate(source_files):
        file_id = f"{i+1}/{len(source_files)}"
        task = process_file_async(pipeline, source_file, output_dir, file_id, semaphore)
        tasks.append(task)

    # 并发执行所有任务（semaphore 自动控制并发数）
    results = await asyncio.gather(*tasks)

    # Calculate statistics
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    error_count = len(results) - success_count

    # Summary
    print("\n" + "=" * 80)
    print("ASYNC BATCH PROCESSING COMPLETED")
    print("=" * 80)
    print(f"  Total files: {len(source_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    if success_count > 0:
        print(f"  Average time per file: {total_time/len(source_files):.2f}s")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepTransfer Async Batch Processing")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/root/datasets/arxiv_style_transfer/source",
        help="Directory containing source texts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/batch_transformed",
        help="Output directory for transferred texts"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = all)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--use-journal-filter",
        action="store_true",
        help="Enable journal-based retrieval filtering"
    )
    parser.add_argument(
        "--use-section-filter",
        action="store_true",
        help="Enable section-based retrieval filtering"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="src_*.txt",
        help="File pattern to match (default: src_*.txt)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for building index (if index doesn't exist)"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip automatic index building even if index doesn't exist"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of files to process per batch (default: 20, prevents task explosion)"
    )

    args = parser.parse_args()

    # Check if index exists, build if necessary
    config = Config()
    if not os.path.exists(config.INDEX_FILE):
        print("=" * 80)
        print(f"⚠ Index file not found: {config.INDEX_FILE}")
        print("=" * 80)

        if args.skip_build:
            print("✗ Skipping index build (--skip-build flag set)")
            print("Please build index first using: python main.py build --data_dir <path>")
            return

        # Determine data directory
        data_dir = args.data_dir if args.data_dir else config.DEFAULT_DATA_DIR

        if not os.path.exists(data_dir):
            print(f"✗ Data directory not found: {data_dir}")
            print("Please specify --data-dir or ensure default data directory exists")
            return

        print(f"\nBuilding index from: {data_dir}")
        print("This is a one-time operation and may take a few minutes...")
        print("-" * 80)

        # Create temporary pipeline to build index
        # Note: Using sync pipeline for index building
        from pipeline import DeepTransferPipeline
        temp_pipeline = DeepTransferPipeline()
        temp_pipeline.build_index(data_dir)

        print("-" * 80)
        print(f"✓ Index built successfully: {config.INDEX_FILE}")
        print("=" * 80)
        print()

    # Run async batch processing
    results = asyncio.run(
        batch_process_async(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_concurrent=args.max_concurrent,
            use_journal_filter=args.use_journal_filter,
            use_section_filter=args.use_section_filter,
            pattern=args.pattern,
            batch_size=args.batch_size
        )
    )

    # Save statistics
    if results:
        import json
        stats_file = os.path.join(args.output_dir, "batch_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    main()

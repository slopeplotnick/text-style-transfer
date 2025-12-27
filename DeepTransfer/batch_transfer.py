"""
DeepTransfer Batch Processing Script
Processes multiple source files and applies style transfer
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import DeepTransferPipeline
from config import Config

def main():
    import argparse

    parser = argparse.ArgumentParser(description="DeepTransfer Batch Processing")
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

    args = parser.parse_args()

    print("="*80)
    print("DeepTransfer Batch Processing")
    print("="*80)

    # Check if index exists, build if necessary
    config = Config()
    if not os.path.exists(config.INDEX_FILE):
        print(f"\n⚠ Index file not found: {config.INDEX_FILE}")

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

        print(f"Building index from: {data_dir}")
        print("This is a one-time operation and may take a few minutes...")
        print("-" * 80)

        # Create temporary pipeline to build index
        temp_pipeline = DeepTransferPipeline()
        temp_pipeline.build_index(data_dir)

        print("-" * 80)
        print(f"✓ Index built successfully: {config.INDEX_FILE}")
        print()

    # Initialize pipeline
    print(f"\nInitializing pipeline...")
    print(f"  Journal filter: {args.use_journal_filter}")
    print(f"  Section filter: {args.use_section_filter}")

    pipeline = DeepTransferPipeline(
        use_journal_filter=args.use_journal_filter,
        use_section_filter=args.use_section_filter
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all source files
    source_files = sorted(Path(args.source_dir).glob(args.pattern))

    if args.max_samples:
        source_files = source_files[:args.max_samples]

    if not source_files:
        print(f"\nError: No files matching '{args.pattern}' found in {args.source_dir}")
        return

    print(f"\nFound {len(source_files)} files to process")
    print(f"Output directory: {args.output_dir}")

    # Process each file
    success_count = 0
    error_count = 0

    for i, source_file in enumerate(source_files):
        print(f"\n[{i+1}/{len(source_files)}] Processing {source_file.name}...")

        try:
            # Read source text
            with open(source_file, 'r', encoding='utf-8') as f:
                source_text = f.read()

            if not source_text.strip():
                print(f"  Warning: Empty file, skipping")
                error_count += 1
                continue

            # Perform style transfer
            transferred_text = pipeline.transfer_style(source_text)

            # Save result (match naming pattern)
            # If input is src_0000.txt, output is ref_0000.txt (to match evaluation framework)
            base_name = source_file.stem.replace('src_', 'ref_')
            output_file = os.path.join(args.output_dir, f"{base_name}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transferred_text)

            print(f"  ✓ Saved to {output_file}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETED")
    print("="*80)
    print(f"  Total files: {len(source_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

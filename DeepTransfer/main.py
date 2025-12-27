import argparse
import sys
import os

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import DeepTransferPipeline
from config import Config

def main():
    parser = argparse.ArgumentParser(description="DeepTransfer: Style-Aware RAG for Text Style Transfer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Build Command
    build_parser = subparsers.add_parser("build", help="Build dual-granularity vector index from data directory")
    build_parser.add_argument("--data_dir", type=str, default=Config.DEFAULT_DATA_DIR, help="Path to data directory")
    build_parser.add_argument("--verbose", action="store_true", help="Show detailed statistics after building")

    # Transfer Command
    transfer_parser = subparsers.add_parser("transfer", help="Transfer style of an input text or file")
    transfer_parser.add_argument("--input", type=str, help="Input text string")
    transfer_parser.add_argument("--file", type=str, help="Input text file path")
    transfer_parser.add_argument("--output", type=str, help="Output file path to save result")
    transfer_parser.add_argument("--journal", type=str, help="Target journal name (must match subfolder name in data_dir)")
    transfer_parser.add_argument("--use-journal-filter", action="store_true", help="Enable journal-based retrieval filtering")
    transfer_parser.add_argument("--use-section-filter", action="store_true", help="Enable section-based retrieval filtering")

    args = parser.parse_args()

    # Initialize pipeline with filter settings
    pipeline = DeepTransferPipeline(
        use_journal_filter=args.use_journal_filter if args.command == "transfer" else None,
        use_section_filter=args.use_section_filter if args.command == "transfer" else None
    )

    if args.command == "build":
        import glob

        print("=" * 80)
        print("Building Dual-Granularity Index for DeepTransfer")
        print("=" * 80)

        # Check data directory
        data_dir = args.data_dir
        if not os.path.exists(data_dir):
            print(f"\n✗ Error: Data directory not found: {data_dir}")
            return

        # Count files
        txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        print(f"\nFound {len(txt_files)} text files in {data_dir}")

        if len(txt_files) == 0:
            print("\n✗ Error: No .txt files found in data directory.")
            return

        # Build index
        print(f"\nConfiguration:")
        print(f"  - Dual-Granularity: {pipeline.config.USE_DUAL_GRANULARITY}")
        print(f"  - Chunk Size: {pipeline.config.CHUNK_SIZE} sentences")
        print(f"  - Content Encoder: {pipeline.config.CONTENT_ENCODER_MODEL}")
        print(f"  - Style Encoder: {pipeline.config.STYLE_ENCODER_MODEL}")
        print()

        pipeline.build_index(data_dir)

        # Show statistics if verbose
        if args.verbose or pipeline.config.USE_DUAL_GRANULARITY:
            print("\n" + "=" * 80)
            print("Index Statistics:")
            print(f"  - Chunk-level documents: {len(pipeline.store.chunk_documents)}")
            print(f"  - Paragraph-level documents: {len(pipeline.store.para_documents)}")

            if pipeline.store.chunk_documents:
                sample_meta = pipeline.store.chunk_metadata[0]
                print(f"\nSample Metadata:")
                print(f"  - Journal: {sample_meta.get('journal', 'N/A')}")
                print(f"  - Section: {sample_meta.get('section_type', 'N/A')}")
                print(f"  - Source: {sample_meta.get('source', 'N/A')}")

            print("=" * 80)

        print(f"\n✓ Index successfully saved to: {pipeline.config.INDEX_FILE}")
        print("\nNext steps:")
        print(f"  python main.py transfer --file your_input.txt --journal Nature")

    elif args.command == "transfer":
        text = ""
        if args.input:
            text = args.input
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print("Please provide --input or --file")
            return

        if not text:
            print("Input text is empty.")
            return

        result = pipeline.transfer_style(text, target_journal=args.journal)

        print("\n=== Transfer Complete ===\n")
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Result saved to {args.output}")
        else:
            print(result[:500] + "...\n[Output truncated, use --output to save full text]")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
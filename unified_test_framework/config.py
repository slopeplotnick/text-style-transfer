"""
Configuration for Unified Style Transfer Testing Framework
"""
import os

class TestConfig:
    """Configuration for testing CAT-LLM, ZeroStylus, and DeepTransfer"""

    # ========== Dataset Paths ==========
    DATASET_BASE_DIR = "/root/datasets/arxiv_style_transfer"
    REFERENCE_DIR = os.path.join(DATASET_BASE_DIR, "reference")
    SOURCE_DIR = os.path.join(DATASET_BASE_DIR, "source")
    METADATA_FILE = os.path.join(DATASET_BASE_DIR, "metadata.json")

    # ========== Model Output Paths ==========
    # Unified output directory (as per run_all_500_unified.sh)
    OUTPUT_BASE_DIR = "/root/rzy/tst/output"

    # CAT-LLM outputs
    CATLLM_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "catllm")

    # ZeroStylus outputs
    ZEROSTYLUS_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "zerostylus")

    # DeepTransfer outputs
    DEEPTRANSFER_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "deeptransfer")

    # ========== Evaluation Results Output ==========
    RESULTS_DIR = "/root/rzy/tst/unified_test_framework/results"

    # ========== DeepTransfer Encoder Models ==========
    CONTENT_ENCODER_MODEL = "allenai/specter2_base"
    STYLE_ENCODER_MODEL = "AnnaWegmann/Style-Embedding"

    # ========== Evaluation Settings ==========
    # Maximum number of samples to evaluate (None = all)
    MAX_SAMPLES = None

    # Batch size for encoding
    ENCODING_BATCH_SIZE = 32

    # Whether to use GPU for encoding
    USE_GPU = True

    # Classifier settings
    CLASSIFIER_EPOCH = 25
    CLASSIFIER_LR = 0.1
    CLASSIFIER_WORD_NGRAMS = 2

    # Cache settings
    ENABLE_CACHE = True
    CACHE_DIR = "/root/rzy/tst/unified_test_framework/cache"

    # ========== Report Settings ==========
    GENERATE_JSON = True
    GENERATE_MARKDOWN = True
    GENERATE_CSV = True

    # ========== Metric Weights ==========
    # For Dual-Space Alignment Score
    CONTENT_WEIGHT = 0.5
    STYLE_WEIGHT = 0.5

    @classmethod
    def ensure_dirs(cls):
        """Ensure all output directories exist"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        if cls.ENABLE_CACHE:
            os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def get_model_output_dir(cls, model_name: str) -> str:
        """Get output directory for a specific model"""
        model_dirs = {
            'catllm': cls.CATLLM_OUTPUT_DIR,
            'zerostylus': cls.ZEROSTYLUS_OUTPUT_DIR,
            'deeptransfer': cls.DEEPTRANSFER_OUTPUT_DIR
        }
        return model_dirs.get(model_name.lower())

    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("UNIFIED TEST FRAMEWORK CONFIGURATION")
        print("="*80)
        print(f"\nDataset:")
        print(f"  Base Directory: {cls.DATASET_BASE_DIR}")
        print(f"  Reference: {cls.REFERENCE_DIR}")
        print(f"  Source: {cls.SOURCE_DIR}")
        print(f"\nModel Outputs:")
        print(f"  CAT-LLM: {cls.CATLLM_OUTPUT_DIR}")
        print(f"  ZeroStylus: {cls.ZEROSTYLUS_OUTPUT_DIR}")
        print(f"  DeepTransfer: {cls.DEEPTRANSFER_OUTPUT_DIR}")
        print(f"\nResults:")
        print(f"  Output Directory: {cls.RESULTS_DIR}")
        print(f"\nSettings:")
        print(f"  Max Samples: {cls.MAX_SAMPLES or 'All'}")
        print(f"  Batch Size: {cls.ENCODING_BATCH_SIZE}")
        print(f"  Use GPU: {cls.USE_GPU}")
        print(f"  Enable Cache: {cls.ENABLE_CACHE}")
        print("="*80 + "\n")

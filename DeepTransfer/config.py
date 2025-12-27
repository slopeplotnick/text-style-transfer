import os

class Config:
    # Model Configurations
    CONTENT_ENCODER_MODEL = "allenai/specter2_base"
    STYLE_ENCODER_MODEL = "AnnaWegmann/Style-Embedding"
    
    # Vector Store
    INDEX_FILE = "deep_transfer_index.pkl"
    
    # Text Processing
    CHUNK_SIZE = 3  # Sentences per chunk
    USE_DUAL_GRANULARITY = True  # Enable dual-granularity indexing (chunk + paragraph)

    # Retrieval Strategy
    USE_JOURNAL_FILTER = False  # Whether to use journal name in retrieval
    USE_SECTION_FILTER = False  # Whether to use section type in retrieval

    # Dual-Granularity Settings
    PARAGRAPH_TOP_K = 3  # Number of paragraph templates to retrieve
    CHUNK_TOP_K = 2  # Number of chunk examples to retrieve per chunk
    ENABLE_COHERENCE_ENHANCEMENT = True  # Whether to use paragraph-level coherence enhancement

    # Style-Aware Reranking Settings
    ENABLE_STYLE_RERANKING = True  # Whether to enable style-aware reranking
    STYLE_RERANKING_RETRIEVE_K = 10  # Number of candidates to retrieve before reranking
    STYLE_RERANKING_WEIGHT = 0.3  # Weight for style similarity (0.0-1.0)
                                   # Final score = content_score * (1 - weight) + style_score * weight
                                   # Higher values = more emphasis on style matching

    # LLM Configuration
    API_URL = os.getenv("OPENAI_API_URL", "https://newapi.deepwisdom.ai/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = "gpt-4o"
    
    # Paths
    DEFAULT_DATA_DIR = "../data"
    DEFAULT_OUTPUT_DIR = "./output"

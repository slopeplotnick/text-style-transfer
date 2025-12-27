"""
Configuration Loader for ZeroStylus

This module provides functionality to load and manage configuration from YAML files.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration manager for ZeroStylus framework.

    Loads configuration from YAML file and provides easy access to parameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to config.yaml file. If None, searches in current directory.
        """
        if config_path is None:
            # Search for config.yaml in current directory and parent directories
            config_path = self._find_config_file()

        self.config_path = config_path
        self.config = self._load_config()

    def _find_config_file(self) -> str:
        """
        Find config.yaml file in current or parent directories.

        Returns:
            Path to config.yaml file

        Raises:
            FileNotFoundError: If config.yaml is not found
        """
        # Start from current directory
        current_dir = Path.cwd()

        # Check current directory and up to 3 parent directories
        for _ in range(4):
            config_file = current_dir / "config.yaml"
            if config_file.exists():
                return str(config_file)
            current_dir = current_dir.parent

        # If not found, check the script's directory
        script_dir = Path(__file__).parent
        config_file = script_dir / "config.yaml"
        if config_file.exists():
            return str(config_file)

        raise FileNotFoundError(
            "config.yaml not found. Please create a config.yaml file or "
            "copy config.example.py to config.yaml"
        )

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config if config is not None else {}
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "llm.openai.api_key")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> api_key = config.get("llm.openai.api_key")
            >>> eps = config.get("phase1.dbscan.eps", 0.3)
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    # ===========================
    # Convenience properties for common config values
    # ===========================

    @property
    def encoder_model(self) -> str:
        """Get encoder model name."""
        return self.get("model.encoder_model", "all-MiniLM-L6-v2")

    @property
    def dbscan_eps(self) -> float:
        """Get DBSCAN epsilon parameter."""
        return self.get("phase1.dbscan.eps", 0.3)

    @property
    def dbscan_min_samples(self) -> int:
        """Get DBSCAN min_samples parameter."""
        return self.get("phase1.dbscan.min_samples", 2)

    @property
    def paragraph_threshold(self) -> float:
        """Get paragraph template threshold."""
        return self.get("phase1.paragraph_threshold", 0.1)

    @property
    def style_intensity(self) -> float:
        """Get style transfer intensity."""
        return self.get("phase2.style_intensity", 0.8)

    @property
    def max_chunk_length(self) -> int:
        """Get maximum chunk length."""
        return self.get("phase2.max_chunk_length", 2000)

    @property
    def use_bleurt(self) -> bool:
        """Get BLEURT usage flag."""
        return self.get("evaluation.use_bleurt", False)

    @property
    def keyword_top_k(self) -> int:
        """Get number of keywords to extract."""
        return self.get("evaluation.keyword_top_k", 20)

    @property
    def bleurt_weight(self) -> float:
        """Get BLEURT weight for content preservation."""
        return self.get("evaluation.content_weights.bleurt", 0.6)

    @property
    def keyword_weight(self) -> float:
        """Get keyword retention weight."""
        return self.get("evaluation.content_weights.keyword", 0.4)

    @property
    def length_variation_weight(self) -> float:
        """Get length variation weight for quality evaluation."""
        return self.get("evaluation.quality_weights.length_variation", 0.3)

    @property
    def vocabulary_diversity_weight(self) -> float:
        """Get vocabulary diversity weight."""
        return self.get("evaluation.quality_weights.vocabulary_diversity", 0.4)

    @property
    def readability_weight(self) -> float:
        """Get readability weight."""
        return self.get("evaluation.quality_weights.readability", 0.3)

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.get("llm.openai.api_key")

    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL."""
        return self.get("llm.openai.base_url")

    @property
    def openai_model(self) -> str:
        """Get OpenAI model name."""
        return self.get("llm.openai.model", "gpt-4o")

    @property
    def openai_temperature(self) -> float:
        """Get OpenAI temperature."""
        return self.get("llm.openai.temperature", 0.7)

    @property
    def openai_max_tokens(self) -> Optional[int]:
        """Get OpenAI max tokens. Returns None if not set (uses API default)."""
        value = self.get("llm.openai.max_tokens")
        # Return None if value is None, otherwise return as int
        return None if value is None else int(value)

    @property
    def num_workers(self) -> int:
        """Get number of workers for parallel processing."""
        return self.get("advanced.num_workers", 4)

    @property
    def use_parallel_processing(self) -> bool:
        """Get parallel processing flag."""
        return self.get("advanced.use_parallel_processing", False)

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.get("logging.level", "INFO")

    @property
    def log_file(self) -> str:
        """Get log file path."""
        return self.get("logging.file", "zerostylus.log")

    @property
    def reference_texts_dir(self) -> str:
        """Get reference texts directory."""
        return self.get("paths.reference_texts_dir", "data/reference_texts/")

    @property
    def source_texts_dir(self) -> str:
        """Get source texts directory."""
        return self.get("paths.source_texts_dir", "data/style_removed/")

    @property
    def output_dir(self) -> str:
        """Get output directory."""
        return self.get("paths.output_dir", "outputs/")

    @property
    def batch_output_dir(self) -> str:
        """Get batch output directory."""
        return self.get("paths.batch_output_dir", "outputs/batch_transformed")

    @property
    def evaluation_dir(self) -> str:
        """Get evaluation results directory."""
        return self.get("paths.evaluation_dir", "evaluation_results/")

    def __repr__(self) -> str:
        """String representation of Config."""
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None, reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Optional path to config file
        reload: Force reload configuration

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None or reload:
        _global_config = Config(config_path)

    return _global_config


def reset_config():
    """Reset global configuration instance."""
    global _global_config
    _global_config = None


if __name__ == "__main__":
    # Test configuration loader
    try:
        config = get_config()
        print(f"Configuration loaded from: {config.config_path}")
        print(f"\nSample configuration values:")
        print(f"  Encoder model: {config.encoder_model}")
        print(f"  DBSCAN eps: {config.dbscan_eps}")
        print(f"  Style intensity: {config.style_intensity}")
        print(f"  Max chunk length: {config.max_chunk_length}")
        print(f"  Use BLEURT: {config.use_bleurt}")
        print(f"  OpenAI model: {config.openai_model}")
        print(f"  Log level: {config.log_level}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

"""
DT-NoRerank Ablation Configuration
消融实验：禁用风格感知重排序
验证风格编码器和重排序机制的贡献
"""
from .base_config import AblationBaseConfig


class Config(AblationBaseConfig):
    """
    消融实验：禁用风格重排序
    - 原始：ENABLE_STYLE_RERANKING = True（使用风格编码器进行候选重排序）
    - 消融：ENABLE_STYLE_RERANKING = False（仅基于内容相似度检索）

    这验证了DeepTransfer新增的风格感知重排序机制的贡献
    """
    ABLATION_NAME = "dt_no_rerank"

    # 禁用风格重排序
    ENABLE_STYLE_RERANKING = False

    # 保持其他配置不变
    CONTENT_ENCODER_MODEL = "allenai/specter2_base"
    STYLE_ENCODER_MODEL = "AnnaWegmann/Style-Embedding"
    CHUNK_SIZE = 3

    # 可以复用DeepTransfer的索引（编码器相同）
    # 但为了隔离，仍使用独立索引
    INDEX_FILE = "ablation_dt_no_rerank_index.pkl"

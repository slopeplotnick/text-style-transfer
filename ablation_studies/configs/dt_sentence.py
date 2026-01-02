"""
DT-Sentence Ablation Configuration
消融实验：使用句子粒度（chunk_size=1）替代chunk粒度（chunk_size=3）
验证chunk粒度相比句子粒度的优势
"""
from .base_config import AblationBaseConfig


class Config(AblationBaseConfig):
    """
    消融实验：句子粒度
    - 原始：CHUNK_SIZE = 3（每个chunk包含3个句子）
    - 消融：CHUNK_SIZE = 1（每个chunk只有1个句子，即句子粒度）

    这模拟了ZeroStylus使用的句子级别处理方式
    """
    ABLATION_NAME = "dt_sentence"

    # 句子粒度（ZeroStylus风格）
    CHUNK_SIZE = 1

    # 保持其他配置不变
    CONTENT_ENCODER_MODEL = "allenai/specter2_base"
    STYLE_ENCODER_MODEL = "AnnaWegmann/Style-Embedding"

    # 索引文件（需要重新构建，因为分块方式不同）
    INDEX_FILE = "ablation_dt_sentence_index.pkl"

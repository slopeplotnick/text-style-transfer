"""
DT-NoSPECTER Ablation Configuration
消融实验：使用all-mpnet-base-v2替代SPECTER2作为内容编码器
验证SPECTER2对学术文本编码的优化效果
"""
from .base_config import AblationBaseConfig


class Config(AblationBaseConfig):
    """
    消融实验：替换内容编码器
    - 原始：allenai/specter2_base（针对学术文本优化）
    - 消融：sentence-transformers/all-mpnet-base-v2（通用文本编码器，ZeroStylus使用的模型）
    """
    ABLATION_NAME = "dt_no_specter"

    # 替换为ZeroStylus使用的通用编码器
    CONTENT_ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"

    # 风格编码器保持不变
    STYLE_ENCODER_MODEL = "AnnaWegmann/Style-Embedding"

    # 索引文件（需要重新构建，因为编码器不同）
    INDEX_FILE = "ablation_dt_no_specter_index.pkl"

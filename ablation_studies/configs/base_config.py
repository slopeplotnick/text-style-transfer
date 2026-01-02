"""
Base Configuration for Ablation Studies
继承DeepTransfer的配置，添加消融相关功能
"""
import os
import sys

# 添加DeepTransfer路径
sys.path.insert(0, '/root/rzy/tst/DeepTransfer')

from config import Config as DeepTransferConfig


class AblationBaseConfig(DeepTransferConfig):
    """
    消融实验基础配置类
    继承自DeepTransfer的Config，提供通用的消融实验配置
    """
    # 消融实验名称（子类覆盖）
    ABLATION_NAME = "base"

    # 索引文件目录
    INDEX_DIR = "/root/rzy/tst/ablation_studies/indices"

    # 输出目录基础路径
    OUTPUT_BASE_DIR = "/root/rzy/tst/output"

    @classmethod
    def get_index_path(cls):
        """获取消融实验的索引文件路径"""
        return os.path.join(cls.INDEX_DIR, f"{cls.ABLATION_NAME}_index.pkl")

    @classmethod
    def get_output_dir(cls):
        """获取消融实验的输出目录"""
        return os.path.join(cls.OUTPUT_BASE_DIR, f"ablation_{cls.ABLATION_NAME}")

    @classmethod
    def ensure_dirs(cls):
        """确保所需目录存在"""
        os.makedirs(cls.INDEX_DIR, exist_ok=True)
        os.makedirs(cls.get_output_dir(), exist_ok=True)

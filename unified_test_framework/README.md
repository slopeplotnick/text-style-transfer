# Unified Style Transfer Testing Framework

一套完整的测试框架，用于评估 CAT-LLM、ZeroStylus 和 DeepTransfer 三种文本风格转换方法。

## 📋 功能特性

### 支持的评估指标

#### CAT-LLM 指标
- **BLEU (1-4)**: 内容保留和风格相似度
- **BERTScore**: 基于BERT的语义相似度评估
- **Style Transfer Accuracy**: 使用fastText分类器评估风格转换准确率
- **Perplexity**: 使用GPT2-large评估流畅度
- **Overall Score**: 综合评分

#### ZeroStylus 指标
- **Style Consistency**: 风格一致性(基于embedding相似度)
- **Content Preservation**: 内容保留(BLEURT + 关键词保留)
- **Expression Quality**: 表达质量(启发式指标)
- **Average Score**: 平均评分

#### DeepTransfer 指标 (新增)
- **Content Preservation**: 使用SPECTER2内容编码器评估
- **Style Similarity**: 使用Style-Embedding编码器评估
- **Content-Style Disentanglement**: 内容-风格解耦质量
- **Dual-Space Alignment**: 双空间对齐评分
- **Embedding Diversity**: 生成多样性

## 🗂️ 文件结构

```
unified_test_framework/
├── config.py                   # 配置文件
├── data_loader.py              # 数据加载模块
├── deeptransfer_metrics.py     # DeepTransfer特有指标
├── extended_evaluator.py       # 扩展评估器(整合所有指标)
├── report_generator.py         # 报告生成器
├── test_runner.py              # 主测试脚本
├── quick_test.py               # 快速测试脚本
└── README.md                   # 本文件
```

## 📊 数据格式

### 输入数据

需要准备三种类型的文本数据：

1. **reference**: 原始学术论文文本(带有学术风格)
2. **source**: 去风格化后的文本(内容保留但风格中性)
3. **transferred**: 模型输出的风格转换文本

### 数据集目录结构

```
/root/datasets/arxiv_style_transfer/
├── reference/           # Reference文本
│   ├── ref_0000.txt
│   ├── ref_0001.txt
│   └── ...
├── source/              # Source文本
│   ├── src_0000.txt
│   ├── src_0001.txt
│   └── ...
└── metadata.json        # 元数据(包含文件对应关系)
```

### 模型输出目录

每个模型的transferred文本应放在独立目录中，例如：
```
/path/to/catllm/outputs/
├── ref_0000.txt  (或其他命名方式)
├── ref_0001.txt
└── ...
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers sentence-transformers
pip install bert-score fasttext nltk
pip install scikit-learn scipy numpy
```

可选依赖：
```bash
# BLEURT (推荐但非必需)
pip install bleurt
```

### 2. 配置路径

编辑 `config.py` 文件，设置数据集和模型输出路径：

```python
# Dataset paths
DATASET_BASE_DIR = "/root/datasets/arxiv_style_transfer"
REFERENCE_DIR = os.path.join(DATASET_BASE_DIR, "reference")
SOURCE_DIR = os.path.join(DATASET_BASE_DIR, "source")

# Model output paths
CATLLM_OUTPUT_DIR = "/path/to/catllm/outputs"
ZEROSTYLUS_OUTPUT_DIR = "/path/to/zerostylus/outputs"
DEEPTRANSFER_OUTPUT_DIR = "/path/to/deeptransfer/outputs"
```

### 3. 快速测试 (2个样本)

```bash
cd /root/rzy/tst
python unified_test_framework/quick_test.py
```

这将：
- 使用2个样本进行快速测试
- 自动配置API (使用指定的OpenAI API)
- 交互式选择要测试的模型目录

### 4. 完整测试

#### 测试单个模型

```bash
python unified_test_framework/test_runner.py \
    --model catllm \
    --catllm-dir /path/to/catllm/outputs \
    --max-samples 10
```

#### 测试所有模型

```bash
python unified_test_framework/test_runner.py \
    --model all \
    --catllm-dir /path/to/catllm/outputs \
    --zerostylus-dir /path/to/zerostylus/outputs \
    --deeptransfer-dir /path/to/deeptransfer/outputs \
    --max-samples 100
```

#### 仅使用特定指标

```bash
# 只使用CAT-LLM指标
python unified_test_framework/test_runner.py \
    --model catllm \
    --catllm-dir /path/to/catllm/outputs \
    --skip-zerostylus \
    --skip-deeptransfer

# 跳过DeepTransfer的双编码器指标(节省时间)
python unified_test_framework/test_runner.py \
    --model all \
    --no-dt-metrics
```

## 📈 输出报告

测试完成后，会在 `results/` 目录生成以下报告：

1. **JSON格式**: `evaluation_results.json`
   - 完整的评估结果，包含所有指标
   - 适合程序化处理

2. **Markdown格式**: `evaluation_report.md`
   - 人类可读的对比报告
   - 包含所有模型的详细指标和对比表格

3. **CSV格式**: `evaluation_results.csv`
   - 表格形式的评估结果
   - 适合Excel等工具打开

4. **单模型报告**: `<model_name>_results.json`
   - 每个模型的独立报告

## 🔧 高级用法

### 自定义配置

在 `config.py` 中可以调整：

- `MAX_SAMPLES`: 最大样本数
- `ENCODING_BATCH_SIZE`: 编码批次大小
- `USE_GPU`: 是否使用GPU
- `CONTENT_WEIGHT`, `STYLE_WEIGHT`: DeepTransfer对齐评分的权重
- `ENABLE_CACHE`: 是否启用缓存

### 使用预训练的分类器

```python
# 在代码中
evaluator.evaluate_all_extended(
    source_texts=sources,
    reference_texts=references,
    transferred_texts=transferred,
    train_classifier=False,
    classifier_model_path="/path/to/saved_classifier.bin"
)
```

### 自定义文件命名模式

如果你的transferred文件使用特殊命名方式：

```python
loader.load_transferred_texts(
    samples,
    transferred_dir="/path/to/outputs",
    naming_pattern="paper_id"  # 或 "sequential", "src_XXXX", "ref_XXXX"
)
```

## 📝 API配置

框架使用OpenAI API进行某些评估(如perplexity计算)。

在 `quick_test.py` 中已配置：
- Base URL: `https://newapi.deepwisdom.ai/v1`
- Model: `gpt-4o`
- API Key: (已配置)

如需修改，编辑 `quick_test.py` 中的 `setup_api_config()` 函数。

## 🎯 评估指标说明

### CAT-LLM Overall Score
几何平均: (BLEU_content × BLEU_style × Style_Acc × Fluency)^(1/4)

### ZeroStylus Average Score
算术平均: (Style_Consistency + Content_Preservation + Expression_Quality) / 3

### DeepTransfer Disentanglement Score
(Content_Preservation + Style_Transfer) / 2

### DeepTransfer Alignment Score
α × Content_Similarity + β × Style_Similarity (默认α=β=0.5)

## 🐛 故障排除

### 问题：NLTK资源下载失败
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 问题：GPU内存不足
在 `config.py` 中：
```python
USE_GPU = False
# 或减小批次大小
ENCODING_BATCH_SIZE = 8
```

### 问题：某些指标计算失败
大多数指标都是独立的，单个指标失败不会影响其他指标。检查错误信息并根据需要安装缺失的依赖。

## 📚 引用

如果使用了本测试框架，请引用相关论文：

- **CAT-LLM**: [论文信息]
- **ZeroStylus**: [论文信息]
- **DeepTransfer**: [论文信息]

## 📧 联系方式

如有问题或建议，请联系开发者或提交Issue。

---

**最后更新**: 2024-12-04

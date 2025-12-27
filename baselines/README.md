# Baseline Methods for Academic Style Transfer

本目录包含两个基线方法：Zero-Shot和Few-Shot，用于学术文本风格转换任务的基准对比。

## 方法说明

### 1. Zero-Shot Baseline (零样本基线)
- **文件**: `zero_shot.py`
- **方法**: 使用简单的提示词，不提供任何示例，直接让LLM进行风格转换
- **特点**: 最简单的基线方法，完全依赖LLM的预训练能力

### 2. Few-Shot Baseline (少样本基线)
- **文件**: `few_shot.py`
- **方法**: 在提示词中提供3个示例对（源文本→目标风格文本），让LLM学习目标风格
- **示例来源**: 从数据集的500-600范围随机选择（避免与测试集0-499重叠）
- **特点**: 通过示例演示目标风格，性能优于Zero-Shot

## API配置
两个基线方法与其他方法（CAT-LLM、ZeroStylus、DeepTransfer）使用相同的API配置：
- **API URL**: `https://newapi.deepwisdom.ai/v1/chat/completions`
- **Model**: `gpt-4o`
- **Temperature**: 0.7
- **Max Concurrent**: 10

## 使用方法

### 一键运行所有流程
```bash
bash run_baselines_500.sh
```

该脚本会自动完成：
1. 运行Zero-Shot基线（500个样本）
2. 运行Few-Shot基线（500个样本）
3. 评估两个基线的性能
4. 生成统一的评估报告（包含所有方法）

### 单独运行各个组件

#### 1. 运行Zero-Shot
```bash
python baselines/zero_shot.py \
    --source-dir /root/datasets/arxiv_style_transfer/source \
    --output-dir output/zero_shot \
    --max-samples 500 \
    --max-concurrent 10
```

#### 2. 运行Few-Shot
```bash
python baselines/few_shot.py \
    --source-dir /root/datasets/arxiv_style_transfer/source \
    --reference-dir /root/datasets/arxiv_style_transfer/reference \
    --output-dir output/few_shot \
    --max-samples 500 \
    --max-concurrent 10 \
    --num-examples 3
```

#### 3. 评估基线
```bash
# 评估Zero-Shot
python baselines/evaluate_baseline.py \
    --model-name "Zero-Shot" \
    --transferred-dir output/zero_shot \
    --max-samples 500

# 评估Few-Shot
python baselines/evaluate_baseline.py \
    --model-name "Few-Shot" \
    --transferred-dir output/few_shot \
    --max-samples 500
```

#### 4. 生成统一报告
```bash
python baselines/generate_unified_report.py
```

## 输出结果

### 转换结果
- **Zero-Shot**: `output/zero_shot/transferred_XXXX.txt`
- **Few-Shot**: `output/few_shot/transferred_XXXX.txt`

### 评估结果
所有结果保存在 `unified_test_framework/results/` 目录：
- `zero_shot_results.json` - Zero-Shot评估结果（JSON）
- `zero_shot_samples_summary.json` - Zero-Shot样本摘要
- `few_shot_results.json` - Few-Shot评估结果（JSON）
- `few_shot_samples_summary.json` - Few-Shot样本摘要
- `evaluation_report.md` - **统一评估报告**（包含所有5个方法）
- `evaluation_results.json` - 统一评估结果（JSON）
- `evaluation_results.csv` - 统一评估结果（CSV）

## 评估指标
基线方法使用与其他方法相同的评估框架：

### CAT-LLM指标
- BLEU (1-4)
- BERTScore (Precision, Recall, F1)
- Style Transfer Accuracy
- Perplexity

### ZeroStylus指标
- Style Consistency
- Content Preservation
- Expression Quality

### DeepTransfer指标
- Content Preservation (Specter2)
- Style Similarity (Style-Embedding)
- Disentanglement Score
- Diversity

## 数据泄漏防护
- **测试集**: 使用0-499样本
- **Few-Shot示例**: 从500-600范围随机选择，确保不与测试集重叠
- **评估**: 所有方法使用相同的测试集，保证公平对比

## 文件结构
```
baselines/
├── zero_shot.py              # Zero-Shot实现
├── few_shot.py               # Few-Shot实现
├── evaluate_baseline.py      # 评估脚本
├── generate_unified_report.py # 统一报告生成脚本
└── README.md                 # 本文档

run_baselines_500.sh          # 一键运行脚本
```

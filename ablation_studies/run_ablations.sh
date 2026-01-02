#!/bin/bash
# =============================================================================
# DeepTransfer Ablation Studies - Full Pipeline
# 完整消融实验流程：运行所有消融 -> 评估 -> 生成统一报告
#
# 用法:
#   ./run_ablations.sh                    # 运行所有消融，默认500样本
#   ./run_ablations.sh 100                # 运行所有消融，100样本
#   ./run_ablations.sh 500 dt_sentence    # 从dt_sentence开始，500样本
# =============================================================================

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# 配置代理（用于下载HuggingFace模型）
# =============================================================================
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

echo "Proxy configured: http://127.0.0.1:7890"

# 配置
MAX_SAMPLES=${1:-500}
START_FROM=${2:-""}
SOURCE_DIR="/root/datasets/arxiv_style_transfer/source"
DATA_DIR="/root/datasets/arxiv_style_transfer/reference"

echo "======================================================================"
echo "DeepTransfer Ablation Studies"
echo "======================================================================"
echo "Script directory: $SCRIPT_DIR"
echo "Max samples: $MAX_SAMPLES"
echo "Start from: ${START_FROM:-'(all ablations)'}"
echo "Source directory: $SOURCE_DIR"
echo "Data directory: $DATA_DIR"
echo "======================================================================"

# # 阶段1：运行所有消融实验
# echo ""
# echo "======================================================================"
# echo "STAGE 1: Running all ablation experiments"
# echo "======================================================================"
#
# if [ -n "$START_FROM" ]; then
#     python run_all_ablations.py \
#         --source-dir "$SOURCE_DIR" \
#         --data-dir "$DATA_DIR" \
#         --max-samples "$MAX_SAMPLES" \
#         --start-from "$START_FROM"
# else
#     python run_all_ablations.py \
#         --source-dir "$SOURCE_DIR" \
#         --data-dir "$DATA_DIR" \
#         --max-samples "$MAX_SAMPLES"
# fi
#
# # 阶段2：评估所有消融实验
# echo ""
# echo "======================================================================"
# echo "STAGE 2: Evaluating all ablation experiments"
# echo "======================================================================"
#
# python evaluate_ablations.py

# 阶段3：生成统一报告
echo ""
echo "======================================================================"
echo "STAGE 3: Generating unified report (including ablations)"
echo "======================================================================"

cd /root/rzy/tst/baselines
python generate_unified_report.py

echo ""
echo "======================================================================"
echo "ABLATION STUDIES COMPLETED"
echo "======================================================================"
echo ""
echo "Results:"
echo "  - Ablation outputs: /root/rzy/tst/output/ablation_*"
echo "  - Evaluation results: /root/rzy/tst/unified_test_framework/results/"
echo "  - Unified report: /root/rzy/tst/unified_test_framework/results/evaluation_report.md"
echo ""
echo "======================================================================"

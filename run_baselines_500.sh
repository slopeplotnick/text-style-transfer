#!/bin/bash
set -e

echo "=========================================="
echo "Running Baseline Methods - 500 Samples"
echo "Zero-Shot and Few-Shot Baselines"
echo "=========================================="

# 创建输出目录
mkdir -p output/zero_shot
mkdir -p output/few_shot

# API 配置（与其他方法保持一致）
API_URL="https://newapi.deepwisdom.ai/v1/chat/completions"
API_KEY="${OPENAI_API_KEY:-YOUR_API_KEY_HERE}"
MODEL="gpt-4o"
MAX_SAMPLES=500
MAX_CONCURRENT=10

#################################
# 1. Zero-Shot Baseline
#################################
echo -e "\n[1/2] Running Zero-Shot Baseline..."
python baselines/zero_shot.py \
    --source-dir /root/datasets/arxiv_style_transfer/source \
    --output-dir output/zero_shot \
    --api-url "$API_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --max-samples $MAX_SAMPLES \
    --max-concurrent $MAX_CONCURRENT

echo "  ✓ Zero-Shot completed: output/zero_shot/"

#################################
# 2. Few-Shot Baseline
#################################
echo -e "\n[2/2] Running Few-Shot Baseline..."
python baselines/few_shot.py \
    --source-dir /root/datasets/arxiv_style_transfer/source \
    --reference-dir /root/datasets/arxiv_style_transfer/reference \
    --output-dir output/few_shot \
    --api-url "$API_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --max-samples $MAX_SAMPLES \
    --max-concurrent $MAX_CONCURRENT \
    --num-examples 3

echo "  ✓ Few-Shot completed: output/few_shot/"

#################################
# 3. Evaluation
#################################
echo -e "\n[3/4] Running Evaluation for Baselines..."

# 评估 Zero-Shot
echo -e "\nEvaluating Zero-Shot..."
python baselines/evaluate_baseline.py \
    --model-name "Zero-Shot" \
    --transferred-dir output/zero_shot \
    --max-samples $MAX_SAMPLES

# 评估 Few-Shot
echo -e "\nEvaluating Few-Shot..."
python baselines/evaluate_baseline.py \
    --model-name "Few-Shot" \
    --transferred-dir output/few_shot \
    --max-samples $MAX_SAMPLES

#################################
# 4. Generate Unified Report
#################################
echo -e "\n[4/4] Generating Unified Evaluation Report..."
python baselines/generate_unified_report.py

echo -e "\n=========================================="
echo "Baseline Evaluation Completed!"
echo "=========================================="
echo "Results:"
echo "  Zero-Shot output:    /root/rzy/tst/output/zero_shot/"
echo "  Few-Shot output:     /root/rzy/tst/output/few_shot/"
echo "  Evaluation results:  /root/rzy/tst/unified_test_framework/results/"
echo "  Unified report:      /root/rzy/tst/unified_test_framework/results/evaluation_report.md"
echo ""

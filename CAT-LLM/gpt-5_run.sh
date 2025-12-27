#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 模型名称
MODEL_NAME="gpt-5-2025-08-07"

# 构建输出目录
OUTPUT_DIR="./output-${MODEL_NAME}-${TIMESTAMP}"

echo "======================================"
echo "Starting pipeline with GPT-5"
echo "Model: ${MODEL_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"
echo ""

python main_pipeline.py \
    --data-dir /root/datasets/text_style_transfer/ipm_abstracts \
    --output-dir "${OUTPUT_DIR}" \
    --api-url https://yunwu.zeabur.app/v1/chat/completions \
    --api-key ${OPENAI_API_KEY:-YOUR_API_KEY_HERE} \
    --model "${MODEL_NAME}"

echo ""
echo "======================================"
echo "Pipeline completed!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved to: ${OUTPUT_DIR}"
echo "======================================"

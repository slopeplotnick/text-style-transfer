#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 模型名称
MODEL_NAME="gemini-2.5-flash"

# 构建输出目录
OUTPUT_DIR="./output-async-${MODEL_NAME}-${TIMESTAMP}"

# 最大并发数
MAX_CONCURRENT=10

echo "======================================"
echo "Starting ASYNC pipeline with Gemini"
echo "Model: ${MODEL_NAME}"
echo "Max concurrent requests: ${MAX_CONCURRENT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"
echo ""

python main_pipeline_async.py \
    --data-dir /root/datasets/text_style_transfer/ipm_abstracts \
    --output-dir "${OUTPUT_DIR}" \
    --api-url https://yunwu.zeabur.app/v1/chat/completions \
    --api-key ${OPENAI_API_KEY:-YOUR_API_KEY_HERE} \
    --model "${MODEL_NAME}" \
    --max-concurrent ${MAX_CONCURRENT}

echo ""
echo "======================================"
echo "ASYNC Pipeline completed!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved to: ${OUTPUT_DIR}"
echo "======================================"

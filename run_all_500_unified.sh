#!/bin/bash
set -e

echo "=========================================="
echo "Running All Three Methods - 500 Samples"
echo "Unified Output Directory: /root/rzy/tst/output/"
echo "=========================================="

# 创建统一输出目录
mkdir -p output/catllm
mkdir -p output/zerostylus
mkdir -p output/deeptransfer

#################################
# 1. CAT-LLM
#################################
echo -e "\n[1/3] Running CAT-LLM (Async)..."
cd CAT-LLM

# 准备目录
mkdir -p temp_catllm/style_removed reference_for_style

# 复制文件
echo "  Preparing files..."
cp /root/datasets/arxiv_style_transfer/source/src_*.txt temp_catllm/style_removed/
cp /root/datasets/arxiv_style_transfer/reference/ref_*.txt reference_for_style/

# 运行
echo "  Running pipeline..."
python main_pipeline_async.py \
    --data-dir reference_for_style \
    --output-dir temp_catllm \
    --api-url "https://newapi.deepwisdom.ai/v1/chat/completions" \
    --api-key "${OPENAI_API_KEY:-YOUR_API_KEY_HERE}" \
    --model "gpt-4o" \
    --max-concurrent 10 \
    --skip-style-removal \
    --skip-evaluation

# 重命名输出文件
echo "  Renaming output files..."
cd temp_catllm/transferred
for file in src_*_transferred.txt; do
    # 从 src_0000_transferred.txt 提取 0000
    num=$(echo $file | sed 's/src_\([0-9]*\)_transferred.txt/\1/')
    # 确保是4位数字格式（10# 强制十进制解释，避免前导零被当作八进制）
    num_padded=$(printf "%04d" $((10#$num)))
    cp "$file" "../../../output/catllm/transferred_${num_padded}.txt"
done
cd ../..

echo "  ✓ CAT-LLM completed: output/catllm/"
cd ..

#################################
# 2. ZeroStylus
#################################
echo -e "\n[2/3] Running ZeroStylus..."
cd ZeroStylus

# 创建临时目录
mkdir -p temp_zerostylus

# 运行
echo "  Running pipeline..."
python batch_process.py \
    --reference_dir /root/datasets/arxiv_style_transfer/reference \
    --source_dir /root/datasets/arxiv_style_transfer/source \
    --output_dir temp_zerostylus \
    --max_samples 500

# 重命名输出文件
echo "  Renaming output files..."
cd temp_zerostylus
for file in src_*_transformed.txt; do
    # 从 src_0000_transformed.txt 提取 0000
    num=$(echo $file | sed 's/src_\([0-9]*\)_transformed.txt/\1/')
    # 确保是4位数字格式（10# 强制十进制解释，避免前导零被当作八进制）
    num_padded=$(printf "%04d" $((10#$num)))
    cp "$file" "../../output/zerostylus/transferred_${num_padded}.txt"
done
cd ..

echo "  ✓ ZeroStylus completed: output/zerostylus/"
cd ..

#################################
# 3. DeepTransfer
#################################
echo -e "\n[3/3] Running DeepTransfer..."
cd DeepTransfer

# 构建索引
if [ ! -f deep_transfer_index.pkl ]; then
    echo "  Building index..."
    python main.py build --data_dir /root/datasets/arxiv_style_transfer/reference
fi

# 创建临时目录
mkdir -p temp_deeptransfer

# 运行（使用异步版本）
echo "  Running async pipeline..."
python batch_transfer_async.py \
    --source_dir /root/datasets/arxiv_style_transfer/source \
    --output_dir temp_deeptransfer \
    --max_samples 500 \
    --max-concurrent 10 \
    --batch-size 20

# 重命名输出文件
echo "  Renaming output files..."
cd temp_deeptransfer
for file in ref_*.txt; do
    num=$(echo $file | sed 's/ref_\([0-9]*\).txt/\1/')
    # 10# 强制十进制解释，避免前导零被当作八进制
    num_padded=$(printf "%04d" $((10#$num)))
    cp "$file" "../../output/deeptransfer/transferred_${num_padded}.txt"
done
cd ..

echo "  ✓ DeepTransfer completed: output/deeptransfer/"
cd ..

#################################
# 4. Evaluation
#################################
echo -e "\n[4/4] Running Unified Evaluation..."
python -m unified_test_framework.test_runner \
    --model all \
    --catllm-dir output/catllm \
    --zerostylus-dir output/zerostylus \
    --deeptransfer-dir output/deeptransfer \
    --max-samples 500

echo -e "\n=========================================="
echo "All processes completed!"
echo "=========================================="
echo "Results:"
echo "  CAT-LLM:      /root/rzy/tst/output/catllm/"
echo "  ZeroStylus:   /root/rzy/tst/output/zerostylus/"
echo "  DeepTransfer: /root/rzy/tst/output/deeptransfer/"
echo "  Evaluation:   /root/rzy/tst/unified_test_framework/results/"

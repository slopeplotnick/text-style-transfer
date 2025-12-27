#!/bin/bash

# 重命名输出文件脚本
echo "Renaming output files..."
cd temp_deeptransfer

count=0
total=$(ls ref_*.txt 2>/dev/null | wc -l)
echo "Found $total files to process"

for file in ref_*.txt; do
    if [ -f "$file" ]; then
        num=$(echo $file | sed 's/ref_\([0-9]*\).txt/\1/')
        # 10# 强制十进制解释，避免前导零被当作八进制
        num_padded=$(printf "%04d" $((10#$num)))
        cp "$file" "../../output/deeptransfer/transferred_${num_padded}.txt"
        ((count++))
        # 每处理50个文件显示一次进度
        if [ $((count % 50)) -eq 0 ]; then
            echo "Progress: $count/$total files processed"
        fi
    fi
done

echo "✓ Completed: ${count} files transferred"

#!/bin/bash
# Script to run DeepTransfer with proxy for faster model downloads

# Set proxy for HuggingFace model downloads
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# Also set HuggingFace specific environment variables
export HF_ENDPOINT=https://hf-mirror.com  # Optional: use mirror if needed

echo "Proxy configured: http://127.0.0.1:7890"
echo "Running: python main.py $@"
echo ""

# Run the main script with all passed arguments
python main.py "$@"

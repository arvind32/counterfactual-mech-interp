#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_vllm.sh Qwen/Qwen3.5-27B on
#   GPU=1 MAX_NUM_SEQS=4 bash scripts/run_vllm.sh Qwen/Qwen3.5-9B off

MODEL="${1:-Qwen/Qwen3.5-27B}"
THINKING="${2:-on}"
GPU="${GPU:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
BATCH_SIZE="${BATCH_SIZE:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

python run_dataset.py \
  --model "$MODEL" \
  --thinking "$THINKING" \
  --batch-size "$BATCH_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --sampler-backend native \
  --gdn-prefill-backend triton

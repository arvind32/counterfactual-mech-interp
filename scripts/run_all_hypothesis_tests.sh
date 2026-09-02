#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=("$@")

run_test() {
  local model="$1"
  local thinking="$2"
  local max_num_seqs="$3"

  echo
  echo "Running ${model} with thinking=${thinking}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" prompts/run_hypothesis_tests.py \
    --model "${model}" \
    --thinking "${thinking}" \
    --max-num-seqs "${max_num_seqs}" \
    "${EXTRA_ARGS[@]}"
}

run_test "Qwen/Qwen3.5-9B" off 16
run_test "Qwen/Qwen3.5-9B" on 16
run_test "Qwen/Qwen3.5-27B" off 6
run_test "Qwen/Qwen3.5-27B" on 6
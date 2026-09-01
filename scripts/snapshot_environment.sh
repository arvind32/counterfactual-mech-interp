#!/usr/bin/env bash
set -euo pipefail

# Run this only after the inference environment successfully completes a smoke
# test. It records the exact working environment for reproducibility.

python -m pip freeze --all > requirements-lock.txt
conda env export --no-builds > environment-lock.yml

{
  python --version
  python -c 'import torch; print("torch:", torch.__version__); print("torch CUDA:", torch.version.cuda)'
  python -c 'import vllm; print("vllm:", vllm.__version__)'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > environment-report.txt

echo "Wrote requirements-lock.txt"
echo "Wrote environment-lock.yml"
echo "Wrote environment-report.txt"

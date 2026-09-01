# Inference workflow

This project uses one Python runner for both Qwen models and both thinking
conditions. Runtime choices are explicit command-line arguments and are saved
with every result record.

## Files

- `run_dataset.py`: experiment and vLLM batch-inference logic.
- `scripts/run_vllm.sh`: reproducible launch command and GPU selection.
- `scripts/snapshot_environment.sh`: captures the exact working software and
  hardware environment after a successful smoke test.
- `results/`: resumable JSONL output.

## Stable backend profile

The default launcher uses vLLM's native top-k/top-p sampler and the Triton GDN
prefill backend. This avoids runtime FlashInfer compilation. The chosen
backends are stored in each JSONL record under `generation_config`.

## Smoke test

Run one unfinished example before launching the full dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python run_dataset.py \
  --model Qwen/Qwen3.5-27B \
  --thinking on \
  --limit 1 \
  --max-num-seqs 1 \
  --sampler-backend native \
  --gdn-prefill-backend triton
```

The runner is resumable. If the smoke test writes one valid result, the full
run skips that example rather than regenerating it.

## Full batch

```bash
bash scripts/run_vllm.sh Qwen/Qwen3.5-27B on
```

All unfinished prompts are submitted together by default. vLLM schedules up
to `MAX_NUM_SEQS` concurrently. Reduce concurrency without changing the
submitted dataset when necessary:

```bash
MAX_NUM_SEQS=4 bash scripts/run_vllm.sh Qwen/Qwen3.5-27B on
```

## Capture the working environment

After the smoke test succeeds:

```bash
bash scripts/snapshot_environment.sh
```

Commit the resulting lock and report files with the code. Do not create the
lock files before the environment works, because that would preserve a broken
dependency combination.

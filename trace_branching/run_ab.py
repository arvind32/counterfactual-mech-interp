#!/usr/bin/env python3
"""Run the preregistered 60-continuation A/B trace-branching screen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Any

# Hugging Face reads these settings during import, so configure offline behavior
# before importing vLLM through the custom processor module.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
for _cache_variable in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
    _cache_value = os.environ.get(_cache_variable)
    if _cache_value and _cache_value.startswith("/~"):
        raise RuntimeError(
            f"{_cache_variable} has a malformed cache path: {_cache_value!r}. "
            "Use an absolute path such as /path/to/huggingface/cache."
        )
    if _cache_value and _cache_value.startswith("~"):
        os.environ[_cache_variable] = str(Path(_cache_value).expanduser())

from history_presence import (
    HistoryPresenceProcessor,
    PENALTY_KEY,
    PREFIX_IDS_KEY,
    PrefixPresenceRequestProcessor,
)


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MAX_MODEL_LEN = 40960
ORIGINAL_MAX_NEW_TOKENS = 32768
PRESENCE_PENALTY = 1.5
EXPECTED_VLLM_VERSION = "0.28.1rc1.dev177+g3a2ed6cba"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=ROOT / "ab_jobs.jsonl")
    parser.add_argument("--sources", type=Path, default=ROOT / "source_records.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "ab_results.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--cuts", nargs="*", choices=["A", "B"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--runtime-smoke-only",
        action="store_true",
        help="Load the real engine, test the processor, generate 32 tokens, then stop.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hash_ints(values: list[int]) -> str:
    payload = ",".join(str(value) for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_token_ids(value, *, label: str) -> list[int]:
    """Normalize tokenizer/processor outputs to one flat list of integer IDs."""
    original_type = type(value).__name__
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError(
                f"{label} returned {original_type} without an input_ids field"
            )
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in value
    ):
        preview = repr(value)[:200]
        raise TypeError(
            f"{label} did not produce a flat integer token list "
            f"(original type {original_type}; value preview {preview})"
        )
    return list(value)


def split_completion(raw_completion: str, budget_exhausted: bool) -> dict[str, Any]:
    cleaned = raw_completion
    for marker in ("<|im_end|>", "<|endoftext|>"):
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip()
    if "</think>" not in cleaned:
        return {
            "thinking": cleaned.replace("<think>", "", 1).strip(),
            "response": None,
            "thinking_status": (
                "budget_exhausted" if budget_exhausted else "incomplete_thinking_trace"
            ),
        }
    thinking, response = cleaned.split("</think>", 1)
    return {
        "thinking": thinking.replace("<think>", "", 1).strip(),
        "response": response.strip(),
        "thinking_status": "completed",
    }


def sampling_params(SamplingParams, *, seed: int, max_tokens: int, prefix_ids: list[int]):
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=PRESENCE_PENALTY,
        repetition_penalty=1.0,
        seed=seed,
        skip_special_tokens=False,
        extra_args={
            PREFIX_IDS_KEY: sorted(set(prefix_ids)),
            PENALTY_KEY: PRESENCE_PENALTY,
        },
    )


def build_engine(gpu_memory_utilization: float):
    """Load the exact model/backend configuration used by the source runs."""
    from vllm import LLM, SamplingParams

    installed_vllm = version("vllm")
    if installed_vllm != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            "vLLM version mismatch: expected "
            f"{EXPECTED_VLLM_VERSION}, found {installed_vllm}. "
            "Resolve the environment mismatch before generating."
        )
    llm = LLM(
        model=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=16,
        seed=0,
        language_model_only=True,
        additional_config={"gdn_prefill_backend": "triton"},
        logits_processors=[HistoryPresenceProcessor],
    )
    return llm, SamplingParams, installed_vllm


def prepare_jobs(jobs: list[dict], sources: list[dict], tokenizer) -> list[dict]:
    """Reconstruct and verify every exact branch prefix, then fix token budgets."""
    prepared: list[dict] = []
    for job in jobs:
        source = sources[job["source_index"]]["record"]
        if not source["raw_completion"].startswith(job["raw_prefix"]):
            raise RuntimeError(f"raw prefix mismatch for {job['job_id']}")
        actual_cut_line = job["raw_prefix"].splitlines()[-1]
        if actual_cut_line != job["cut_line_text"]:
            raise RuntimeError(f"cut line mismatch for {job['job_id']}")
        chat_ids = tokenizer.apply_chat_template(
            source["conversation"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        chat_ids = normalize_token_ids(chat_ids, label="chat template")
        prefix_ids = normalize_token_ids(
            tokenizer.encode(job["raw_prefix"], add_special_tokens=False),
            label="trace prefix",
        )
        decoded_prefix = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_prefix != job["raw_prefix"]:
            raise RuntimeError(
                f"tokenizer text round trip failed for {job['job_id']}; refusing replay"
            )
        prepared.append(
            {
                "job": job,
                "source": source,
                "chat_ids": chat_ids,
                "prefix_ids": prefix_ids,
                "prompt_ids": chat_ids + prefix_ids,
            }
        )

    family_max_prompt: dict[str, int] = {}
    for item in prepared:
        family = item["job"]["story_family"]
        family_max_prompt[family] = max(
            family_max_prompt.get(family, 0), len(item["prompt_ids"])
        )
    for item in prepared:
        family = item["job"]["story_family"]
        item["max_tokens"] = min(
            ORIGINAL_MAX_NEW_TOKENS,
            MAX_MODEL_LEN - family_max_prompt[family] - 1,
        )
        if item["max_tokens"] < 1:
            raise RuntimeError(f"no continuation budget for {item['job']['job_id']}")
    return prepared


def runtime_smoke(llm, tokenizer, SamplingParams) -> None:
    """Test processor arithmetic and request-level enable/disable behavior."""
    import torch

    logits = torch.tensor([4.0, 4.0, 4.0], device="cuda")
    processor = PrefixPresenceRequestProcessor([0, 1], PRESENCE_PENALTY)
    processor([], logits)
    assert torch.allclose(logits.cpu(), torch.tensor([2.5, 2.5, 4.0]))
    processor([1], logits := torch.tensor([4.0, 4.0, 4.0], device="cuda"))
    # This custom supplement leaves token 1 untouched because native presence
    # penalty is responsible for it after it appears in the new output.
    assert torch.allclose(logits.cpu(), torch.tensor([2.5, 4.0, 4.0]))

    base = dict(
        max_tokens=1,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=PRESENCE_PENALTY,
        repetition_penalty=1.0,
        seed=424242,
        skip_special_tokens=False,
    )
    params_empty = SamplingParams(
        **base,
        extra_args={PREFIX_IDS_KEY: [], PENALTY_KEY: PRESENCE_PENALTY},
    )
    adapter = object.__new__(HistoryPresenceProcessor)
    disabled = adapter.new_req_logits_processor(params_empty)
    if disabled is not None:
        raise RuntimeError("empty history did not disable the custom processor")
    params_active = SamplingParams(
        **base,
        extra_args={PREFIX_IDS_KEY: [0, 1], PENALTY_KEY: PRESENCE_PENALTY},
    )
    active = adapter.new_req_logits_processor(params_active)
    if not isinstance(active, PrefixPresenceRequestProcessor):
        raise RuntimeError("nonempty history did not enable the custom processor")
    print("Runtime smoke test passed (GPU arithmetic and processor dispatch).")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    all_jobs = load_jsonl(args.jobs)
    jobs = list(all_jobs)
    sources = load_jsonl(args.sources)
    if args.cases:
        jobs = [job for job in jobs if job["case_name"] in set(args.cases)]
    if args.cuts:
        jobs = [job for job in jobs if job["cut"] in set(args.cuts)]
    completed = {
        row["job_id"] for row in load_jsonl(args.output)
    } if args.output.exists() else set()
    jobs = [job for job in jobs if job["job_id"] not in completed]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    llm, SamplingParams, _ = build_engine(args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    runtime_smoke(llm, tokenizer, SamplingParams)
    if args.runtime_smoke_only:
        return 0
    if not jobs:
        print("No unfinished jobs selected.")
        return 0

    # Prepare the full preregistered set so partial/resumed runs keep identical
    # token budgets, then select only the requested unfinished jobs.
    prepared_all = prepare_jobs(all_jobs, sources, tokenizer)

    selected_ids = {job["job_id"] for job in jobs}
    prepared = [
        item for item in prepared_all if item["job"]["job_id"] in selected_ids
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start : start + args.batch_size]
            prompts = [{"prompt_token_ids": item["prompt_ids"]} for item in batch]
            params = [
                sampling_params(
                    SamplingParams,
                    seed=item["job"]["sample_seed"],
                    max_tokens=item["max_tokens"],
                    prefix_ids=item["prefix_ids"],
                )
                for item in batch
            ]
            print(f"Submitting jobs {start + 1}-{start + len(batch)} of {len(prepared)}")
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of outputs")
            for item, output in zip(batch, outputs):
                completion = output.outputs[0]
                continuation_ids = list(completion.token_ids)
                continuation_text = tokenizer.decode(
                    continuation_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                full_completion = item["job"]["raw_prefix"] + continuation_text
                budget_exhausted = (
                    completion.finish_reason == "length"
                    or len(continuation_ids) >= item["max_tokens"]
                )
                parsed = split_completion(full_completion, budget_exhausted)
                record = {
                    **item["job"],
                    "model": MODEL,
                    "revision": REVISION,
                    "sampling": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": PRESENCE_PENALTY,
                        "repetition_penalty": 1.0,
                        "max_tokens": item["max_tokens"],
                    },
                    "chat_token_count": len(item["chat_ids"]),
                    "prefix_token_count": len(item["prefix_ids"]),
                    "prefix_token_hash": hash_ints(item["prefix_ids"]),
                    "prefix_text_roundtrip_verified": True,
                    "original_prefix_token_ids_were_saved": False,
                    "continuation_token_count": len(continuation_ids),
                    "continuation_token_hash": hash_ints(continuation_ids),
                    "continuation_text": continuation_text,
                    "full_completion": full_completion,
                    "finish_reason": completion.finish_reason,
                    "budget_exhausted": budget_exhausted,
                    **parsed,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"Completed {len(prepared)} jobs; results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

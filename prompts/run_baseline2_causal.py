#!/usr/bin/env python3
"""Run causal-prompt counterfactual edits with vLLM offline batching."""

import argparse
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REVISIONS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the baseline_2 causal-prompt condition with vLLM continuous "
            "batching and write resumable JSONL results."
        )
    )
    parser.add_argument("--model", required=True, choices=list(REVISIONS))
    parser.add_argument(
        "--input",
        default="data/counterfactual_causal_propagation_examples_19.xlsm",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Results JSONL path. Defaults to "
            "results/<model>-baseline2-causal-<mode>.jsonl."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only this many unfinished examples.",
    )
    parser.add_argument(
        "--thinking",
        required=True,
        choices=["on", "off"],
        help="Enable or disable Qwen's thinking mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Generation budget. Defaults to 32768 in thinking mode "
            "and 4096 in non-thinking mode."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for reproducible thinking-mode sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Prompts submitted in each resumable chunk. The default 0 submits "
            "all unfinished prompts together for maximum vLLM throughput."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of visible GPUs across which vLLM shards one model.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of each selected GPU's memory available to vLLM.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Maximum number of sequences vLLM may process concurrently.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Maximum prompt plus completion length. Defaults to max-new-tokens "
            "+ 8192, reserving prompt space for this dataset."
        ),
    )
    parser.add_argument(
        "--sampler-backend",
        choices=["native", "flashinfer"],
        default="native",
        help=(
            "Top-k/top-p sampler implementation. 'native' avoids FlashInfer "
            "JIT compilation and is the reproducible default."
        ),
    )
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=["triton", "flashinfer", "auto"],
        default="triton",
        help=(
            "Qwen GDN prefill implementation. 'triton' avoids FlashInfer JIT "
            "compilation and is the reproducible default."
        ),
    )
    args = parser.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = 32768 if args.thinking == "on" else 4096
    if args.max_model_len is None:
        args.max_model_len = args.max_new_tokens + 8192

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must be greater than --max-new-tokens.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.batch_size < 0:
        parser.error("--batch-size must be 0 or a positive integer.")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1.")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be at least 1.")

    return args


def load_vllm():
    """Import vLLM lazily so importing this module cannot initialize CUDA."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise SystemExit(
            "vLLM is not installed in this environment. Qwen3.5 requires a "
            "recent vLLM build. Install it with:\n\n"
            "  pip install --upgrade uv\n"
            "  uv pip install -U vllm --torch-backend=auto "
            "--extra-index-url https://wheels.vllm.ai/nightly"
        ) from error

    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = "unknown"

    return LLM, SamplingParams, vllm_version


def split_qwen_completion(raw_completion, thinking_enabled, budget_exhausted):
    """Separate Qwen's thinking trace without failing on truncation."""
    cleaned = raw_completion
    for marker in ("<|im_end|>", "<|endoftext|>"):
        cleaned = cleaned.replace(marker, "")
    cleaned = cleaned.strip()

    if not thinking_enabled:
        return None, cleaned, "not_applicable"

    if "</think>" not in cleaned:
        thinking_part = cleaned.replace("<think>", "", 1).strip()
        thinking_status = (
            "budget_exhausted"
            if budget_exhausted
            else "incomplete_thinking_trace"
        )
        return thinking_part, None, thinking_status

    thinking_part, final_part = cleaned.split("</think>", 1)
    thinking_part = thinking_part.replace("<think>", "", 1).strip()
    return thinking_part, final_part.strip(), "completed"


def build_prompt(example):
    """Build the fixed baseline_2 causal prompt without reference annotations."""
    return f"""Story:
{example["story"]}

Counterfactual intervention:
{example["query"]}

Treat the requested change as an intervention on the world described by the
story. Keep fixed all background facts that are causally upstream of the
intervention, along with any other story details it does not affect. Update
the intervened event and every downstream consequence it changes so that the
edited story remains causally consistent with the original context. Do not
make any other changes.

Return only the fully edited story."""


def make_sampling_params(
    SamplingParams,
    thinking_enabled,
    max_new_tokens,
    example_seed,
):
    """Create request-local settings, including a stable sampling seed."""
    common = {
        "max_tokens": max_new_tokens,
        "skip_special_tokens": False,
    }
    if thinking_enabled:
        return SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            seed=example_seed,
            **common,
        )
    return SamplingParams(temperature=0.0, **common)


def append_record(output_path, record):
    """Write and flush one result so completed chunks remain resumable."""
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def print_result(
    position,
    total,
    example,
    response,
    budget_exhausted,
    generated_token_count,
    thinking_status,
):
    print(f"\n[{position}/{total}] Spreadsheet row {example['row_number']}")
    print("MODEL RESPONSE:")
    print("[No completed final response]" if response is None else response)

    if budget_exhausted:
        print(
            f"STATUS: budget_exhausted after {generated_token_count} "
            "generated tokens; continuing."
        )
    elif thinking_status == "incomplete_thinking_trace":
        print("STATUS: incomplete_thinking_trace; saved and continuing.")

    print("\nEXPECTED CONSEQUENCE:")
    print(example["expectation"])
    print("-" * 70)


def load_completed_ids(output_path):
    completed_ids = set()
    if not output_path.exists():
        return completed_ids

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                completed_ids.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed_ids


def load_examples(input_path, output_path, limit):
    """Load the spreadsheet and omit examples already present in the JSONL."""
    import pandas as pd

    data = pd.read_excel(input_path, sheet_name="Final Examples")
    required_columns = {
        "Story",
        "Counterfactual Query",
        "Expected Edit",
        "Reasoning",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing spreadsheet columns: {sorted(missing)}")

    data = data.dropna(subset=["Story", "Counterfactual Query"])
    completed_ids = load_completed_ids(output_path)
    examples = []

    for row_number, (_, row) in enumerate(data.iterrows(), start=1):
        story = str(row["Story"]).strip()
        query = str(row["Counterfactual Query"]).strip()
        digest = hashlib.sha256(f"{story}\n{query}".encode("utf-8")).hexdigest()[:12]
        example_id = f"example-{digest}"

        if example_id in completed_ids:
            continue

        examples.append(
            {
                "example_id": example_id,
                "row_number": row_number,
                "story": story,
                "query": query,
                "expectation": str(row["Expected Edit"]).strip(),
                "knowledge_reasoning": str(row["Reasoning"]).strip(),
            }
        )

    return examples if limit is None else examples[:limit]


def save_batch_outputs(
    *,
    batch,
    prompts,
    example_seeds,
    outputs,
    output_path,
    args,
    revision,
    mode_name,
    generation_profile,
    vllm_version,
    thinking_enabled,
    batch_start,
    total_examples,
):
    if len(outputs) != len(batch):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts."
        )

    for offset, (example, prompt, example_seed, request_output) in enumerate(
        zip(batch, prompts, example_seeds, outputs),
        start=1,
    ):
        completion = request_output.outputs[0]
        raw_completion = completion.text.strip()
        generated_token_count = len(completion.token_ids)
        budget_exhausted = (
            completion.finish_reason == "length"
            or generated_token_count >= args.max_new_tokens
        )
        thinking_text, response, thinking_status = split_qwen_completion(
            raw_completion,
            thinking_enabled,
            budget_exhausted,
        )

        tags = []
        if budget_exhausted:
            tags.append("budget_exhausted")
        if thinking_status == "incomplete_thinking_trace":
            tags.append("incomplete_thinking_trace")

        record = {
            **example,
            "model": args.model,
            "revision": revision,
            "condition": f"baseline_2_causal_{mode_name}",
            "prompt_profile": "baseline_2_causal_v1",
            "generation_profile": generation_profile,
            "inference_backend": "vllm",
            "inference_backend_version": vllm_version,
            "thinking_enabled": thinking_enabled,
            "prompt": prompt,
            "raw_completion": raw_completion,
            "thinking": thinking_text,
            "thinking_status": thinking_status,
            "response": response,
            "tags": tags,
            "generated_token_count": generated_token_count,
            "generation_config": {
                "do_sample": thinking_enabled,
                "max_new_tokens": args.max_new_tokens,
                "temperature": 1.0 if thinking_enabled else 0.0,
                "top_p": 0.95 if thinking_enabled else None,
                "top_k": 20 if thinking_enabled else None,
                "min_p": 0.0 if thinking_enabled else None,
                "presence_penalty": 1.5 if thinking_enabled else None,
                "repetition_penalty": 1.0 if thinking_enabled else None,
                "seed": example_seed,
                "tensor_parallel_size": args.tensor_parallel_size,
                "sampler_backend": args.sampler_backend,
                "gdn_prefill_backend": args.gdn_prefill_backend,
            },
        }
        append_record(output_path, record)
        print_result(
            batch_start + offset,
            total_examples,
            example,
            response,
            budget_exhausted,
            generated_token_count,
            thinking_status,
        )


def main():
    # Nothing above this function initializes CUDA or starts a worker. This is
    # required because vLLM's spawned engine process imports this file again.
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    thinking_enabled = args.thinking == "on"
    mode_name = "thinking" if thinking_enabled else "nonthinking"
    generation_profile = (
        "qwen3.5_thinking_general_v1"
        if thinking_enabled
        else "greedy_nonthinking_v1"
    )
    revision = REVISIONS[args.model]
    model_short_name = args.model.split("/")[-1].lower()
    output_path = (
        Path(args.output)
        if args.output
        else Path("results")
        / f"{model_short_name}-baseline2-causal-{mode_name}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_examples(input_path, output_path, args.limit)
    if not examples:
        print("No unfinished examples remain.")
        print("Results:", output_path)
        return 0

    print("Model:", args.model)
    print("Revision:", revision)
    print("Inference backend: vLLM")
    print("Thinking mode:", mode_name)
    print("Prompt profile: baseline_2_causal_v1")
    print("Generation profile:", generation_profile)
    print("Maximum new tokens:", args.max_new_tokens)
    print("Maximum model length:", args.max_model_len)
    print("Tensor parallel size:", args.tensor_parallel_size)
    print("Maximum concurrent sequences:", args.max_num_seqs)
    print("GPU memory utilization:", args.gpu_memory_utilization)
    print("Sampler backend:", args.sampler_backend)
    print("GDN prefill backend:", args.gdn_prefill_backend)
    if thinking_enabled:
        print(
            "Thinking sampling: temperature=1.0, top_p=0.95, "
            "top_k=20, min_p=0.0, presence_penalty=1.5"
        )
    print("Examples to run:", len(examples))
    print(
        "Submission batch size:",
        len(examples) if args.batch_size == 0 else args.batch_size,
    )
    print("Results:", output_path)
    print("Loading vLLM model...")

    # This must be set before importing vLLM/Hugging Face libraries.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.sampler_backend == "flashinfer" else "0"
    )
    LLM, SamplingParams, vllm_version = load_vllm()

    llm = LLM(
        model=args.model,
        revision=revision,
        tokenizer_revision=revision,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        language_model_only=True,
        additional_config={
            "gdn_prefill_backend": args.gdn_prefill_backend,
        },
    )

    submission_batch_size = args.batch_size or len(examples)
    total_batches = (
        len(examples) + submission_batch_size - 1
    ) // submission_batch_size

    for batch_start in range(0, len(examples), submission_batch_size):
        batch = examples[batch_start : batch_start + submission_batch_size]
        prompts = [build_prompt(example) for example in batch]
        conversations = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        example_seeds = [
            args.seed + example["row_number"] if thinking_enabled else None
            for example in batch
        ]
        sampling_params = [
            make_sampling_params(
                SamplingParams,
                thinking_enabled,
                args.max_new_tokens,
                example_seed,
            )
            for example_seed in example_seeds
        ]

        batch_number = (batch_start // submission_batch_size) + 1
        print(
            f"\nSubmitting batch {batch_number}/{total_batches} "
            f"({len(batch)} prompts) to vLLM..."
        )
        outputs = llm.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": thinking_enabled},
        )
        save_batch_outputs(
            batch=batch,
            prompts=prompts,
            example_seeds=example_seeds,
            outputs=outputs,
            output_path=output_path,
            args=args,
            revision=revision,
            mode_name=mode_name,
            generation_profile=generation_profile,
            vllm_version=vllm_version,
            thinking_enabled=thinking_enabled,
            batch_start=batch_start,
            total_examples=len(examples),
        )

    print("\nFinished.")
    print("Results saved to:", output_path)
    return 0


if __name__ == "__main__":
    # Harmless for normal scripts and required only for frozen executables;
    # keeping it here makes the multiprocessing entry point explicit.
    from multiprocessing import freeze_support

    freeze_support()
    raise SystemExit(main())


#!/usr/bin/env python3
"""Run cumulative P1-P5 hypothesis-testing prompts with one vLLM model load."""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import freeze_support
from pathlib import Path

try:
    from prompts import run_baseline2_causal as common
except ModuleNotFoundError:
    # Backward compatibility with the earlier project layout.
    import run_baseline2_causal as common


PROMPT_INSTRUCTIONS = {
    "p1": """Treat the requested change as an intervention on the world described by the
story.

Keep fixed all background facts that are causally upstream of the intervention,
along with any other story details it does not affect. Update the intervened event
and every downstream consequence it changes so that the edited story remains causally
consistent with the original context.

First identify which statements in the story are causally affected by
the requested change. These are not to be treated as background facts,
even when explicitly stated. Update them where necessary; preserve only causally
unaffected details.

Do not make any other changes.

Return only the fully edited story.""",
    "p2": """Treat the requested change as an intervention on the world described by the
story.

Keep fixed all background facts that are causally upstream of the intervention,
along with any other story details it does not affect. Update the intervened event
and every downstream consequence it changes so that the edited story remains causally
consistent with the original context.

Identify which statements in the story are causally affected by
the requested change. These are not to be treated as background facts,
even when explicitly stated. Update them where necessary; preserve only causally
unaffected details.

Treat every requested change as externally fixed. Identify which events/facts in the story
are not causally affected by the requested change. These could be prior events/facts which themselves
causally affected the event/fact being changed, or are simply prior in the narrative.
Never alter these upstream events to make the response more plausible. An unusual stipulated
combination may remain unexplained. You may change relevant connector words like "because",
"therefore", "however", etc., to reflect the new causal relationships in the edited story.

Do not make any other changes.

Return only the fully edited story.""",
    "p3": """Treat the requested change as an intervention on the world described by the
story.

Keep fixed all background facts that are causally upstream of the intervention,
along with any other story details it does not affect. Update the intervened event
and every downstream consequence it changes so that the edited story remains causally
consistent with the original context.

Identify which statements in the story are causally affected by
the requested change. These are not to be treated as background facts,
even when explicitly stated. Update them where necessary; preserve only causally
unaffected details.

Treat every requested change as externally fixed. Identify which events/facts in the story
are not causally affected by the requested change. These could be prior events/facts which themselves
causally affected the event/fact being changed, or are simply prior in the narrative.
Never alter these upstream events to make the response more plausible. An unusual stipulated
combination may remain unexplained. You may change relevant connector words like "because",
"therefore", "however", etc., to reflect the new causal relationships in the edited story.

Assume ordinary/default circumstances in the original story. Do not invent unstated exceptions,
extreme circumstances, compensating causes, or unusual mechanisms to forcibly preserve an original
outcome.

Do not make any other changes.

Return only the fully edited story.""",
    "p4": """Treat the requested change as an intervention on the world described by the
story.

Keep fixed all background facts that are causally upstream of the intervention,
along with any other story details it does not affect. Update the intervened event
and every downstream consequence it changes so that the edited story remains causally
consistent with the original context.

Identify which statements in the story are causally affected by
the requested change. These are not to be treated as background facts,
even when explicitly stated. Update them where necessary; preserve only causally
unaffected details.

Treat every requested change as externally fixed. Identify which events/facts in the story
are not causally affected by the requested change. These could be prior events/facts which themselves
causally affected the event/fact being changed, or are simply prior in the narrative.
Never alter these upstream events to make the response more plausible. An unusual stipulated
combination may remain unexplained. You may change relevant connector words like "because",
"therefore", "however", etc., to reflect the new causal relationships in the edited story.

Assume ordinary/default circumstances in the original story. Do not invent unstated exceptions,
extreme circumstances, compensating causes, or unusual mechanisms to forcibly preserve an original
outcome.

If the exact counterfactual outcome cannot be numerically determined, state the strongest justified
qualitative change—for example, less, more, delayed, or absent. Directional changes are fine too.

Do not make any other changes.

Return only the fully edited story.""",
    "p5": """Treat the requested change as an intervention on the world described by the
story.

Keep fixed all background facts that are causally upstream of the intervention,
along with any other story details it does not affect. Update the intervened event
and every downstream consequence it changes so that the edited story remains causally
consistent with the original context.

Identify which statements in the story are causally affected by
the requested change. These are not to be treated as background facts,
even when explicitly stated. Update them where necessary; preserve only causally
unaffected details.

Before editing, silently generate relevant unstated facts implied by the story and actively
recall any ordinary world knowledge needed to determine the consequences. Use those facts
when updating the story.

Treat every requested change as externally fixed. Identify which events/facts in the story
are not causally affected by the requested change. These could be prior events/facts which themselves
causally affected the event/fact being changed, or are simply prior in the narrative.
Never alter these upstream events to make the response more plausible. An unusual stipulated
combination may remain unexplained. You may change relevant connector words like "because",
"therefore", "however", etc., to reflect the new causal relationships in the edited story.

Assume ordinary/default circumstances in the original story. Do not invent unstated exceptions,
extreme circumstances, compensating causes, or unusual mechanisms to forcibly preserve an original
outcome.

If the exact counterfactual outcome cannot be numerically determined, state the strongest justified
qualitative change—for example, less, more, delayed, or absent. Directional changes are fine too.

Do not make any other changes.

Return only the fully edited story.""",
    "p6": """Treat the requested change as an intervention on the world described by the
story.

Keep fixed all background facts that are causally upstream of the intervention,
along with any other story details it does not affect. Update the intervened event
and every downstream consequence it changes so that the edited story remains causally
consistent with the original context.

Identify which statements in the story are causally affected by
the requested change. These are not to be treated as background facts,
even when explicitly stated. Update them where necessary; preserve only causally
unaffected details.

Before editing, silently generate relevant unstated facts implied by the story and actively
recall any ordinary world knowledge needed to determine the consequences. Use those facts
when updating the story.

Treat every requested change as externally fixed. Identify which events/facts in the story
are not causally affected by the requested change. These could be prior events/facts which themselves
causally affected the event/fact being changed, or are simply prior in the narrative.
Never alter these upstream events to make the response more plausible. An unusual stipulated
combination may remain unexplained. You may change relevant connector words like "because",
"therefore", "however", etc., to reflect the new causal relationships in the edited story.

If the exact counterfactual outcome cannot be numerically determined, state the strongest justified
qualitative change—for example, less, more, delayed, or absent. Directional changes are fine too.

Do not make any other changes.

Return only the fully edited story."""
}

# Added prompt p6 to remove the seemingly unhelpful p3 paragraph, just to see if it improves performance
# Removed from p6: Assume ordinary/default circumstances in the original story. Do not invent unstated exceptions, 
# extreme circumstances, compensating causes, or unusual mechanisms in the original story.

HYPOTHESES = {
    "p1": ["H1.1"],
    "p2": ["H1.1", "H1.2"],
    "p3": ["H1.1", "H1.2", "H1.3"],
    "p4": ["H1.1", "H1.2", "H1.3", "H2"],
    "p5": ["H1.1", "H1.2", "H1.3", "H2", "H3/H5"],
    "p6": ["H1.1", "H1.2", "H2", "H3/H5"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cumulative P1-P5 prompt tests using one vLLM model load."
    )
    parser.add_argument("--model", required=True, choices=list(common.REVISIONS))
    parser.add_argument("--thinking", required=True, choices=["on", "off"])
    parser.add_argument(
        "--input",
        default="data/counterfactual_causal_propagation_examples_19.xlsm",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hypothesis_tests",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=list(PROMPT_INSTRUCTIONS),
        default=list(PROMPT_INSTRUCTIONS),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--sampler-backend",
        choices=["native", "flashinfer"],
        default="native",
    )
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=["triton", "flashinfer", "auto"],
        default="triton",
    )
    args = parser.parse_args()

    thinking_enabled = args.thinking == "on"
    if args.max_new_tokens is None:
        args.max_new_tokens = 32768 if thinking_enabled else 4096
    if args.max_model_len is None:
        args.max_model_len = args.max_new_tokens + 8192
    if args.max_num_seqs is None:
        args.max_num_seqs = 16 if args.model.endswith("9B") else 6

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1.")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must exceed --max-new-tokens.")
    if args.batch_size < 0:
        parser.error("--batch-size must be 0 or a positive integer.")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be at least 1.")
    return args


def build_prompt(example: dict, prompt_id: str) -> str:
    return f"""Story:
{example["story"]}

Counterfactual intervention:
{example["query"]}

{PROMPT_INSTRUCTIONS[prompt_id]}"""


def save_outputs(
    *,
    outputs,
    batch: list[dict],
    prompts: list[str],
    seeds: list[int | None],
    output_path: Path,
    prompt_id: str,
    args: argparse.Namespace,
    revision: str,
    mode_name: str,
    generation_profile: str,
    vllm_version: str,
    batch_start: int,
    total: int,
) -> None:
    if len(outputs) != len(batch):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts."
        )

    for offset, (example, prompt, example_seed, request_output) in enumerate(
        zip(batch, prompts, seeds, outputs), start=1
    ):
        completion = request_output.outputs[0]
        raw_completion = completion.text.strip()
        token_count = len(completion.token_ids)
        exhausted = (
            completion.finish_reason == "length"
            or token_count >= args.max_new_tokens
        )
        thinking, response, thinking_status = common.split_qwen_completion(
            raw_completion,
            args.thinking == "on",
            exhausted,
        )
        tags = []
        if exhausted:
            tags.append("budget_exhausted")
        if thinking_status == "incomplete_thinking_trace":
            tags.append("incomplete_thinking_trace")

        record = {
            **example,
            "model": args.model,
            "revision": revision,
            "condition": f"hypothesis_{prompt_id}_{mode_name}",
            "prompt_id": prompt_id,
            "prompt_profile": f"hypothesis_{prompt_id}_cumulative_v1",
            "hypotheses_targeted": HYPOTHESES[prompt_id],
            "generation_profile": generation_profile,
            "inference_backend": "vllm",
            "inference_backend_version": vllm_version,
            "thinking_enabled": args.thinking == "on",
            "prompt": prompt,
            "raw_completion": raw_completion,
            "thinking": thinking,
            "thinking_status": thinking_status,
            "response": response,
            "tags": tags,
            "generated_token_count": token_count,
            "generation_config": {
                "do_sample": args.thinking == "on",
                "max_new_tokens": args.max_new_tokens,
                "temperature": 1.0 if args.thinking == "on" else 0.0,
                "top_p": 0.95 if args.thinking == "on" else None,
                "top_k": 20 if args.thinking == "on" else None,
                "min_p": 0.0 if args.thinking == "on" else None,
                "presence_penalty": 1.5 if args.thinking == "on" else None,
                "repetition_penalty": 1.0 if args.thinking == "on" else None,
                "seed": example_seed,
                "tensor_parallel_size": args.tensor_parallel_size,
                "sampler_backend": args.sampler_backend,
                "gdn_prefill_backend": args.gdn_prefill_backend,
            },
        }
        common.append_record(output_path, record)
        common.print_result(
            batch_start + offset,
            total,
            example,
            response,
            exhausted,
            token_count,
            thinking_status,
        )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thinking_enabled = args.thinking == "on"
    mode_name = "thinking" if thinking_enabled else "nonthinking"
    generation_profile = (
        "qwen3.5_thinking_general_v1"
        if thinking_enabled
        else "greedy_nonthinking_v1"
    )
    model_short_name = args.model.split("/")[-1].lower()
    revision = common.REVISIONS[args.model]

    jobs = []
    for prompt_id in args.prompts:
        output_path = output_dir / (
            f"{model_short_name}-hypothesis-{prompt_id}-{mode_name}.jsonl"
        )
        examples = common.load_examples(input_path, output_path, args.limit)
        if examples:
            jobs.append((prompt_id, output_path, examples))
        else:
            print(f"{prompt_id.upper()}: no unfinished examples ({output_path})")

    if not jobs:
        print("No unfinished hypothesis tests remain for this model/mode.")
        return 0

    print("Model:", args.model)
    print("Revision:", revision)
    print("Thinking mode:", mode_name)
    print("Prompt levels:", ", ".join(prompt_id for prompt_id, _, _ in jobs))
    print("Output directory:", output_dir)
    print("Maximum new tokens:", args.max_new_tokens)
    print("Maximum concurrent sequences:", args.max_num_seqs)
    print("Loading vLLM model once for all selected prompt levels...")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.sampler_backend == "flashinfer" else "0"
    )
    LLM, SamplingParams, vllm_version = common.load_vllm()
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
        additional_config={"gdn_prefill_backend": args.gdn_prefill_backend},
    )

    for prompt_id, output_path, examples in jobs:
        print("\n" + "=" * 78)
        print(f"Running {prompt_id.upper()} -> {output_path}")
        submission_size = args.batch_size or len(examples)
        for batch_start in range(0, len(examples), submission_size):
            batch = examples[batch_start : batch_start + submission_size]
            prompts = [build_prompt(example, prompt_id) for example in batch]
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in prompts
            ]
            seeds = [
                args.seed + example["row_number"] if thinking_enabled else None
                for example in batch
            ]
            sampling_params = [
                common.make_sampling_params(
                    SamplingParams,
                    thinking_enabled,
                    args.max_new_tokens,
                    example_seed,
                )
                for example_seed in seeds
            ]
            print(
                f"Submitting {len(batch)} prompts "
                f"({batch_start + 1}-{batch_start + len(batch)} of {len(examples)})..."
            )
            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                use_tqdm=True,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
            )
            save_outputs(
                outputs=outputs,
                batch=batch,
                prompts=prompts,
                seeds=seeds,
                output_path=output_path,
                prompt_id=prompt_id,
                args=args,
                revision=revision,
                mode_name=mode_name,
                generation_profile=generation_profile,
                vllm_version=vllm_version,
                batch_start=batch_start,
                total=len(examples),
            )

    print("\nFinished all selected prompt levels.")
    print("Results directory:", output_dir)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

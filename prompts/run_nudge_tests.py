#!/usr/bin/env python3
"""Run targeted prompt nudges with Qwen thinking mode and vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing import freeze_support
from pathlib import Path
from typing import Iterator

try:
    from prompts import run_baseline2_causal as common
except (ImportError, ModuleNotFoundError):
    import run_baseline2_causal as common


P5_INSTRUCTIONS = """Treat the requested change as an intervention on the world described by the
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

Return only the fully edited story."""


CASE_CONFIG = {
    "salah": {
        "example_id": "example-0968735aaf5d",
        "elicitation_question": (
            "In a normal soccer match, do teams switch which end they attack "
            "after half-time?\n\nAnswer briefly."
        ),
        "background_fact": (
            "Soccer teams normally switch ends at half-time. At Anfield, the "
            "Anfield Road end is opposite the Kop."
        ),
        "intermediate_reasoning": (
            "Moving the scoring event from immediately before half-time to the "
            "start of the second half crosses the change of ends. Liverpool "
            "would therefore be attacking the opposite physical end of the stadium."
        ),
    },
    "madrid": {
        "example_id": "example-24c6a9c9ae59",
        "elicitation_question": (
            "Consider this story:\n\n"
            "In an incredible soccer game, Real Madrid players Ronaldo, Bale, "
            "and Benzema each scored 2 goals in the first 30 minutes. The only "
            "two goals in the second half came from Ronaldo and Bale. Madrid "
            "ended up winning 11-2.\n\n"
            "Based on all the statements in the story, how many of Madrid's "
            "goals occurred in the first half?\n\n"
            "Answer concisely and show the arithmetic in one sentence."
        ),
        "background_fact": (
            "In the original game, Madrid scored nine goals in the first half: "
            "11 total goals minus the only two second-half goals."
        ),
        "intermediate_reasoning": (
            "Changing Madrid's second-half total to zero removes the two "
            "second-half goals but does not remove any first-half goals, "
            "including goals whose scorers were not individually named."
        ),
    },
    "starry_night": {
        "example_id": "example-7ad4e247e511",
        "elicitation_question": (
            "Which New York museum normally holds Van Gogh's The Starry Night?\n\n"
            "Answer concisely with the museum's name."
        ),
        "background_fact": (
            "Van Gogh's The Starry Night is normally held by the Museum of "
            "Modern Art (MoMA), not the Metropolitan Museum of Art."
        ),
        "intermediate_reasoning": (
            "Opening the Metropolitan Museum of Art's underground collections "
            "changes access to collections at that museum. It does not transfer "
            "artworks from another museum or change which museum holds them."
        ),
    },
}

CONDITIONS = ("baseline_p5", "nudge_1", "nudge_2", "nudge_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted P5 prompt nudges with Qwen thinking mode."
    )
    parser.add_argument(
        "--model",
        choices=list(common.REVISIONS),
        default="Qwen/Qwen3.5-9B",
    )
    parser.add_argument(
        "--input",
        default="data/counterfactual_causal_propagation_examples_19.xlsm",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL destination. Existing results are resumed by default.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASE_CONFIG),
        default=list(CASE_CONFIG),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Requests per saved batch. Use 0 to submit all requests together.",
    )
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

    if args.max_model_len is None:
        args.max_model_len = args.max_new_tokens + 8192
    if args.max_num_seqs is None:
        args.max_num_seqs = 16 if args.model.endswith("9B") else 6
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
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates.")
    return args


def load_selected_examples(input_path: Path, case_names: list[str]) -> list[dict]:
    """Load configured examples from the source spreadsheet."""
    import pandas as pd

    data = pd.read_excel(input_path, sheet_name="Final Examples")
    required = {"Story", "Counterfactual Query", "Expected Edit", "Reasoning"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing spreadsheet columns: {sorted(missing)}")

    wanted = {
        CASE_CONFIG[case_name]["example_id"]: case_name
        for case_name in case_names
    }
    found = {}
    data = data.dropna(subset=["Story", "Counterfactual Query"])
    for row_number, (_, row) in enumerate(data.iterrows(), start=1):
        story = str(row["Story"]).strip()
        query = str(row["Counterfactual Query"]).strip()
        digest = hashlib.sha256(f"{story}\n{query}".encode("utf-8")).hexdigest()[:12]
        example_id = f"example-{digest}"
        if example_id not in wanted:
            continue
        case_name = wanted[example_id]
        found[case_name] = {
            "case_name": case_name,
            "example_id": example_id,
            "row_number": row_number,
            "story": story,
            "query": query,
            "expectation": str(row["Expected Edit"]).strip(),
            "knowledge_reasoning": str(row["Reasoning"]).strip(),
        }

    missing_cases = [case_name for case_name in case_names if case_name not in found]
    if missing_cases:
        raise ValueError(
            "Configured examples were not found in the spreadsheet: "
            + ", ".join(missing_cases)
        )
    return [found[case_name] for case_name in case_names]


def build_edit_prompt(example: dict, condition: str) -> str:
    config = CASE_CONFIG[example["case_name"]]
    additions = ""
    if condition == "nudge_2":
        additions = f'\nBackground facts:\n{config["background_fact"]}\n'
    elif condition == "nudge_3":
        additions = (
            f'\nBackground facts:\n{config["background_fact"]}\n\n'
            f'Intermediate reasoning step:\n{config["intermediate_reasoning"]}\n'
        )

    return f"""Story:
{example["story"]}

Counterfactual intervention:
{example["query"]}
{additions}
{P5_INSTRUCTIONS}"""


def read_records(path: Path) -> Iterator[dict]:
    """Read complete JSONL records without silently skipping damaged data."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}. "
                    "Check the incomplete record before resuming."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            yield record


def append_record(path: Path, record: dict) -> None:
    """Persist one record before starting another inference batch."""
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if file.tell():
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                file.write(b"\n")
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def completed_keys(output_path: Path) -> set[tuple[str, str, int]]:
    return {
        (str(record["example_id"]), str(record["condition"]), int(record["seed"]))
        for record in read_records(output_path)
    }


def load_elicitations(
    cache_path: Path,
    jobs: list[dict],
    args: argparse.Namespace,
    revision: str,
    vllm_version: str,
) -> dict:
    """Reuse factual responses generated with matching inputs and settings."""
    wanted = {
        (job["example"]["example_id"], job["seed"]): job
        for job in jobs
    }
    results = {}
    for record in read_records(cache_path):
        key = (record["example_id"], record["seed"])
        if key not in wanted:
            continue
        job = wanted[key]
        question = CASE_CONFIG[job["example"]["case_name"]]["elicitation_question"]
        if (
            record.get("model") != args.model
            or record.get("revision") != revision
            or record.get("inference_backend_version") != vllm_version
            or record.get("question") != question
            or record.get("generation_config") != generation_config(args, job["seed"])
        ):
            raise ValueError(
                f"Cached factual response has different inputs or settings: {key}. "
                "Use a new --output path for a different run."
            )
        results[key] = record["result"]
    return results


def unpack_output(request_output, max_tokens: int) -> dict:
    completion = request_output.outputs[0]
    raw_completion = completion.text.strip()
    token_count = len(completion.token_ids)
    budget_exhausted = (
        completion.finish_reason == "length" or token_count >= max_tokens
    )
    thinking, response, thinking_status = common.split_qwen_completion(
        raw_completion,
        True,
        budget_exhausted,
    )
    return {
        "raw_completion": raw_completion,
        "thinking": thinking,
        "thinking_status": thinking_status,
        "response": response,
        "generated_token_count": token_count,
        "budget_exhausted": budget_exhausted,
    }


def run_chat_batches(
    *,
    llm,
    SamplingParams,
    conversations: list[list[dict]],
    seeds: list[int],
    args: argparse.Namespace,
) -> Iterator[tuple[int, object]]:
    """Yield each completed batch's outputs before submitting another batch."""
    if len(conversations) != len(seeds):
        raise ValueError("Each conversation must have one sampling seed.")
    if not conversations:
        return
    submission_size = args.batch_size or len(conversations)
    for start in range(0, len(conversations), submission_size):
        batch_conversations = conversations[start : start + submission_size]
        batch_seeds = seeds[start : start + submission_size]
        sampling_params = [
            common.make_sampling_params(
                SamplingParams,
                True,
                args.max_new_tokens,
                seed,
            )
            for seed in batch_seeds
        ]
        print(
            f"Submitting {len(batch_conversations)} requests "
            f"({start + 1}-{start + len(batch_conversations)} of "
            f"{len(conversations)})..."
        )
        batch_outputs = llm.chat(
            batch_conversations,
            sampling_params=sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": True},
        )
        if len(batch_outputs) != len(batch_conversations):
            raise RuntimeError(
                f"vLLM returned {len(batch_outputs)} outputs for "
                f"{len(batch_conversations)} requests."
            )
        for offset, output in enumerate(batch_outputs):
            yield start + offset, output


def generation_config(args: argparse.Namespace, seed: int) -> dict:
    return {
        "do_sample": True,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "seed": seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "sampler_backend": args.sampler_backend,
        "gdn_prefill_backend": args.gdn_prefill_backend,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find: {input_path}")

    model_short_name = args.model.split("/")[-1].lower()
    output_path = (
        Path(args.output)
        if args.output
        else Path("results/nudge_tests")
        / f"{model_short_name}-thinking.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = load_selected_examples(input_path, args.cases)
    done = completed_keys(output_path)

    jobs = [
        {"example": example, "condition": condition, "seed": seed}
        for example in examples
        for condition in args.conditions
        for seed in args.seeds
        if (example["example_id"], condition, seed) not in done
    ]
    if not jobs:
        print("No unfinished requests remain.")
        print("Results:", output_path)
        return 0

    print("Model:", args.model)
    print("Revision:", common.REVISIONS[args.model])
    print("Cases:", ", ".join(args.cases))
    print("Conditions:", ", ".join(args.conditions))
    print("Seeds:", ", ".join(str(seed) for seed in args.seeds))
    print("Pending edit requests:", len(jobs))
    print("Requests per saved batch:", args.batch_size or "all")
    print("Results:", output_path)
    print("Loading vLLM model...")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.sampler_backend == "flashinfer" else "0"
    )
    LLM, SamplingParams, vllm_version = common.load_vllm()
    revision = common.REVISIONS[args.model]
    llm = LLM(
        model=args.model,
        revision=revision,
        tokenizer_revision=revision,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=min(args.seeds),
        language_model_only=True,
        additional_config={"gdn_prefill_backend": args.gdn_prefill_backend},
    )

    nudge_1_jobs = [job for job in jobs if job["condition"] == "nudge_1"]
    elicitations = {}
    if nudge_1_jobs:
        cache_path = output_path.with_suffix(".elicitations.jsonl")
        elicitations = load_elicitations(
            cache_path, nudge_1_jobs, args, revision, vllm_version
        )
        missing_elicitations = [
            job for job in nudge_1_jobs
            if (job["example"]["example_id"], job["seed"]) not in elicitations
        ]
        print(f"\nReusing {len(elicitations)} saved factual responses.")
        print(f"Generating {len(missing_elicitations)} factual responses for nudge_1...")
        elicitation_conversations = [
            [
                {
                    "role": "user",
                    "content": CASE_CONFIG[job["example"]["case_name"]][
                        "elicitation_question"
                    ],
                }
            ]
            for job in missing_elicitations
        ]
        for index, request_output in run_chat_batches(
            llm=llm,
            SamplingParams=SamplingParams,
            conversations=elicitation_conversations,
            seeds=[job["seed"] for job in missing_elicitations],
            args=args,
        ):
            job = missing_elicitations[index]
            key = (job["example"]["example_id"], job["seed"])
            result = unpack_output(request_output, args.max_new_tokens)
            append_record(cache_path, {
                "example_id": job["example"]["example_id"],
                "model": args.model,
                "revision": revision,
                "seed": job["seed"],
                "question": elicitation_conversations[index][0]["content"],
                "inference_backend_version": vllm_version,
                "generation_config": generation_config(args, job["seed"]),
                "result": result,
            })
            elicitations[key] = result

    runnable_jobs = []
    conversations = []
    seeds = []
    for job in jobs:
        example = job["example"]
        condition = job["condition"]
        seed = job["seed"]
        edit_prompt = build_edit_prompt(example, condition)
        if condition == "nudge_1":
            elicitation = elicitations[(example["example_id"], seed)]
            if not elicitation["response"]:
                record = {
                    **example,
                    "model": args.model,
                    "revision": revision,
                    "condition": condition,
                    "prompt_profile": "p5_nudge_v1",
                    "seed": seed,
                    "thinking_enabled": True,
                    "prompt": edit_prompt,
                    "conversation": None,
                    "elicitation_question": CASE_CONFIG[example["case_name"]][
                        "elicitation_question"
                    ],
                    "elicitation_raw_completion": elicitation["raw_completion"],
                    "elicitation_thinking": elicitation["thinking"],
                    "elicitation_thinking_status": elicitation["thinking_status"],
                    "elicited_fact_response": None,
                    "raw_completion": None,
                    "thinking": None,
                    "thinking_status": "not_run",
                    "response": None,
                    "tags": ["elicitation_no_completed_response"],
                    "generated_token_count": 0,
                    "generation_profile": "qwen3.5_thinking_general_v1",
                    "inference_backend": "vllm",
                    "inference_backend_version": vllm_version,
                    "generation_config": generation_config(args, seed),
                }
                if elicitation["budget_exhausted"]:
                    record["tags"].append("elicitation_budget_exhausted")
                append_record(output_path, record)
                continue
            conversation = [
                {
                    "role": "user",
                    "content": CASE_CONFIG[example["case_name"]][
                        "elicitation_question"
                    ],
                },
                {"role": "assistant", "content": elicitation["response"]},
                {"role": "user", "content": edit_prompt},
            ]
        else:
            conversation = [{"role": "user", "content": edit_prompt}]

        runnable_jobs.append({**job, "prompt": edit_prompt, "conversation": conversation})
        conversations.append(conversation)
        seeds.append(seed)

    if runnable_jobs:
        print("\nGenerating edited stories...")
        for index, request_output in run_chat_batches(
            llm=llm,
            SamplingParams=SamplingParams,
            conversations=conversations,
            seeds=seeds,
            args=args,
        ):
            job = runnable_jobs[index]
            position = index + 1
            example = job["example"]
            condition = job["condition"]
            seed = job["seed"]
            result = unpack_output(request_output, args.max_new_tokens)
            elicitation = (
                elicitations.get((example["example_id"], seed))
                if condition == "nudge_1"
                else None
            )
            tags = []
            if result["budget_exhausted"]:
                tags.append("budget_exhausted")
            if result["thinking_status"] == "incomplete_thinking_trace":
                tags.append("incomplete_thinking_trace")
            if elicitation and elicitation["budget_exhausted"]:
                tags.append("elicitation_budget_exhausted")

            record = {
                **example,
                "model": args.model,
                "revision": revision,
                "condition": condition,
                "prompt_profile": "p5_nudge_v1",
                "seed": seed,
                "generation_profile": "qwen3.5_thinking_general_v1",
                "inference_backend": "vllm",
                "inference_backend_version": vllm_version,
                "thinking_enabled": True,
                "prompt": job["prompt"],
                "conversation": job["conversation"],
                "elicitation_question": (
                    CASE_CONFIG[example["case_name"]]["elicitation_question"]
                    if condition == "nudge_1"
                    else None
                ),
                "elicitation_raw_completion": (
                    elicitation["raw_completion"] if elicitation else None
                ),
                "elicitation_thinking": (
                    elicitation["thinking"] if elicitation else None
                ),
                "elicitation_thinking_status": (
                    elicitation["thinking_status"] if elicitation else None
                ),
                "elicited_fact_response": (
                    elicitation["response"] if elicitation else None
                ),
                "raw_completion": result["raw_completion"],
                "thinking": result["thinking"],
                "thinking_status": result["thinking_status"],
                "response": result["response"],
                "tags": tags,
                "generated_token_count": result["generated_token_count"],
                "generation_config": generation_config(args, seed),
            }
            append_record(output_path, record)
            print(
                f"[{position}/{len(runnable_jobs)}] {example['case_name']} | "
                f"{condition} | seed {seed}"
            )
            print(
                "[No completed final response]"
                if result["response"] is None
                else result["response"]
            )

    print("\nFinished.")
    print("Results saved to:", output_path)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
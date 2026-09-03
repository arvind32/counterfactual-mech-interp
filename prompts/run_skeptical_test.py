#!/usr/bin/env python3
"""Run six controlled story variants with Qwen thinking mode and vLLM.

Run from the project root: python -m prompts.run_skeptical_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from multiprocessing import freeze_support
from pathlib import Path

from prompts import run_nudge_tests as nudge


EXPERIMENT = "skeptical_test"
PROMPT_PROFILE = "p5_skeptical_v1"
CONDITIONS = ("baseline_p5", "nudge_1", "nudge_2", "nudge_3")
DEFAULT_CONDITIONS = ("baseline_p5", "nudge_2", "nudge_3")
ELICITATION_FIELDS = {
    "raw_completion": "elicitation_raw_completion",
    "thinking": "elicitation_thinking",
    "thinking_status": "elicitation_thinking_status",
    "response": "elicited_fact_response",
    "generated_token_count": "elicitation_generated_token_count",
    "budget_exhausted": "elicitation_budget_exhausted",
}
VARIANTS = {
    "a": ("a", "Real Madrid", "Madrid", ("Ronaldo", "Bale", "Benzema"), 11),
    "b": ("b", "Real Madrid", "Madrid", ("Ronaldo", "Bale", "Benzema"), 11),
    "a_names": ("a", "Harbor FC", "Harbor FC", ("Alex", "Ben", "Chris"), 11),
    "b_names": (
        "b", "Riverside FC", "Riverside FC", ("Daniel", "Ethan", "Felix"), 11,
    ),
    "a_total12": ("a", "Real Madrid", "Madrid", ("Ronaldo", "Bale", "Benzema"), 12),
    "b_total10": ("b", "Real Madrid", "Madrid", ("Ronaldo", "Bale", "Benzema"), 10),
}


def build_examples() -> list[dict]:
    """Construct story text, reference annotations, and condition-specific facts."""
    examples = []
    for row_number, (case_name, variant) in enumerate(VARIANTS.items(), start=1):
        wording, full_team, team, players, original_total = variant
        first, second, third = players
        if wording == "a":
            story = (
                f"In an incredible soccer game, {full_team} players {first}, "
                f"{second}, and {third} each scored two goals in the first 30 "
                "minutes. The only two goals in the second half came from "
                f"{first}, who completed his hat trick at minute 50, and "
                f"{second}, who completed his hat trick at minute 70. "
                f"{team} won {original_total}–2."
            )
        else:
            story = (
                f"During the first 30 minutes, {first}, {second}, and {third} "
                f"scored twice each for {full_team}. {first} completed his "
                f"hat trick in the 50th minute, and {second} completed his in "
                "the 70th minute. Those were the only two goals scored in "
                f"the second half. The final score was {original_total}–2 "
                f"to {team}."
            )
        query = (
            f"Change this story to make {team} score zero goals in the second "
            "half instead, keeping all other background factors in the story "
            "the same."
        )
        first_half = original_total - 2
        unnamed = first_half - 6
        first_half_word = {8: "eight", 9: "nine", 10: "ten"}[first_half]
        digest = hashlib.sha256(f"{story}\n{query}".encode("utf-8")).hexdigest()[:12]
        examples.append({
            "case_name": case_name,
            "example_id": f"example-{digest}",
            "source_example_id": "example-24c6a9c9ae59",
            "row_number": row_number,
            "story": story,
            "query": query,
            "expectation": (
                f"{team} scores zero second-half goals and wins {first_half}–2. "
                f"Preserve {first}, {second}, and {third} scoring two goals each "
                f"in the first 30 minutes and all {first_half} first-half goals, "
                f"including the {unnamed} additional goals whose scorers were "
                "not specified. Preserve the opponent's two goals. Remove "
                f"{first}'s minute-50 and {second}'s minute-70 scoring and "
                "hat-trick events. Do not invent scorers for the unnamed goals."
            ),
            "knowledge_reasoning": (
                f"The original {original_total} goals minus the only two "
                f"second-half goals imply {first_half} first-half goals. "
                "The three named players' two early goals each account for "
                f"six, leaving {unnamed} additional first-half goals implied "
                "by the total. The named-goal list is not exhaustive. Removing "
                "second-half goals changes neither any first-half scoring nor "
                f"the opponent's two goals, so the edited score is {first_half}–2. "
                "Rewriting the explicit two-goals-each detail to fit the total "
                "would incorrectly alter unaffected facts."
            ),
            "reference_background_fact": (
                f"In the original game, {team} scored {first_half_word} goals "
                f"in the first half: {original_total} total goals minus the "
                "only two second-half goals."
            ),
            "reference_intermediate_reasoning": (
                f"Changing {team}'s second-half total to zero removes the two "
                "second-half goals but does not remove any first-half goals, "
                "including goals whose scorers were not individually named."
            ),
            "variant_details": {
                "wording": wording,
                "team": team,
                "players": list(players),
                "original_total_goals": original_total,
                "original_second_half_goals": 2,
                "expected_first_half_goals": first_half,
                "implied_unnamed_first_half_goals": unnamed,
                "expected_final_score": f"{first_half}–2",
            },
        })
    return examples


def build_edit_prompt(example: dict, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    additions = ""
    if condition in {"nudge_2", "nudge_3"}:
        additions = f'\nBackground facts:\n{example["reference_background_fact"]}\n'
    if condition == "nudge_3":
        additions += (
            "\nIntermediate reasoning step:\n"
            f'{example["reference_intermediate_reasoning"]}\n'
        )
    return f"""Story:
{example["story"]}

Counterfactual intervention:
{example["query"]}
{additions}
{nudge.P5_INSTRUCTIONS}"""


def build_elicitation_question(example: dict) -> str:
    """Ask about the original story without supplying the reference answer."""
    team = example["variant_details"]["team"]
    return f"""Consider this story:

{example["story"]}

Based on all the statements in the story, how many of {team}'s goals occurred in the first half?

Answer concisely and show the arithmetic in one sentence."""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(nudge.common.REVISIONS),
                        default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cases", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(DEFAULT_CONDITIONS),
                        help="Default: baseline_p5 nudge_2 nudge_3. Use nudge_1 for two-turn elicitation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--output", help=(
        "JSONL path; default results/skeptical_test/<model>-thinking_skeptical_test.jsonl. "
        "A nudge_1-only run defaults to a separate *_nudge1.jsonl file. "
        "Use separate output files for concurrent processes."
    ))
    parser.add_argument("--limit", type=int, help="Run at most this many unfinished requests.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the first pending record; no model load or file writes.")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Requests per saved batch; 0 submits all pending requests.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--sampler-backend", choices=["native", "flashinfer"], default="native")
    parser.add_argument("--gdn-prefill-backend", choices=["triton", "flashinfer", "auto"],
                        default="triton")
    args = parser.parse_args(argv)
    if args.max_model_len is None:
        args.max_model_len = args.max_new_tokens + 8192
    if args.max_num_seqs is None:
        args.max_num_seqs = 16 if args.model.endswith("9B") else 6
    for field in ("max_new_tokens", "tensor_parallel_size", "max_num_seqs"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be at least 1.")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must exceed --max-new-tokens.")
    if args.batch_size < 0:
        parser.error("--batch-size must be nonnegative.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1.")
    if any(seed < 0 or seed >= 2**63 for seed in args.seeds):
        parser.error("Seeds must be integers between 0 and 2**63 - 1.")
    for field in ("cases", "conditions", "seeds"):
        values = getattr(args, field)
        if len(values) != len(set(values)):
            parser.error(f"--{field} must not contain duplicates.")
    return args


def default_output_path(model: str, conditions=None) -> Path:
    model_name = model.split("/")[-1].lower()
    suffix = "_nudge1" if conditions == ["nudge_1"] else ""
    return Path("results/skeptical_test") / f"{model_name}-thinking_skeptical_test{suffix}.jsonl"


def engine_config(args: argparse.Namespace) -> dict:
    revision = nudge.common.REVISIONS[args.model]
    return {
        "model": args.model,
        "revision": revision,
        "tokenizer_revision": revision,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "seed": 0,
        "language_model_only": True,
        "additional_config": {"gdn_prefill_backend": args.gdn_prefill_backend},
    }


def base_record(example: dict, condition: str, seed: int, args: argparse.Namespace) -> dict:
    prompt = build_edit_prompt(example, condition)
    record = {
        **example,
        "experiment": EXPERIMENT,
        "model": args.model,
        "revision": nudge.common.REVISIONS[args.model],
        "condition": condition,
        "prompt_profile": PROMPT_PROFILE,
        "seed": seed,
        "thinking_enabled": True,
        "generation_profile": "qwen3.5_thinking_general_v1",
        "inference_backend": "vllm",
        "prompt": prompt,
        "conversation": [{"role": "user", "content": prompt}],
        "provided_background_fact": (
            example["reference_background_fact"] if condition in {"nudge_2", "nudge_3"} else None
        ),
        "provided_intermediate_reasoning": (
            example["reference_intermediate_reasoning"] if condition == "nudge_3" else None
        ),
        "elicited_fact_response": None,
        "generation_config": nudge.generation_config(args, seed),
        "engine_config": engine_config(args),
    }
    if condition == "nudge_1":
        record["elicitation_question"] = build_elicitation_question(example)
        record["conversation"] = None
    return record


def elicitation_record(example: dict, seed: int, args: argparse.Namespace) -> dict:
    return {
        "experiment": EXPERIMENT,
        "record_type": "elicitation",
        "prompt_profile": PROMPT_PROFILE,
        "example_id": example["example_id"],
        "case_name": example["case_name"],
        "model": args.model,
        "revision": nudge.common.REVISIONS[args.model],
        "seed": seed,
        "question": build_elicitation_question(example),
        "generation_config": nudge.generation_config(args, seed),
        "engine_config": engine_config(args),
    }


def validate_elicitation_result(record: dict, max_tokens: int) -> None:
    result = record.get("result")
    required = {"finish_reason", "inference_backend_version"}
    if (not required.issubset(record) or not isinstance(result, dict)
            or not ELICITATION_FIELDS.keys() <= result.keys()):
        raise ValueError("Incomplete saved elicitation; inspect it before resuming.")
    count = result["generated_token_count"]
    raw = result["raw_completion"]
    if (not isinstance(raw, str) or type(count) is not int or count < 0
            or type(result["budget_exhausted"]) is not bool):
        raise ValueError("Invalid saved elicitation result.")
    exhausted = record["finish_reason"] == "length" or count >= max_tokens
    thinking, response, status = nudge.common.split_qwen_completion(raw, True, exhausted)
    if (result["budget_exhausted"] != exhausted or result["thinking"] != thinking
            or result["response"] != response or result["thinking_status"] != status):
        raise ValueError("Saved elicitation result differs from its raw completion.")


def load_elicitations(path: Path, args: argparse.Namespace) -> dict:
    """Validate and reuse completed first turns after an interrupted run."""
    examples = {example["example_id"]: example for example in build_examples()}
    saved = {}
    for record in nudge.read_records(path):
        example = examples.get(record.get("example_id"))
        seed = record.get("seed")
        if example is None or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Unrecognized saved elicitation; use a new --output path.")
        expected = elicitation_record(example, seed, args)
        changed = [field for field, value in expected.items() if record.get(field) != value]
        if changed:
            raise ValueError(f"Saved elicitation differs in {', '.join(changed)}; use a new --output path.")
        validate_elicitation_result(record, args.max_new_tokens)
        key = (example["example_id"], seed)
        if key in saved:
            raise ValueError(f"Duplicate saved elicitation: {key}")
        saved[key] = record
    return saved


def attach_elicitation(job: dict, elicitation: dict) -> dict:
    """Retain both traces, but feed back only the model's first-turn answer."""
    result = elicitation["result"]
    response = result["response"]
    return {
        **job,
        **{field: result[key] for key, field in ELICITATION_FIELDS.items()},
        "elicitation_generation_config": elicitation["generation_config"],
        "elicitation_finish_reason": elicitation["finish_reason"],
        "elicitation_inference_backend_version": elicitation["inference_backend_version"],
        "conversation": [
            {"role": "user", "content": job["elicitation_question"]},
            {"role": "assistant", "content": response},
            {"role": "user", "content": job["prompt"]},
        ] if response else None,
    }


def completed_keys(records: list[dict], args: argparse.Namespace) -> set[tuple]:
    """Validate saved inputs and settings before resuming an output file."""
    examples = {example["example_id"]: example for example in build_examples()}
    done = set()
    for row, record in enumerate(records, start=1):
        example = examples.get(record.get("example_id"))
        condition = record.get("condition")
        seed = record.get("seed")
        if (example is None or condition not in CONDITIONS
                or type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError(f"Unrecognized record {row}; use a new --output path.")
        expected = base_record(example, condition, seed, args)
        if condition == "nudge_1":
            fields = set(ELICITATION_FIELDS.values()) | {
                "elicitation_finish_reason", "elicitation_inference_backend_version",
                "elicitation_generation_config",
            }
            if not fields.issubset(record):
                raise ValueError(f"Incomplete saved elicitation at record {row}.")
            elicitation = {
                **elicitation_record(example, seed, args),
                "result": {key: record[field] for key, field in ELICITATION_FIELDS.items()},
                "finish_reason": record["elicitation_finish_reason"],
                "inference_backend_version": record["elicitation_inference_backend_version"],
            }
            validate_elicitation_result(elicitation, args.max_new_tokens)
            expected = attach_elicitation(expected, elicitation)
        changed = [field for field, value in expected.items() if record.get(field) != value]
        if changed:
            raise ValueError(
                f"Saved record {row} differs in {', '.join(changed)}. "
                "Use a new --output path for changed inputs or settings."
            )
        required = {"raw_completion", "thinking", "thinking_status", "response",
                    "tags", "generated_token_count", "inference_backend_version"}
        if not required.issubset(record):
            raise ValueError(f"Incomplete saved record {row}; inspect it before resuming.")
        key = (record["example_id"], condition, seed)
        if key in done:
            raise ValueError(f"Duplicate saved request at record {row}: {key}")
        done.add(key)
    return done


def main(argv=None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output) if args.output else default_output_path(args.model, args.conditions)
    existing = list(nudge.read_records(output_path))
    done = completed_keys(existing, args)
    examples = {example["case_name"]: example for example in build_examples()}
    jobs = [
        base_record(examples[case], condition, seed, args)
        for case in args.cases
        for condition in args.conditions
        for seed in args.seeds
        if (examples[case]["example_id"], condition, seed) not in done
    ]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print("Model:", args.model, "| thinking enabled")
    print("Cases:", ", ".join(args.cases))
    print("Conditions:", ", ".join(args.conditions))
    print("Seeds:", args.seeds)
    print("Pending requests:", len(jobs))
    print("Results:", output_path)
    if not jobs:
        print("No unfinished requests remain.")
        return 0
    nudge_1_jobs = [job for job in jobs if job["condition"] == "nudge_1"]
    cache_path = output_path.with_suffix(".elicitations.jsonl")
    elicitations = load_elicitations(cache_path, args) if nudge_1_jobs else {}
    missing_elicitations = [
        job for job in nudge_1_jobs if (job["example_id"], job["seed"]) not in elicitations
    ]
    if nudge_1_jobs:
        print("Pending factual questions:", len(missing_elicitations))
        print("Elicitation checkpoint:", cache_path)
    if args.dry_run:
        print("\nFirst pending record (reference annotations are not sent to the model):")
        print(json.dumps(jobs[0], ensure_ascii=False, indent=2))
        if nudge_1_jobs:
            print("\nnudge_1 first asks elicitation_question, then continues with the model's")
            print("own final answer as an assistant turn, followed by the editing prompt.")
        print("\nDry run only; no inference or file writes.")
        return 0

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = (
        "1" if args.sampler_backend == "flashinfer" else "0"
    )
    LLM, SamplingParams, vllm_version = nudge.common.load_vllm()
    saved_versions = [record["inference_backend_version"] for record in existing]
    saved_versions.extend(record["elicitation_inference_backend_version"]
                          for record in existing if record["condition"] == "nudge_1")
    saved_versions.extend(record["inference_backend_version"] for record in elicitations.values())
    if any(saved != vllm_version for saved in saved_versions):
        raise ValueError("Saved results use a different vLLM version; use a new --output path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Loading vLLM model...")
    llm = LLM(**engine_config(args))
    for index, request_output in nudge.run_chat_batches(
        llm=llm,
        SamplingParams=SamplingParams,
        conversations=[[{"role": "user", "content": job["elicitation_question"]}]
                       for job in missing_elicitations],
        seeds=[job["seed"] for job in missing_elicitations],
        args=args,
    ):
        job = missing_elicitations[index]
        record = {
            **elicitation_record(job, job["seed"], args),
            "result": nudge.unpack_output(request_output, args.max_new_tokens),
            "finish_reason": request_output.outputs[0].finish_reason,
            "inference_backend_version": vllm_version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        nudge.append_record(cache_path, record)
        elicitations[(job["example_id"], job["seed"])] = record

    runnable_jobs = []
    for job in jobs:
        if job["condition"] == "nudge_1":
            job = attach_elicitation(job, elicitations[(job["example_id"], job["seed"])])
            if not job["elicited_fact_response"]:
                tags = ["elicitation_no_completed_response"]
                if job["elicitation_budget_exhausted"]:
                    tags.append("elicitation_budget_exhausted")
                nudge.append_record(output_path, {
                    **job,
                    "raw_completion": None,
                    "thinking": None,
                    "thinking_status": "not_run",
                    "response": None,
                    "generated_token_count": 0,
                    "budget_exhausted": False,
                    "finish_reason": None,
                    "tags": tags,
                    "inference_backend_version": vllm_version,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                print(f"{job['case_name']} | seed {job['seed']}: "
                      "no completed elicitation answer; edit skipped and saved.")
                continue
        runnable_jobs.append(job)

    print("\nGenerating edited stories...")
    for index, request_output in nudge.run_chat_batches(
        llm=llm,
        SamplingParams=SamplingParams,
        conversations=[job["conversation"] for job in runnable_jobs],
        seeds=[job["seed"] for job in runnable_jobs],
        args=args,
    ):
        result = nudge.unpack_output(request_output, args.max_new_tokens)
        tags = []
        if result["budget_exhausted"]:
            tags.append("budget_exhausted")
        if result["thinking_status"] == "incomplete_thinking_trace":
            tags.append("incomplete_thinking_trace")
        if runnable_jobs[index].get("elicitation_budget_exhausted"):
            tags.append("elicitation_budget_exhausted")
        record = {
            **runnable_jobs[index],
            **result,
            "tags": tags,
            "finish_reason": request_output.outputs[0].finish_reason,
            "inference_backend_version": vllm_version,
            "submission_batch_size": args.batch_size or len(runnable_jobs),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        nudge.append_record(output_path, record)
        print(f"[{index + 1}/{len(runnable_jobs)}] {record['case_name']} | "
              f"{record['condition']} | seed {record['seed']}")
        print(record["response"] or "[No completed final response]")
        if tags:
            print("Tags:", ", ".join(tags))
    print("\nFinished. Results saved to:", output_path)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())

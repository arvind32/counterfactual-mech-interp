#!/usr/bin/env python3
"""Run unaltered continuations from configured A* and A** trace states."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent
PROJECT_ROOT = TRACE_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "trace_branching"

# The validated base runner owns model configuration, token normalization,
# sampling parameters, and presence-history restoration.
sys.path.insert(0, str(TRACE_ROOT))
import run_ab as base_runner  # noqa: E402


EXPERIMENT_NAME = "continuations_2_control_rollouts"
RESULT_BASENAME = "continuations_2_controls"
LOCATIONS = ("A_STAR", "A_STAR_STAR")
ROLLOUTS_PER_CELL = 30
SAMPLE_SEED_START = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 30 unaltered continuations from A* and A** for configured "
            "baseline traces."
        )
    )
    parser.add_argument("--sources", type=Path, default=ROOT / "source_records.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSONL path. The default is under results/trace_branching and "
            "uses a shard-specific filename when --num-shards is greater than 1."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-ids", nargs="*")
    parser.add_argument("--locations", nargs="*", choices=list(LOCATIONS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def hash_ints(values: list[int]) -> str:
    payload = ",".join(str(value) for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def default_output_path(num_shards: int, shard_index: int) -> Path:
    if num_shards == 1:
        return RESULTS_ROOT / f"{RESULT_BASENAME}.jsonl"
    return RESULTS_ROOT / (
        f"{RESULT_BASENAME}.part-{shard_index:02d}-of-{num_shards:02d}.jsonl"
    )


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be between 0 and --num-shards - 1")


def prefix_through_line_number(
    raw_completion: str, line_number: int, expected_text: str
) -> str:
    lines = raw_completion.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise RuntimeError(f"cut line {line_number} is outside the completion")
    actual = lines[line_number - 1].rstrip("\r\n")
    if actual != expected_text:
        raise RuntimeError(
            f"cut-line mismatch at line {line_number}: "
            f"expected {expected_text!r}, found {actual!r}"
        )
    prefix = "".join(lines[:line_number])
    if not prefix.endswith("\n"):
        raise RuntimeError(f"cut line {line_number} is not newline terminated")
    return prefix


def validate_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        raise RuntimeError("at least one source is required")
    ids: list[str] = []
    family_ranks: list[tuple[str, int]] = []
    for source_row in sources:
        selected = source_row.get("selection", {})
        record = source_row.get("record", {})
        source_id = selected.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("every source must have a nonempty source_id")
        if selected.get("role") != "failed_source":
            raise RuntimeError(f"{source_id} is not labeled as a failed source")
        if selected.get("condition") != "baseline_p5":
            raise RuntimeError(f"{source_id} is not a baseline_p5 source")
        if not isinstance(selected.get("story_key"), str):
            raise RuntimeError(f"missing story key for {source_id}")
        if not isinstance(selected.get("source_rank"), int):
            raise RuntimeError(f"missing source rank for {source_id}")
        if selected.get("case_name") != record.get("case_name"):
            raise RuntimeError(f"case mismatch for {source_id}")
        if selected.get("condition") != record.get("condition"):
            raise RuntimeError(f"condition mismatch for {source_id}")
        if str(selected.get("seed")) != str(record.get("seed")):
            raise RuntimeError(f"seed mismatch for {source_id}")
        raw_completion = record.get("raw_completion", "")
        if not isinstance(raw_completion, str) or not raw_completion:
            raise RuntimeError(f"missing raw completion for {source_id}")
        previous = 0
        for location in LOCATIONS:
            anchor = selected.get("anchors", {}).get(location, {})
            line_number = anchor.get("raw_completion_line_1_based")
            text = anchor.get("text")
            if not isinstance(line_number, int) or line_number <= previous:
                raise RuntimeError(f"invalid {location} ordering for {source_id}")
            if not isinstance(text, str):
                raise RuntimeError(f"missing {location} text for {source_id}")
            prefix_through_line_number(raw_completion, line_number, text)
            previous = line_number
        ids.append(source_id)
        family_ranks.append((selected["story_key"], selected.get("source_rank")))
    if len(set(ids)) != len(ids):
        raise RuntimeError("source IDs are not unique")
    if len(set(family_ranks)) != len(family_ranks):
        raise RuntimeError(f"duplicate family/rank labels: {family_ranks}")


def build_jobs(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_sources(sources)
    jobs: list[dict[str, Any]] = []
    for source_index, source_row in enumerate(sources):
        selected = source_row["selection"]
        record = source_row["record"]
        for location in LOCATIONS:
            anchor = selected["anchors"][location]
            raw_prefix = prefix_through_line_number(
                record["raw_completion"],
                anchor["raw_completion_line_1_based"],
                anchor["text"],
            )
            for rollout_index in range(ROLLOUTS_PER_CELL):
                sample_seed = SAMPLE_SEED_START + rollout_index
                jobs.append(
                    {
                        "job_id": (
                            f"{selected['source_id']}__{location}__"
                            f"r{rollout_index:02d}"
                        ),
                        "experiment": EXPERIMENT_NAME,
                        "intervention_type": "unaltered_control",
                        "source_index": source_index,
                        "source_id": selected["source_id"],
                        "source_label": selected["source_label"],
                        "source_rank": selected["source_rank"],
                        "story_family": selected["domain"],
                        "story_key": selected["story_key"],
                        "case_name": selected["case_name"],
                        "source_condition": selected["condition"],
                        "source_seed": selected["seed"],
                        "fact_origin": selected["fact_origin"],
                        "location": location,
                        "cut_line_1_based": anchor[
                            "raw_completion_line_1_based"
                        ],
                        "cut_line_text": anchor["text"],
                        "rollout_index": rollout_index,
                        "sample_seed": sample_seed,
                        "raw_prefix": raw_prefix,
                    }
                )
    return jobs


def validate_design(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        raise RuntimeError("rollout manifest is empty")
    job_ids = [job["job_id"] for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError("manifest contains duplicate job IDs")
    cells = Counter((job["source_id"], job["location"]) for job in jobs)
    source_ids = {job["source_id"] for job in jobs}
    expected_cells = len(source_ids) * len(LOCATIONS)
    expected_jobs = expected_cells * ROLLOUTS_PER_CELL
    if len(jobs) != expected_jobs:
        raise RuntimeError(f"expected {expected_jobs} jobs, found {len(jobs)}")
    if len(cells) != expected_cells or set(cells.values()) != {ROLLOUTS_PER_CELL}:
        raise RuntimeError(f"unexpected experimental cell counts: {dict(cells)}")
    if {job["sample_seed"] for job in jobs} != set(
        range(SAMPLE_SEED_START, SAMPLE_SEED_START + ROLLOUTS_PER_CELL)
    ):
        raise RuntimeError("unexpected sampling seed range")


def prepare_jobs(
    jobs: list[dict[str, Any]], sources: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for job in jobs:
        source_index = job["source_index"]
        if not isinstance(source_index, int) or not 0 <= source_index < len(sources):
            raise IndexError(f"invalid source index for {job['job_id']}")
        record = sources[source_index]["record"]
        raw_prefix = job["raw_prefix"]
        if not record["raw_completion"].startswith(raw_prefix):
            raise RuntimeError(f"raw-prefix mismatch for {job['job_id']}")
        if raw_prefix.splitlines()[-1] != job["cut_line_text"]:
            raise RuntimeError(f"cut-line mismatch for {job['job_id']}")

        chat_ids = base_runner.normalize_token_ids(
            tokenizer.apply_chat_template(
                record["conversation"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
            label="chat template",
        )
        prefix_ids = base_runner.normalize_token_ids(
            tokenizer.encode(raw_prefix, add_special_tokens=False),
            label="trace prefix",
        )
        decoded_prefix = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_prefix != raw_prefix:
            raise RuntimeError(
                f"tokenizer text round trip failed for {job['job_id']}; refusing replay"
            )
        prompt_ids = chat_ids + prefix_ids
        if not prompt_ids or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in prompt_ids
        ):
            raise TypeError(f"prompt IDs are not flat integers for {job['job_id']}")
        prepared.append(
            {
                "job": job,
                "record": record,
                "chat_ids": chat_ids,
                "prefix_ids": prefix_ids,
                "prompt_ids": prompt_ids,
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
            base_runner.ORIGINAL_MAX_NEW_TOKENS,
            base_runner.MAX_MODEL_LEN - family_max_prompt[family] - 1,
        )
        if item["max_tokens"] < 1:
            raise RuntimeError(f"no continuation budget for {item['job']['job_id']}")
    return prepared


def validate_existing_output(
    output_path: Path,
    manifest_jobs: list[dict[str, Any]],
    allowed_job_ids: set[str],
) -> set[str]:
    if not output_path.exists():
        return set()
    manifest = {job["job_id"]: job for job in manifest_jobs}
    rows = load_jsonl(output_path)
    ids = [row.get("job_id") for row in rows]
    if any(not isinstance(job_id, str) for job_id in ids):
        raise RuntimeError(f"existing output contains a row without a job ID")
    duplicates = sorted(job_id for job_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"existing output has duplicate job IDs: {duplicates[:5]}")
    unknown = sorted(set(ids) - set(manifest))
    if unknown:
        raise RuntimeError(f"existing output has unknown job IDs: {unknown[:5]}")
    misplaced = sorted(set(ids) - allowed_job_ids)
    if misplaced:
        raise RuntimeError(
            "existing output contains jobs assigned to another shard: "
            f"{misplaced[:5]}"
        )
    identity_fields = (
        "source_id",
        "source_rank",
        "case_name",
        "source_condition",
        "source_seed",
        "location",
        "cut_line_1_based",
        "cut_line_text",
        "rollout_index",
        "sample_seed",
        "intervention_type",
    )
    for row in rows:
        expected = manifest[row["job_id"]]
        mismatches = [
            field for field in identity_fields if row.get(field) != expected.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                f"existing output does not match the design for {row['job_id']}: "
                f"{mismatches}"
            )
    return set(ids)


def filter_jobs(
    jobs: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    allowed_sources = set(args.source_ids) if args.source_ids else None
    allowed_locations = set(args.locations) if args.locations else None
    if allowed_sources is not None:
        unknown_sources = sorted(
            allowed_sources - {job["source_id"] for job in jobs}
        )
        if unknown_sources:
            raise ValueError(f"unknown --source-ids: {unknown_sources}")
    selected: list[dict[str, Any]] = []
    for manifest_index, job in enumerate(jobs):
        if manifest_index % args.num_shards != args.shard_index:
            continue
        if allowed_sources is not None and job["source_id"] not in allowed_sources:
            continue
        if allowed_locations is not None and job["location"] not in allowed_locations:
            continue
        selected.append(job)
    return selected


def write_rollouts(
    prepared: list[dict[str, Any]],
    output_path: Path,
    llm: Any,
    SamplingParams: Any,
    tokenizer: Any,
    batch_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            prompts = [{"prompt_token_ids": item["prompt_ids"]} for item in batch]
            params = [
                base_runner.sampling_params(
                    SamplingParams,
                    seed=item["job"]["sample_seed"],
                    max_tokens=item["max_tokens"],
                    prefix_ids=item["prefix_ids"],
                )
                for item in batch
            ]
            print(
                f"Submitting jobs {start + 1}-{start + len(batch)} "
                f"of {len(prepared)}",
                flush=True,
            )
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
                parsed = base_runner.split_completion(full_completion, budget_exhausted)
                record = {
                    **item["job"],
                    "model": base_runner.MODEL,
                    "revision": base_runner.REVISION,
                    "sampling": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": base_runner.PRESENCE_PENALTY,
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


def main() -> int:
    args = parse_args()
    validate_shard_args(args.num_shards, args.shard_index)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if not args.sources.is_file():
        raise FileNotFoundError(
            f"missing {args.sources}; run "
            "trace_branching_pilot/continuations_2/prepare_sources.py first"
        )
    output_path = args.output or default_output_path(args.num_shards, args.shard_index)

    sources = load_jsonl(args.sources)
    all_jobs = build_jobs(sources)
    validate_design(all_jobs)
    shard_job_ids = {
        job["job_id"]
        for manifest_index, job in enumerate(all_jobs)
        if manifest_index % args.num_shards == args.shard_index
    }
    selected_jobs = filter_jobs(all_jobs, args)
    completed = validate_existing_output(output_path, all_jobs, shard_job_ids)
    selected_jobs = [job for job in selected_jobs if job["job_id"] not in completed]
    if args.limit is not None:
        selected_jobs = selected_jobs[: args.limit]
    if not selected_jobs:
        print(f"No unfinished jobs selected for {output_path}")
        return 0

    llm, SamplingParams, _ = base_runner.build_engine(args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    base_runner.runtime_smoke(llm, tokenizer, SamplingParams)

    # Prepare the complete design so resumed or filtered jobs retain the same
    # family-level continuation budgets.
    prepared_all = prepare_jobs(all_jobs, sources, tokenizer)
    selected_ids = {job["job_id"] for job in selected_jobs}
    prepared = [
        item for item in prepared_all if item["job"]["job_id"] in selected_ids
    ]
    write_rollouts(
        prepared,
        output_path,
        llm,
        SamplingParams,
        tokenizer,
        args.batch_size,
    )
    print(f"Completed {len(prepared)} jobs; results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

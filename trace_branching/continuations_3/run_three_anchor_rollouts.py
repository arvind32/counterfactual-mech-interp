#!/usr/bin/env python3
"""Run natural continuations immediately before and after three trace anchors."""

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

# The validated base runner fixes the model revision, sampling configuration,
# tokenizer normalization, and presence-penalty history restoration.
sys.path.insert(0, str(TRACE_ROOT))
import run_ab as base_runner  # noqa: E402


EXPERIMENT_NAME = "three_anchor_before_after"
RESULT_BASENAME = "three_anchor_rollouts"
ANCHORS = ("A", "A_STAR", "C")
BRANCHES = ("BEFORE", "AFTER")
ROLLOUTS_PER_CELL = 30
SAMPLE_SEEDS = tuple(range(3000, 3030))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 30 matched-seed continuations immediately before and after "
            "A, A*, and C in six baseline traces."
        )
    )
    parser.add_argument("--sources", type=Path, default=ROOT / "source_records.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSONL path. By default, results are written under "
            "results/trace_branching with a shard-specific filename."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-ids", nargs="*")
    parser.add_argument("--anchors", nargs="*", choices=list(ANCHORS))
    parser.add_argument("--branches", nargs="*", choices=list(BRANCHES))
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


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def validate_sources(sources: list[dict[str, Any]]) -> None:
    if len(sources) != 6:
        raise RuntimeError(f"expected six source traces, found {len(sources)}")
    ids: list[str] = []
    family_ranks: list[tuple[str, int]] = []
    results: set[str] = set()
    for source_row in sources:
        selected = source_row.get("selection", {})
        record = source_row.get("record", {})
        source_id = selected.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("every source must have a nonempty source_id")
        if selected.get("role") != "anchor_source":
            raise RuntimeError(f"{source_id} is not labeled as an anchor source")
        if selected.get("condition") != "baseline_p5":
            raise RuntimeError(f"{source_id} is not a baseline_p5 source")
        if selected.get("case_name") != record.get("case_name"):
            raise RuntimeError(f"case mismatch for {source_id}")
        if selected.get("condition") != record.get("condition"):
            raise RuntimeError(f"condition mismatch for {source_id}")
        if str(selected.get("seed")) != str(record.get("seed")):
            raise RuntimeError(f"seed mismatch for {source_id}")
        if selected.get("original_result") not in {"PASS", "FAIL"}:
            raise RuntimeError(f"invalid original result for {source_id}")
        if not isinstance(selected.get("c_value"), str):
            raise RuntimeError(f"missing C value for {source_id}")
        raw_completion = record.get("raw_completion")
        if not isinstance(raw_completion, str) or not raw_completion:
            raise RuntimeError(f"missing raw completion for {source_id}")

        anchors = selected.get("anchors", {})
        if set(anchors) != set(ANCHORS):
            raise RuntimeError(f"incorrect anchor set for {source_id}")
        intervals: list[tuple[int, int, str]] = []
        for anchor_name in ANCHORS:
            anchor = anchors[anchor_name]
            required = (
                "start_char_0_based",
                "end_char_0_based_exclusive",
                "after_char_0_based_exclusive",
                "anchor_text",
                "inserted_text",
                "before_prefix_char_count",
                "after_prefix_char_count",
            )
            missing = [field for field in required if field not in anchor]
            if missing:
                raise RuntimeError(
                    f"unresolved {anchor_name} for {source_id}: missing {missing}"
                )
            start = anchor["start_char_0_based"]
            end = anchor["end_char_0_based_exclusive"]
            after = anchor["after_char_0_based_exclusive"]
            if not all(isinstance(value, int) for value in (start, end, after)):
                raise RuntimeError(f"invalid character offsets for {source_id}/{anchor_name}")
            if not 0 < start < end <= after <= len(raw_completion):
                raise RuntimeError(f"invalid anchor interval for {source_id}/{anchor_name}")
            before_prefix = raw_completion[:start]
            after_prefix = raw_completion[:after]
            if len(before_prefix) != anchor["before_prefix_char_count"]:
                raise RuntimeError(f"before-prefix length mismatch for {source_id}/{anchor_name}")
            if len(after_prefix) != anchor["after_prefix_char_count"]:
                raise RuntimeError(f"after-prefix length mismatch for {source_id}/{anchor_name}")
            if raw_completion[start:end] != anchor["anchor_text"]:
                raise RuntimeError(f"anchor-text mismatch for {source_id}/{anchor_name}")
            if after_prefix[len(before_prefix) :] != anchor["inserted_text"]:
                raise RuntimeError(f"inserted-text mismatch for {source_id}/{anchor_name}")
            intervals.append((start, end, anchor_name))

        actual_order = [name for _, _, name in sorted(intervals)]
        if actual_order != selected.get("anchor_order"):
            raise RuntimeError(f"anchor-order mismatch for {source_id}")
        ids.append(source_id)
        family_ranks.append((selected.get("story_key"), selected.get("source_rank")))
        results.add(selected["original_result"])

    if len(set(ids)) != len(ids):
        raise RuntimeError("source IDs are not unique")
    if len(set(family_ranks)) != len(family_ranks):
        raise RuntimeError(f"duplicate family/rank labels: {family_ranks}")
    if {source["selection"]["story_key"] for source in sources} != {
        "madrid",
        "starry_night",
    }:
        raise RuntimeError("expected Madrid and Starry Night sources")
    if results != {"PASS", "FAIL"}:
        raise RuntimeError("expected both successful and failed source traces")


def build_jobs(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_sources(sources)
    jobs: list[dict[str, Any]] = []
    pair_index = 0
    for source_index, source_row in enumerate(sources):
        selected = source_row["selection"]
        record = source_row["record"]
        raw_completion = record["raw_completion"]
        c_position = selected["anchor_order"].index("C")
        for anchor_name in ANCHORS:
            anchor = selected["anchors"][anchor_name]
            anchor_position = selected["anchor_order"].index(anchor_name)
            relative_to_c = (
                "SELF"
                if anchor_name == "C"
                else ("BEFORE_C" if anchor_position < c_position else "AFTER_C")
            )
            prefixes = {
                "BEFORE": raw_completion[: anchor["before_prefix_char_count"]],
                "AFTER": raw_completion[: anchor["after_prefix_char_count"]],
            }
            for rollout_index, sample_seed in enumerate(SAMPLE_SEEDS):
                pair_id = (
                    f"{selected['source_id']}__{anchor_name}__r{rollout_index:02d}"
                )
                for branch in BRANCHES:
                    raw_prefix = prefixes[branch]
                    jobs.append(
                        {
                            "job_id": f"{pair_id}__{branch}",
                            "pair_id": pair_id,
                            "pair_index": pair_index,
                            "experiment": EXPERIMENT_NAME,
                            "intervention_type": "natural_trace_before_after",
                            "source_index": source_index,
                            "source_id": selected["source_id"],
                            "source_label": selected["source_label"],
                            "source_rank": selected["source_rank"],
                            "story_family": selected["domain"],
                            "story_key": selected["story_key"],
                            "case_name": selected["case_name"],
                            "source_condition": selected["condition"],
                            "source_seed": selected["seed"],
                            "source_original_result": selected["original_result"],
                            "fact_origin": selected["fact_origin"],
                            "anchor_order": selected["anchor_order"],
                            "anchor": anchor_name,
                            "anchor_relative_to_c": relative_to_c,
                            "anchor_start_line_1_based": anchor[
                                "start_line_1_based"
                            ],
                            "anchor_end_line_1_based": anchor["end_line_1_based"],
                            "anchor_start_char_0_based": anchor[
                                "start_char_0_based"
                            ],
                            "anchor_end_char_0_based_exclusive": anchor[
                                "end_char_0_based_exclusive"
                            ],
                            "anchor_text": anchor["anchor_text"],
                            "inserted_text": anchor["inserted_text"],
                            "c_value": selected["c_value"],
                            "c_is_correct": selected["c_value"].startswith("correct_"),
                            "branch": branch,
                            "cut_char_count": len(raw_prefix),
                            "rollout_index": rollout_index,
                            "sample_seed": sample_seed,
                            "raw_prefix": raw_prefix,
                        }
                    )
                pair_index += 1
    return jobs


def validate_design(jobs: list[dict[str, Any]]) -> None:
    expected_sources = 6
    expected_jobs = (
        expected_sources
        * len(ANCHORS)
        * len(BRANCHES)
        * ROLLOUTS_PER_CELL
    )
    if len(jobs) != expected_jobs:
        raise RuntimeError(f"expected {expected_jobs} jobs, found {len(jobs)}")
    job_ids = [job["job_id"] for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError("manifest contains duplicate job IDs")

    cells = Counter(
        (job["source_id"], job["anchor"], job["branch"]) for job in jobs
    )
    expected_cells = expected_sources * len(ANCHORS) * len(BRANCHES)
    if len(cells) != expected_cells or set(cells.values()) != {ROLLOUTS_PER_CELL}:
        raise RuntimeError(f"unexpected experimental cell counts: {dict(cells)}")

    pairs: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        pairs.setdefault(job["pair_id"], []).append(job)
    expected_pairs = expected_sources * len(ANCHORS) * ROLLOUTS_PER_CELL
    if len(pairs) != expected_pairs:
        raise RuntimeError(f"expected {expected_pairs} pairs, found {len(pairs)}")
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {job["branch"] for job in pair} != set(BRANCHES):
            raise RuntimeError(f"incomplete before/after pair: {pair_id}")
        if len({job["sample_seed"] for job in pair}) != 1:
            raise RuntimeError(f"seed mismatch within pair: {pair_id}")
        if len({job["pair_index"] for job in pair}) != 1:
            raise RuntimeError(f"pair-index mismatch: {pair_id}")
        before = next(job for job in pair if job["branch"] == "BEFORE")
        after = next(job for job in pair if job["branch"] == "AFTER")
        if not after["raw_prefix"].startswith(before["raw_prefix"]):
            raise RuntimeError(f"after prefix does not extend before prefix: {pair_id}")
        if after["raw_prefix"][len(before["raw_prefix"]) :] != before[
            "inserted_text"
        ]:
            raise RuntimeError(f"pair does not differ by the anchor span: {pair_id}")
    if {job["sample_seed"] for job in jobs} != set(SAMPLE_SEEDS):
        raise RuntimeError("unexpected sampling seeds")


def prepare_jobs(
    jobs: list[dict[str, Any]], sources: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    """Tokenize each unique prefix once and assign equal family-level budgets."""
    state_cache: dict[tuple[int, str, str], dict[str, Any]] = {}
    for job in jobs:
        key = (job["source_index"], job["anchor"], job["branch"])
        if key in state_cache:
            continue
        source_index = job["source_index"]
        if not isinstance(source_index, int) or not 0 <= source_index < len(sources):
            raise IndexError(f"invalid source index for {job['job_id']}")
        record = sources[source_index]["record"]
        raw_prefix = job["raw_prefix"]
        if record["raw_completion"][: len(raw_prefix)] != raw_prefix:
            raise RuntimeError(f"raw-prefix mismatch for {job['job_id']}")

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
        state_cache[key] = {
            "record": record,
            "chat_ids": chat_ids,
            "prefix_ids": prefix_ids,
            "prompt_ids": prompt_ids,
        }

    family_max_prompt: dict[str, int] = {}
    for job in jobs:
        state = state_cache[(job["source_index"], job["anchor"], job["branch"])]
        family = job["story_family"]
        family_max_prompt[family] = max(
            family_max_prompt.get(family, 0), len(state["prompt_ids"])
        )

    prepared: list[dict[str, Any]] = []
    for job in jobs:
        state = state_cache[(job["source_index"], job["anchor"], job["branch"])]
        max_tokens = min(
            base_runner.ORIGINAL_MAX_NEW_TOKENS,
            base_runner.MAX_MODEL_LEN - family_max_prompt[job["story_family"]] - 1,
        )
        if max_tokens < 1:
            raise RuntimeError(f"no continuation budget for {job['job_id']}")
        prepared.append({"job": job, **state, "max_tokens": max_tokens})
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
        raise RuntimeError("existing output contains a row without a job ID")
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
        "pair_id",
        "pair_index",
        "source_id",
        "source_rank",
        "case_name",
        "source_condition",
        "source_seed",
        "source_original_result",
        "anchor_order",
        "anchor",
        "anchor_relative_to_c",
        "anchor_start_line_1_based",
        "anchor_end_line_1_based",
        "anchor_start_char_0_based",
        "anchor_end_char_0_based_exclusive",
        "anchor_text",
        "inserted_text",
        "c_value",
        "c_is_correct",
        "branch",
        "cut_char_count",
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
    allowed_anchors = set(args.anchors) if args.anchors else None
    allowed_branches = set(args.branches) if args.branches else None
    if allowed_sources is not None:
        unknown_sources = sorted(
            allowed_sources - {job["source_id"] for job in jobs}
        )
        if unknown_sources:
            raise ValueError(f"unknown --source-ids: {unknown_sources}")
    selected: list[dict[str, Any]] = []
    for job in jobs:
        # A matched before/after pair always belongs to the same shard.
        if job["pair_index"] % args.num_shards != args.shard_index:
            continue
        if allowed_sources is not None and job["source_id"] not in allowed_sources:
            continue
        if allowed_anchors is not None and job["anchor"] not in allowed_anchors:
            continue
        if allowed_branches is not None and job["branch"] not in allowed_branches:
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
                    "prefix_text_hash": hash_text(item["job"]["raw_prefix"]),
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
    if args.output is not None and args.num_shards > 1:
        raise ValueError(
            "omit --output for sharded runs so each shard receives a distinct file"
        )
    if not args.sources.is_file():
        raise FileNotFoundError(
            f"missing {args.sources}; run "
            "trace_branching_pilot/continuation_3/prepare_sources.py first"
        )
    output_path = args.output or default_output_path(args.num_shards, args.shard_index)

    sources = load_jsonl(args.sources)
    all_jobs = build_jobs(sources)
    validate_design(all_jobs)
    shard_job_ids = {
        job["job_id"]
        for job in all_jobs
        if job["pair_index"] % args.num_shards == args.shard_index
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

    # Prepare the full design so resumed or filtered jobs retain identical
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
    print(f"Completed {len(prepared)} new jobs: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run premise-acceptance continuations from six pinned A/B trace cuts."""

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
# sampling parameters, and presence-penalty history restoration.
sys.path.insert(0, str(TRACE_ROOT))
import run_ab as base_runner  # noqa: E402


EXPERIMENT_NAME = "premise_acceptance_continuations"
RESULT_BASENAME = "premise_acceptance_rollouts"
EXPECTED_SOURCES = 6
EXPECTED_BASE_JOBS = 120
EXPECTED_INTERVENTION_JOBS = 240
EXPECTED_CONTROL_JOBS = 120
EXPECTED_JOBS = 360
ROLLOUTS_PER_CELL = 10

# Each sentence is inserted after both cuts. It restates or assigns authority to
# a fact already present in the corresponding trace prefix; it does not state
# the required downstream edit.
CANDIDATES: dict[str, tuple[dict[str, str], ...]] = {
    "madrid": (
        {
            "candidate_id": "M1",
            "text_template": (
                "This background fact that {club} scored nine goals in the "
                "first half is an authoritative constraint on the upstream world "
                "provided by the task system."
            ),
        },
        {
            "candidate_id": "M2",
            "text_template": (
                "Okay, I will treat the ‘9 goals in first half’ as an immutable "
                "upstream state."
            ),
        },
    ),
    "salah": (
        {
            "candidate_id": "S1",
            "text_template": (
                "Since teams switching ends at half-time is an upstream fact "
                "about this story, this fact must be preserved."
            ),
        },
        {
            "candidate_id": "S2",
            "text_template": (
                "This is a key upstream constraint: ‘Teams normally switch ends "
                "at half-time.’"
            ),
        },
    ),
    "starry_night": (
        {
            "candidate_id": "SN1",
            "text_template": (
                "I must respect the truth that Starry Night is at MoMA, not the "
                "Met; it is an upstream constraint."
            ),
        },
        {
            "candidate_id": "SN2",
            "text_template": (
                "The fact that Starry Night is at MoMA, not the Met constrains "
                "how I interpret the story; it is an upstream background fact."
            ),
        },
    ),
}

CASE_TO_STORY_KEY = {
    "a_names": "madrid",
    "madrid": "madrid",
    "salah": "salah",
    "starry_night": "starry_night",
}

CASE_TO_CLUB = {
    "a_names": "Harbor FC",
    "madrid": "Real Madrid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 120 unaltered controls and 240 premise-acceptance branches "
            "from six pinned failed traces."
        )
    )
    parser.add_argument(
        "--sources", type=Path, default=ROOT / "source_records.jsonl"
    )
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
    parser.add_argument("--cases", nargs="*", choices=sorted(CASE_TO_STORY_KEY))
    parser.add_argument("--cuts", nargs="*", choices=["A", "B"])
    parser.add_argument(
        "--candidate-ids",
        nargs="*",
        choices=[
            candidate["candidate_id"]
            for candidates in CANDIDATES.values()
            for candidate in candidates
        ]
        + ["CONTROL"],
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and reject malformed non-object rows."""
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


def append_candidate(base_prefix: str, line_prefix: str, text: str) -> tuple[str, str]:
    """Append one complete reasoning line to a prefix ending at a line boundary."""
    if not base_prefix.endswith("\n"):
        raise RuntimeError("base trace prefix does not end at a line boundary")
    if "\n" in text or not text.strip().endswith((".", "?", "!", "’", '"')):
        raise ValueError(f"candidate must be one nonempty line: {text!r}")
    insertion_line = f"{line_prefix}{text}\n"
    return base_prefix + insertion_line, insertion_line


def prefix_through_line(raw_completion: str, target: str) -> str:
    """Return the exact raw prefix through one uniquely matched complete line."""
    lines = raw_completion.splitlines(keepends=True)
    matches = [
        index for index, line in enumerate(lines) if line.rstrip("\r\n") == target
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one cut-line match, found {len(matches)} for {target!r}"
        )
    prefix = "".join(lines[: matches[0] + 1])
    if not prefix.endswith("\n"):
        raise RuntimeError(f"cut line is not newline terminated: {target!r}")
    return prefix


def validate_sources(sources: list[dict[str, Any]]) -> None:
    """Validate source identities, ranks, and exact line-numbered cut anchors."""
    if len(sources) != EXPECTED_SOURCES:
        raise RuntimeError(f"expected {EXPECTED_SOURCES} sources, found {len(sources)}")
    ids: list[str] = []
    story_ranks: list[tuple[str, int]] = []
    for source_row in sources:
        selection = source_row.get("selection", {})
        source = source_row.get("record", {})
        if selection.get("role") != "failed_source":
            raise RuntimeError("every source must be a failed source trace")
        source_id = selection.get("source_id")
        story_key = selection.get("story_key")
        source_rank = selection.get("source_rank")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("every source must have a nonempty source_id")
        if story_key not in CANDIDATES:
            raise RuntimeError(f"unexpected story key: {story_key!r}")
        if source_rank not in (1, 2):
            raise RuntimeError(f"unexpected source rank: {source_rank!r}")
        if selection.get("case_name") != source.get("case_name"):
            raise RuntimeError(f"case mismatch for {source_id}")
        if selection.get("condition") != source.get("condition"):
            raise RuntimeError(f"condition mismatch for {source_id}")
        if selection.get("seed") != source.get("seed"):
            raise RuntimeError(f"seed mismatch for {source_id}")
        raw_lines = source.get("raw_completion", "").splitlines()
        for cut in ("A", "B"):
            anchor = selection.get("anchors", {}).get(cut, {})
            line_number = anchor.get("raw_completion_line_1_based")
            if not isinstance(line_number, int) or not 1 <= line_number <= len(raw_lines):
                raise RuntimeError(f"invalid {cut} line number for {source_id}")
            if raw_lines[line_number - 1] != anchor.get("text"):
                raise RuntimeError(f"{cut} anchor text/line mismatch for {source_id}")
            prefix_through_line(source["raw_completion"], anchor["text"])
        if (
            selection["anchors"]["A"]["raw_completion_line_1_based"]
            >= selection["anchors"]["B"]["raw_completion_line_1_based"]
        ):
            raise RuntimeError(f"A must precede B for {source_id}")
        ids.append(source_id)
        story_ranks.append((story_key, source_rank))
    if len(set(ids)) != len(ids):
        raise RuntimeError("source IDs are not unique")
    expected_story_ranks = {
        (story_key, source_rank)
        for story_key in CANDIDATES
        for source_rank in (1, 2)
    }
    if set(story_ranks) != expected_story_ranks:
        raise RuntimeError(f"unexpected story/rank coverage: {story_ranks}")


def build_base_jobs(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ten unaltered rollout specifications at every source and cut."""
    validate_sources(sources)
    jobs: list[dict[str, Any]] = []
    for source_index, source_row in enumerate(sources):
        selection = source_row["selection"]
        source = source_row["record"]
        for cut in ("A", "B"):
            anchor = selection["anchors"][cut]
            raw_prefix = prefix_through_line(source["raw_completion"], anchor["text"])
            for rollout_index in range(ROLLOUTS_PER_CELL):
                jobs.append(
                    {
                        "job_id": (
                            f"{selection['source_id']}__{cut}__r{rollout_index}"
                        ),
                        "source_index": source_index,
                        "source_id": selection["source_id"],
                        "source_rank": selection["source_rank"],
                        "story_family": selection["domain"],
                        "story_key": selection["story_key"],
                        "case_name": selection["case_name"],
                        "source_condition": selection["condition"],
                        "source_seed": selection["seed"],
                        "fact_origin": selection["fact_origin"],
                        "cut": cut,
                        "cut_line_1_based": anchor["raw_completion_line_1_based"],
                        "cut_line_text": anchor["text"],
                        "insertion_prefix": anchor["insertion_prefix"],
                        "rollout_index": rollout_index,
                        "sample_seed": 1000 + rollout_index,
                        "raw_prefix": raw_prefix,
                    }
                )
    return jobs


def candidate_text(base_job: dict[str, Any], template: str) -> str:
    values = {"club": CASE_TO_CLUB.get(base_job["case_name"], "")}
    return template.format(**values)


def build_jobs(base_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add one control and two acceptance interventions to every base job."""
    jobs: list[dict[str, Any]] = []
    for base_job in base_jobs:
        story_key = base_job["story_key"]
        jobs.append(
            {
                **base_job,
                "job_id": f"{base_job['job_id']}__CONTROL",
                "base_job_id": base_job["job_id"],
                "experiment": EXPERIMENT_NAME,
                "intervention_type": "unaltered_control",
                "candidate_id": "CONTROL",
                "candidate_text": None,
                "candidate_provenance": None,
                "base_raw_prefix": base_job["raw_prefix"],
                "insertion_line": "",
            }
        )
        for candidate in CANDIDATES[story_key]:
            text = candidate_text(base_job, candidate["text_template"])
            injected_prefix, insertion_line = append_candidate(
                base_job["raw_prefix"], base_job["insertion_prefix"], text
            )
            jobs.append(
                {
                    **base_job,
                    "job_id": f"{base_job['job_id']}__{candidate['candidate_id']}",
                    "base_job_id": base_job["job_id"],
                    "experiment": EXPERIMENT_NAME,
                    "intervention_type": "premise_acceptance",
                    "candidate_id": candidate["candidate_id"],
                    "candidate_text": text,
                    "candidate_provenance": "nudge_2_informed",
                    "base_raw_prefix": base_job["raw_prefix"],
                    "raw_prefix": injected_prefix,
                    "insertion_line": insertion_line,
                }
            )
    job_ids = [job["job_id"] for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError("expanded manifest contains duplicate job IDs")
    return jobs


def load_design(
    sources_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load sources and return base jobs, sources, and generated jobs."""
    sources = load_jsonl(sources_path)
    base_jobs = build_base_jobs(sources)
    jobs = build_jobs(base_jobs)
    return base_jobs, sources, jobs


def validate_design(base_jobs: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> None:
    """Check the complete factorial design before loading the model."""
    if len(base_jobs) != EXPECTED_BASE_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_BASE_JOBS} base jobs, found {len(base_jobs)}"
        )
    if len(jobs) != EXPECTED_JOBS:
        raise RuntimeError(f"expected {EXPECTED_JOBS} jobs, found {len(jobs)}")
    interventions = [
        job for job in jobs if job["intervention_type"] == "premise_acceptance"
    ]
    controls = [
        job for job in jobs if job["intervention_type"] == "unaltered_control"
    ]
    if len(interventions) != EXPECTED_INTERVENTION_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_INTERVENTION_JOBS} interventions, "
            f"found {len(interventions)}"
        )
    if len(controls) != EXPECTED_CONTROL_JOBS:
        raise RuntimeError(
            f"expected {EXPECTED_CONTROL_JOBS} controls, found {len(controls)}"
        )
    cells = Counter(
        (job["source_id"], job["cut"], job["candidate_id"]) for job in jobs
    )
    if len(cells) != 36 or set(cells.values()) != {ROLLOUTS_PER_CELL}:
        raise RuntimeError(f"unexpected experimental cell counts: {dict(cells)}")


def prepare_jobs(
    jobs: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Verify source cuts, tokenize augmented prefixes, and set fixed budgets."""
    prepared: list[dict[str, Any]] = []
    for job in jobs:
        source_index = job["source_index"]
        if not isinstance(source_index, int) or not 0 <= source_index < len(sources):
            raise IndexError(f"invalid source index for {job['job_id']}")
        source = sources[source_index]["record"]
        base_prefix = job["base_raw_prefix"]
        if not source["raw_completion"].startswith(base_prefix):
            raise RuntimeError(f"base prefix mismatch for {job['job_id']}")
        if base_prefix.splitlines()[-1] != job["cut_line_text"]:
            raise RuntimeError(f"cut line mismatch for {job['job_id']}")
        if job["intervention_type"] == "unaltered_control":
            if job["raw_prefix"] != base_prefix or job["insertion_line"]:
                raise RuntimeError(f"control prefix mismatch for {job['job_id']}")
        else:
            if job["raw_prefix"] != base_prefix + job["insertion_line"]:
                raise RuntimeError(f"injected prefix mismatch for {job['job_id']}")
            if not job["raw_prefix"].endswith(job["insertion_line"]):
                raise RuntimeError(f"candidate is not at the end of {job['job_id']}")

        chat_ids = base_runner.normalize_token_ids(
            tokenizer.apply_chat_template(
                source["conversation"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
            label="chat template",
        )
        prefix_ids = base_runner.normalize_token_ids(
            tokenizer.encode(job["raw_prefix"], add_special_tokens=False),
            label="augmented trace prefix",
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
        prompt_ids = chat_ids + prefix_ids
        if not prompt_ids or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in prompt_ids
        ):
            raise TypeError(f"prompt IDs are not a flat integer list for {job['job_id']}")
        prepared.append(
            {
                "job": job,
                "source": source,
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
    output_path: Path, manifest_jobs: list[dict[str, Any]]
) -> set[str]:
    if not output_path.exists():
        return set()
    manifest = {job["job_id"]: job for job in manifest_jobs}
    rows = load_jsonl(output_path)
    ids = [row.get("job_id") for row in rows]
    if any(not isinstance(job_id, str) for job_id in ids):
        raise RuntimeError(f"existing output has a row without a job ID: {output_path}")
    duplicates = sorted(job_id for job_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"existing output has duplicate job IDs: {duplicates[:5]}")
    unknown = sorted(set(ids) - set(manifest))
    if unknown:
        raise RuntimeError(f"existing output has unknown job IDs: {unknown[:5]}")
    identity_fields = (
        "base_job_id",
        "source_id",
        "source_rank",
        "case_name",
        "cut",
        "cut_line_1_based",
        "intervention_type",
        "candidate_id",
        "candidate_text",
        "sample_seed",
    )
    for row in rows:
        expected = manifest[row["job_id"]]
        mismatched = [
            field for field in identity_fields if row.get(field) != expected.get(field)
        ]
        if mismatched:
            raise RuntimeError(
                f"existing output does not match the current design for "
                f"{row['job_id']}: {mismatched}"
            )
    return set(ids)


def filter_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    allowed_cases = set(args.cases) if args.cases else None
    allowed_cuts = set(args.cuts) if args.cuts else None
    allowed_candidates = set(args.candidate_ids) if args.candidate_ids else None
    selected: list[dict[str, Any]] = []
    for manifest_index, job in enumerate(jobs):
        if manifest_index % args.num_shards != args.shard_index:
            continue
        if allowed_cases is not None and job["case_name"] not in allowed_cases:
            continue
        if allowed_cuts is not None and job["cut"] not in allowed_cuts:
            continue
        if (
            allowed_candidates is not None
            and job["candidate_id"] not in allowed_candidates
        ):
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
    output_path = args.output or default_output_path(args.num_shards, args.shard_index)

    base_jobs, sources, all_jobs = load_design(args.sources)
    validate_design(base_jobs, all_jobs)
    selected_jobs = filter_jobs(all_jobs, args)

    completed = validate_existing_output(output_path, all_jobs)
    selected_jobs = [job for job in selected_jobs if job["job_id"] not in completed]
    if args.limit is not None:
        selected_jobs = selected_jobs[: args.limit]
    if not selected_jobs:
        print(f"No unfinished jobs selected for {output_path}")
        return 0

    llm, SamplingParams, _ = base_runner.build_engine(args.gpu_memory_utilization)
    tokenizer = llm.get_tokenizer()
    base_runner.runtime_smoke(llm, tokenizer, SamplingParams)

    # Prepare the full design so partial and resumed runs use the same fixed
    # token budget for every member of a story family.
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

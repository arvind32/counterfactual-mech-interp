#!/usr/bin/env python3
"""Validate held-out-family rollout output and create a review-ready table."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "trace_branching"
DEFAULT_INPUT = RESULTS_ROOT / "three_anchor_generalization_rollouts.jsonl"
DEFAULT_REPORT = RESULTS_ROOT / "three_anchor_generalization_integrity.json"
DEFAULT_REVIEW = RESULTS_ROOT / "three_anchor_generalization_review.csv"
ANCHORS = ("A", "A_STAR", "C")
BRANCHES = ("BEFORE", "AFTER")
SAMPLE_SEEDS = tuple(range(4000, 4030))
EXPERIMENT_NAME = "three_anchor_generalization_before_after"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check rollout integrity and write a review-ready CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        help="Input JSONL. Repeat to inspect several shard files together.",
    )
    parser.add_argument("--sources", type=Path, default=ROOT / "source_records.jsonl")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--allow-partial", action="store_true")
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


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expected_manifest(
    sources: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if len(sources) != 6:
        raise RuntimeError(f"expected six sources, found {len(sources)}")
    story_keys = [row["selection"].get("story_key") for row in sources]
    if len(set(story_keys)) != 6:
        raise RuntimeError("expected six distinct source families")
    manifest: dict[str, dict[str, Any]] = {}
    prefixes: dict[str, str] = {}
    pair_index = 0
    for source_row in sources:
        selected = source_row["selection"]
        raw_completion = source_row["record"]["raw_completion"]
        c_position = selected["anchor_order"].index("C")
        for anchor_name in ANCHORS:
            anchor = selected["anchors"][anchor_name]
            anchor_position = selected["anchor_order"].index(anchor_name)
            relative_to_c = (
                "SELF"
                if anchor_name == "C"
                else ("BEFORE_C" if anchor_position < c_position else "AFTER_C")
            )
            branch_prefixes = {
                "BEFORE": raw_completion[: anchor["before_prefix_char_count"]],
                "AFTER": raw_completion[: anchor["after_prefix_char_count"]],
            }
            for rollout_index, sample_seed in enumerate(SAMPLE_SEEDS):
                pair_id = (
                    f"{selected['source_id']}__{anchor_name}__r{rollout_index:02d}"
                )
                for branch in BRANCHES:
                    job_id = f"{pair_id}__{branch}"
                    prefix = branch_prefixes[branch]
                    manifest[job_id] = {
                        "experiment": EXPERIMENT_NAME,
                        "pair_id": pair_id,
                        "pair_index": pair_index,
                        "source_id": selected["source_id"],
                        "source_label": selected["source_label"],
                        "source_rank": selected["source_rank"],
                        "story_family": selected["domain"],
                        "story_key": selected["story_key"],
                        "case_name": selected["case_name"],
                        "source_condition": selected["condition"],
                        "source_condition_alias": selected["condition_alias"],
                        "source_seed": selected["seed"],
                        "source_original_result": selected["original_result"],
                        "fact_origin": selected["fact_origin"],
                        "required_fact": selected["required_fact"],
                        "rubric_key": selected["rubric_key"],
                        "rubric_summary": selected["rubric_summary"],
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
                        "c_is_correct": selected["c_is_correct"],
                        "branch": branch,
                        "cut_char_count": len(prefix),
                        "rollout_index": rollout_index,
                        "sample_seed": sample_seed,
                        "intervention_type": "natural_trace_before_after",
                    }
                    prefixes[job_id] = prefix
                pair_index += 1
    return manifest, prefixes


def main() -> int:
    args = parse_args()
    inputs = args.inputs or [DEFAULT_INPUT]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.sources.is_file():
        raise FileNotFoundError(args.sources)

    sources = load_jsonl(args.sources)
    manifest, prefixes = expected_manifest(sources)
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(load_jsonl(path))

    ids = [row.get("job_id") for row in rows]
    if any(not isinstance(job_id, str) for job_id in ids):
        raise RuntimeError("one or more rows have no valid job_id")
    duplicates = sorted(job_id for job_id, count in Counter(ids).items() if count > 1)
    unknown = sorted(set(ids) - set(manifest))
    missing = sorted(set(manifest) - set(ids))
    if duplicates:
        raise RuntimeError(f"duplicate job IDs: {duplicates[:10]}")
    if unknown:
        raise RuntimeError(f"unknown job IDs: {unknown[:10]}")
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"missing {len(missing)} of {len(manifest)} jobs; use --allow-partial "
            "only for an intentional partial inspection"
        )

    identity_fields = (
        "experiment",
        "pair_id",
        "pair_index",
        "source_id",
        "source_label",
        "source_rank",
        "story_family",
        "story_key",
        "case_name",
        "source_condition",
        "source_condition_alias",
        "source_seed",
        "source_original_result",
        "fact_origin",
        "required_fact",
        "rubric_key",
        "rubric_summary",
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
    mismatches: list[dict[str, Any]] = []
    content_errors: list[str] = []
    for row in rows:
        job_id = row["job_id"]
        expected = manifest[job_id]
        fields = [field for field in identity_fields if row.get(field) != expected[field]]
        if fields:
            mismatches.append({"job_id": job_id, "fields": fields})
        prefix = prefixes[job_id]
        if row.get("raw_prefix") != prefix:
            content_errors.append(f"raw prefix mismatch: {job_id}")
        if row.get("prefix_text_hash") != hash_text(prefix):
            content_errors.append(f"prefix hash mismatch: {job_id}")
        continuation = row.get("continuation_text")
        if not isinstance(continuation, str):
            content_errors.append(f"missing continuation text: {job_id}")
        elif row.get("full_completion") != prefix + continuation:
            content_errors.append(f"full completion mismatch: {job_id}")
        if row.get("prefix_text_roundtrip_verified") is not True:
            content_errors.append(f"prefix round trip not verified: {job_id}")
    if mismatches:
        raise RuntimeError(f"manifest mismatches: {mismatches[:5]}")
    if content_errors:
        raise RuntimeError(f"content-integrity errors: {content_errors[:5]}")

    present = set(ids)
    complete_pairs = 0
    incomplete_pairs = 0
    for pair_id in {item["pair_id"] for item in manifest.values()}:
        pair_ids = {f"{pair_id}__BEFORE", f"{pair_id}__AFTER"}
        count = len(pair_ids & present)
        if count == 2:
            complete_pairs += 1
        elif count == 1:
            incomplete_pairs += 1

    cells = Counter(
        (row["source_id"], row["anchor"], row["branch"]) for row in rows
    )
    finish_reasons = Counter(str(row.get("finish_reason")) for row in rows)
    thinking_statuses = Counter(str(row.get("thinking_status")) for row in rows)
    budget_exhausted = sum(bool(row.get("budget_exhausted")) for row in rows)
    missing_response = sum(not isinstance(row.get("response"), str) for row in rows)
    report = {
        "result": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in inputs],
        "rows": len(rows),
        "expected_rows": len(manifest),
        "missing_jobs": len(missing),
        "duplicate_jobs": len(duplicates),
        "unknown_jobs": len(unknown),
        "manifest_mismatches": len(mismatches),
        "content_integrity_errors": len(content_errors),
        "complete_before_after_pairs": complete_pairs,
        "incomplete_before_after_pairs": incomplete_pairs,
        "cell_counts": {
            f"{source_id} / {anchor} / {branch}": count
            for (source_id, anchor, branch), count in sorted(cells.items())
        },
        "finish_reasons": dict(finish_reasons),
        "thinking_statuses": dict(thinking_statuses),
        "budget_exhausted": budget_exhausted,
        "missing_response": missing_response,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    review_fields = [
        "job_id",
        "pair_id",
        "story_family",
        "source_label",
        "source_id",
        "source_original_result",
        "required_fact",
        "rubric_key",
        "rubric_summary",
        "anchor_order",
        "anchor",
        "anchor_relative_to_c",
        "branch",
        "rollout_index",
        "sample_seed",
        "c_value",
        "c_is_correct",
        "finish_reason",
        "budget_exhausted",
        "thinking_status",
        "rubric_result",
        "final_outcome",
        "final_agrees_with_source_c",
        "recreates_anchor_before_first_c",
        "recreates_anchor_anywhere",
        "review_notes",
        "response",
        "thinking",
        "continuation_text",
    ]
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item["story_key"],
                item["source_rank"],
                ANCHORS.index(item["anchor"]),
                item["rollout_index"],
                BRANCHES.index(item["branch"]),
            ),
        ):
            writer.writerow(
                {
                    "job_id": row["job_id"],
                    "pair_id": row["pair_id"],
                    "story_family": row["story_family"],
                    "source_label": row["source_label"],
                    "source_id": row["source_id"],
                    "source_original_result": row["source_original_result"],
                    "required_fact": row["required_fact"],
                    "rubric_key": row["rubric_key"],
                    "rubric_summary": row["rubric_summary"],
                    "anchor_order": " < ".join(row["anchor_order"]),
                    "anchor": row["anchor"],
                    "anchor_relative_to_c": row["anchor_relative_to_c"],
                    "branch": row["branch"],
                    "rollout_index": row["rollout_index"],
                    "sample_seed": row["sample_seed"],
                    "c_value": row["c_value"],
                    "c_is_correct": row["c_is_correct"],
                    "finish_reason": row.get("finish_reason"),
                    "budget_exhausted": row.get("budget_exhausted"),
                    "thinking_status": row.get("thinking_status"),
                    "rubric_result": "",
                    "final_outcome": "",
                    "final_agrees_with_source_c": "",
                    "recreates_anchor_before_first_c": "",
                    "recreates_anchor_anywhere": "",
                    "review_notes": "",
                    "response": row.get("response"),
                    "thinking": row.get("thinking"),
                    "continuation_text": row.get("continuation_text"),
                }
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Integrity report: {args.report}")
    print(f"Review table: {args.review_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

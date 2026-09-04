#!/usr/bin/env python3
"""Validate Continuations 2 output and create a compact review table."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "trace_branching"
DEFAULT_INPUT = RESULTS_ROOT / "continuations_2_controls.jsonl"
DEFAULT_REPORT = RESULTS_ROOT / "continuations_2_integrity.json"
DEFAULT_REVIEW = RESULTS_ROOT / "continuations_2_review.csv"
LOCATIONS = ("A_STAR", "A_STAR_STAR")
ROLLOUTS_PER_CELL = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check rollout integrity and write a review-ready CSV."
    )
    parser.add_argument(
        "--input", type=Path, action="append", dest="inputs",
        help="Input JSONL. Repeat to inspect several shard files together."
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


def expected_manifest(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for source_row in sources:
        selected = source_row["selection"]
        for location in LOCATIONS:
            anchor = selected["anchors"][location]
            for rollout_index in range(ROLLOUTS_PER_CELL):
                job_id = (
                    f"{selected['source_id']}__{location}__r{rollout_index:02d}"
                )
                manifest[job_id] = {
                    "source_id": selected["source_id"],
                    "source_label": selected["source_label"],
                    "source_rank": selected["source_rank"],
                    "story_family": selected["domain"],
                    "story_key": selected["story_key"],
                    "case_name": selected["case_name"],
                    "source_condition": selected["condition"],
                    "source_seed": selected["seed"],
                    "location": location,
                    "cut_line_1_based": anchor["raw_completion_line_1_based"],
                    "cut_line_text": anchor["text"],
                    "rollout_index": rollout_index,
                    "sample_seed": 2000 + rollout_index,
                }
    return manifest


def main() -> int:
    args = parse_args()
    inputs = args.inputs or [DEFAULT_INPUT]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.sources.is_file():
        raise FileNotFoundError(args.sources)

    sources = load_jsonl(args.sources)
    manifest = expected_manifest(sources)
    rows: list[dict[str, Any]] = []
    for path in inputs:
        rows.extend(load_jsonl(path))

    ids = [row.get("job_id") for row in rows]
    invalid_ids = [job_id for job_id in ids if not isinstance(job_id, str)]
    duplicates = sorted(job_id for job_id, n in Counter(ids).items() if n > 1)
    unknown = sorted(set(ids) - set(manifest))
    missing = sorted(set(manifest) - set(ids))
    if invalid_ids:
        raise RuntimeError("one or more rows have no valid job_id")
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
        "source_id",
        "source_rank",
        "story_family",
        "story_key",
        "case_name",
        "source_condition",
        "source_seed",
        "location",
        "cut_line_1_based",
        "cut_line_text",
        "rollout_index",
        "sample_seed",
    )
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        expected = manifest[row["job_id"]]
        fields = [field for field in identity_fields if row.get(field) != expected[field]]
        if fields:
            mismatches.append({"job_id": row["job_id"], "fields": fields})
    if mismatches:
        raise RuntimeError(f"manifest mismatches: {mismatches[:5]}")

    cells = Counter((row["source_id"], row["location"]) for row in rows)
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
        "cell_counts": {
            f"{source_id} / {location}": count
            for (source_id, location), count in sorted(cells.items())
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
        "story_family",
        "source_label",
        "source_id",
        "location",
        "rollout_index",
        "sample_seed",
        "finish_reason",
        "budget_exhausted",
        "thinking_status",
        "overall_result",
        "review_notes",
        "response",
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
                LOCATIONS.index(item["location"]),
                item["rollout_index"],
            ),
        ):
            writer.writerow(
                {
                    "job_id": row["job_id"],
                    "story_family": row["story_family"],
                    "source_label": row["source_label"],
                    "source_id": row["source_id"],
                    "location": row["location"],
                    "rollout_index": row["rollout_index"],
                    "sample_seed": row["sample_seed"],
                    "finish_reason": row.get("finish_reason"),
                    "budget_exhausted": row.get("budget_exhausted"),
                    "thinking_status": row.get("thinking_status"),
                    "overall_result": "",
                    "review_notes": "",
                    "response": row.get("response"),
                    "continuation_text": row.get("continuation_text"),
                }
            )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Integrity report: {args.report}")
    print(f"Review table: {args.review_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

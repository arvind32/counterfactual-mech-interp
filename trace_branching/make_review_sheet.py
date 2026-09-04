#!/usr/bin/env python3
"""Create a compact sheet for blinded whole-answer grading."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
results = [json.loads(line) for line in (ROOT / "ab_results.jsonl").read_text().splitlines()]
sources = [json.loads(line) for line in (ROOT / "source_records.jsonl").read_text().splitlines()]
fields = [
    "review_order", "job_id", "story_family", "cut", "sample_seed", "response",
    "expectation", "q1", "q2", "q3", "q4", "overall_pass", "first_failure_stage", "notes",
]
rows = []
for result in results:
    source = sources[result["source_index"]]["record"]
    rows.append({
        "job_id": result["job_id"],
        "story_family": result["story_family"],
        "cut": result["cut"],
        "sample_seed": result["sample_seed"],
        "response": result["response"],
        "expectation": source["expectation"],
    })
# Deterministic order that mixes conditions without using outcomes.
rows.sort(key=lambda row: (row["sample_seed"], row["story_family"], row["cut"]))
for index, row in enumerate(rows, 1):
    row["review_order"] = index
with (ROOT / "ab_review.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {ROOT / 'ab_review.csv'}")

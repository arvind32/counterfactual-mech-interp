#!/usr/bin/env python3
"""Package pinned source records and build the 60-job A/B manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "source_selection.json",
        help="JSON selection file describing the pinned source records and cuts.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Map a source filename in the selection to its JSONL path. Repeatable.",
    )
    return parser.parse_args()


def parse_source_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source mapping must be NAME=PATH: {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"source mapping must be NAME=PATH: {value!r}")
        result[name] = Path(path)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_prefix(raw_completion: str, line_number: int) -> tuple[str, str]:
    thinking_raw = raw_completion.split("</think>", 1)[0]
    lines = thinking_raw.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise ValueError(f"line {line_number} outside 1..{len(lines)}")
    prefix = "".join(lines[:line_number])
    line_text = lines[line_number - 1].rstrip("\r\n")
    return prefix, line_text


def main() -> int:
    args = parse_args()
    source_paths = parse_source_paths(args.source)
    selection = json.loads(args.selection.read_text())
    required_names = {item["source_file"] for item in selection["sources"]}
    missing = required_names - set(source_paths)
    if missing:
        raise ValueError(
            "missing --source mapping(s): " + ", ".join(sorted(missing))
        )
    source_cache: dict[str, list[dict]] = {}
    packaged: list[dict] = []
    for candidate in selection["sources"]:
        name = candidate["source_file"]
        path = source_paths[name]
        actual_hash = sha256(path)
        if actual_hash != candidate["source_sha256"]:
            raise RuntimeError(f"source hash changed for {path}: {actual_hash}")
        rows = source_cache.setdefault(
            name, [json.loads(line) for line in path.read_text().splitlines()]
        )
        record = rows[candidate["source_record_1_based"] - 1]
        for key in ("case_name", "condition", "seed"):
            if record[key] != candidate[key]:
                raise RuntimeError(f"source identity mismatch: {candidate}")
        packaged.append({"selection": candidate, "record": record})

    source_out = ROOT / "source_records.jsonl"
    source_out.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in packaged)
    )

    jobs: list[dict] = []
    for source_index, item in enumerate(packaged):
        selected = item["selection"]
        if selected["role"] != "failed_source":
            continue
        raw = item["record"]["raw_completion"]
        for cut_name in ("A", "B"):
            anchor = selected["anchors"][cut_name]
            prefix, found_line = line_prefix(
                raw, anchor["thinking_physical_line_1_based"]
            )
            if found_line != anchor["text"]:
                raise RuntimeError(
                    f"anchor mismatch for {selected['case_name']} {cut_name}:\n"
                    f"expected {anchor['text']!r}\nfound {found_line!r}"
                )
            if not raw.startswith(prefix):
                raise AssertionError("prefix is not an exact raw-completion prefix")
            for rollout_index in range(10):
                sample_seed = 1000 + rollout_index
                jobs.append(
                    {
                        "job_id": (
                            f"{selected['dataset']}__{selected['case_name']}__"
                            f"{selected['condition']}__s{selected['seed']}__"
                            f"{cut_name}__r{rollout_index}"
                        ),
                        "source_index": source_index,
                        "story_family": selected["domain"],
                        "case_name": selected["case_name"],
                        "source_condition": selected["condition"],
                        "source_seed": selected["seed"],
                        "cut": cut_name,
                        "cut_line_1_based": anchor[
                            "thinking_physical_line_1_based"
                        ],
                        "cut_line_text": found_line,
                        "rollout_index": rollout_index,
                        "sample_seed": sample_seed,
                        "raw_prefix": prefix,
                    }
                )
    if len(jobs) != 60:
        raise AssertionError(f"expected 60 A/B jobs, got {len(jobs)}")
    (ROOT / "ab_jobs.jsonl").write_text(
        "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
    )
    print(f"Wrote {len(packaged)} pinned sources and {len(jobs)} A/B jobs to {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

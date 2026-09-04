#!/usr/bin/env python3
"""Select and package one reproducible random control line per source trace."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
DEFAULT_CONFIG = ROOT / "random_selection_config.json"
DEFAULT_OUTPUT = ROOT / "random_source_records.jsonl"
DEFAULT_AUDIT = (
    PROJECT_ROOT / "results" / "trace_branching" /
    "random_location_selection_audit.json"
)
KNOWN_ANCHORS = ("A", "A_STAR", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select deterministic random control passages in 12 traces."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def line_records(raw_completion: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    for line_number, line in enumerate(raw_completion.splitlines(keepends=True), 1):
        if line.endswith("\r\n"):
            body = line[:-2]
        elif line.endswith(("\n", "\r")):
            body = line[:-1]
        else:
            body = line
        records.append(
            {
                "line_number": line_number,
                "line": line,
                "body": body,
                "line_start": offset,
                "line_end": offset + len(body),
                "after_line": offset + len(line),
            }
        )
        offset += len(line)
    return records


def anchor_intervals(selected: dict[str, Any]) -> list[tuple[int, int, str]]:
    anchors = selected.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != set(KNOWN_ANCHORS):
        raise RuntimeError(
            f"{selected.get('source_id')} does not contain resolved A, A_STAR, C anchors"
        )
    intervals: list[tuple[int, int, str]] = []
    for name in KNOWN_ANCHORS:
        anchor = anchors[name]
        start = anchor.get("start_char_0_based")
        end = anchor.get("end_char_0_based_exclusive")
        if not isinstance(start, int) or not isinstance(end, int) or not start < end:
            raise RuntimeError(f"unresolved {name} anchor in {selected.get('source_id')}")
        intervals.append((start, end, name))
    return intervals


def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def eligible_lines(
    raw_completion: str,
    selected: dict[str, Any],
    minimum_characters: int,
    minimum_words: int,
) -> list[dict[str, Any]]:
    thinking_end = raw_completion.find("</think>")
    if thinking_end < 0:
        raise RuntimeError(f"missing </think> in {selected['source_id']}")
    known = anchor_intervals(selected)
    candidates: list[dict[str, Any]] = []
    for item in line_records(raw_completion):
        body = item["body"]
        content = body.strip()
        if not content or content in {"<think>", "</think>", "Thinking Process:"}:
            continue
        content_start = item["line_start"] + body.index(content)
        content_end = content_start + len(content)
        if content_end > thinking_end:
            continue
        if len(content) < minimum_characters:
            continue
        if len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", content)) < minimum_words:
            continue
        intersecting = [
            name for start, end, name in known
            if overlaps((content_start, content_end), (start, end))
        ]
        if intersecting:
            continue
        after_offset = item["after_line"]
        before_prefix = raw_completion[:content_start]
        after_prefix = raw_completion[:after_offset]
        inserted_text = after_prefix[len(before_prefix):]
        if not inserted_text.startswith(content):
            raise RuntimeError(
                f"line reconstruction failed for {selected['source_id']} line "
                f"{item['line_number']}"
            )
        candidates.append(
            {
                "line_number_1_based": item["line_number"],
                "start_char_0_based": content_start,
                "end_char_0_based_exclusive": content_end,
                "after_char_0_based_exclusive": after_offset,
                "text": content,
                "text_sha256": sha256_text(content),
                "inserted_text": inserted_text,
                "before_prefix_char_count": len(before_prefix),
                "after_prefix_char_count": len(after_prefix),
                "overlaps_known_anchor": False,
            }
        )
    if not candidates:
        raise RuntimeError(f"no eligible random lines in {selected['source_id']}")
    return candidates


def per_source_seed(global_seed: int, source_id: str) -> int:
    payload = f"{global_seed}:{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def validate_source(row: dict[str, Any], package_name: str) -> None:
    selected = row.get("selection")
    record = row.get("record")
    if not isinstance(selected, dict) or not isinstance(record, dict):
        raise RuntimeError(f"malformed source in {package_name}")
    source_id = selected.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise RuntimeError(f"source without an ID in {package_name}")
    if selected.get("role") != "anchor_source":
        raise RuntimeError(f"{source_id} is not an anchor source")
    baseline_label = selected.get("condition_alias", selected.get("condition"))
    if baseline_label != "baseline_p5":
        raise RuntimeError(f"{source_id} is not baseline_p5")
    if selected.get("original_result") not in {"PASS", "FAIL"}:
        raise RuntimeError(f"{source_id} has no valid source result")
    if not isinstance(record.get("raw_completion"), str):
        raise RuntimeError(f"{source_id} has no raw completion")
    conversation = record.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise RuntimeError(f"{source_id} has no replay conversation")
    anchor_intervals(selected)


def build_package(
    config_path: Path,
    output_path: Path,
    audit_path: Path,
) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment") != "random_location_before_after_control":
        raise RuntimeError("unexpected experiment name")
    if config.get("rollouts_per_cell") != 30:
        raise RuntimeError("configuration must specify 30 rollouts per cell")
    if (config.get("sample_seed_start"), config.get("sample_seed_end_inclusive")) != (
        5000,
        5029,
    ):
        raise RuntimeError("configuration must reserve rollout seeds 5000 through 5029")
    global_seed = config.get("randomization_seed")
    if not isinstance(global_seed, int):
        raise RuntimeError("randomization_seed must be an integer")
    eligibility = config.get("eligibility", {})
    minimum_characters = eligibility.get("minimum_characters")
    minimum_words = eligibility.get("minimum_alphabetic_words")
    if not isinstance(minimum_characters, int) or minimum_characters < 1:
        raise RuntimeError("invalid minimum_characters")
    if not isinstance(minimum_words, int) or minimum_words < 1:
        raise RuntimeError("invalid minimum_alphabetic_words")
    if eligibility.get("exclude_known_anchor_overlaps") is not True:
        raise RuntimeError("known anchor overlaps must be excluded")

    packages = config.get("source_packages")
    if not isinstance(packages, list) or len(packages) != 2:
        raise RuntimeError("configuration must identify two source packages")
    packaged: list[dict[str, Any]] = []
    package_audit: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    origins: set[str] = set()
    for package in packages:
        package_name = package.get("name")
        if not isinstance(package_name, str) or not package_name:
            raise RuntimeError("source package has no name")
        origins.add(package_name)
        source_path = resolve_project_path(package["path"])
        if not source_path.is_file():
            raise FileNotFoundError(
                f"missing {source_path}; run that experiment's smoke test or "
                "prepare_sources.py first"
            )
        rows = load_jsonl(source_path)
        expected_sources = package.get("expected_sources")
        if len(rows) != expected_sources:
            raise RuntimeError(
                f"expected {expected_sources} sources in {source_path}, found {len(rows)}"
            )
        package_audit.append(
            {
                "name": package_name,
                "path": str(source_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(source_path),
                "source_count": len(rows),
            }
        )
        for source_row in rows:
            validate_source(source_row, package_name)
            selected = copy.deepcopy(source_row["selection"])
            record = copy.deepcopy(source_row["record"])
            source_id = selected["source_id"]
            if source_id in source_ids:
                raise RuntimeError(f"duplicate source ID: {source_id}")
            source_ids.add(source_id)
            candidates = eligible_lines(
                record["raw_completion"],
                selected,
                minimum_characters,
                minimum_words,
            )
            selection_seed = per_source_seed(global_seed, source_id)
            selected_index = random.Random(selection_seed).randrange(len(candidates))
            chosen = candidates[selected_index]
            selected["source_package"] = package_name
            selected["condition_alias"] = selected.get(
                "condition_alias", selected.get("condition")
            )
            selected["c_is_correct"] = selected.get(
                "c_is_correct", str(selected.get("c_value", "")).startswith("correct_")
            )
            selected["required_fact"] = selected.get(
                "required_fact",
                record.get("reference_background_fact")
                or record.get("knowledge_reasoning")
                or "See the source expectation and reasoning metadata.",
            )
            selected["random_control"] = {
                "global_randomization_seed": global_seed,
                "per_source_randomization_seed": selection_seed,
                "eligible_candidate_count": len(candidates),
                "selected_candidate_index_0_based": selected_index,
                **chosen,
            }
            selected["rubric_key"] = selected.get(
                "rubric_key", selected.get("story_key", selected["source_id"])
            )
            selected["rubric_summary"] = selected.get(
                "rubric_summary", record.get("expectation", "")
            )
            if not selected["rubric_summary"]:
                raise RuntimeError(f"missing rubric summary for {source_id}")
            packaged.append({"selection": selected, "record": record})

    if len(packaged) != 12 or len(source_ids) != 12:
        raise RuntimeError("expected exactly 12 unique source traces")
    if len(origins) != 2:
        raise RuntimeError("expected both source experiments")
    if {row["selection"]["original_result"] for row in packaged} != {"PASS", "FAIL"}:
        raise RuntimeError("expected both passing and failing source traces")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packaged),
        encoding="utf-8",
    )
    audit = {
        "experiment": config["experiment"],
        "randomization_seed": global_seed,
        "eligibility": eligibility,
        "source_packages": package_audit,
        "source_count": len(packaged),
        "locations": [
            {
                "source_id": row["selection"]["source_id"],
                "source_label": row["selection"]["source_label"],
                "story_key": row["selection"]["story_key"],
                "source_package": row["selection"]["source_package"],
                **row["selection"]["random_control"],
            }
            for row in packaged
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return packaged


def main() -> int:
    args = parse_args()
    packaged = build_package(args.config, args.output, args.audit)
    print(f"Selected and packaged {len(packaged)} random control locations")
    for row in packaged:
        selected = row["selection"]
        control = selected["random_control"]
        print(
            f"- {selected['source_label']}: line {control['line_number_1_based']} "
            f"({control['selected_candidate_index_0_based'] + 1}/"
            f"{control['eligible_candidate_count']})"
        )
    print(f"Source package: {args.output}")
    print(f"Randomization audit: {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify and package held-out-family traces for the three-anchor experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
DEFAULT_SELECTION = ROOT / "source_selection.json"
DEFAULT_OUTPUT = ROOT / "source_records.jsonl"
ANCHORS = ("A", "A_STAR", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and package the six configured baseline traces."
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def resolve_source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def line_start_offsets(raw_completion: str) -> tuple[list[str], list[int]]:
    lines = raw_completion.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    return lines, offsets


def strip_line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith(("\n", "\r")):
        return line[:-1]
    return line


def locate_anchor(raw_completion: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve one exact span and the immediately-before/after prefixes."""
    lines, offsets = line_start_offsets(raw_completion)
    start_line = spec.get("start_line_1_based")
    end_line = spec.get("end_line_1_based")
    start_text = spec.get("start_text")
    end_text = spec.get("end_text")
    if not isinstance(start_line, int) or not 1 <= start_line <= len(lines):
        raise RuntimeError(f"invalid start line: {start_line!r}")
    if not isinstance(end_line, int) or not start_line <= end_line <= len(lines):
        raise RuntimeError(f"invalid end line: {end_line!r}")
    if not isinstance(start_text, str) or not start_text:
        raise RuntimeError("anchor start_text must be nonempty")
    if not isinstance(end_text, str) or not end_text:
        raise RuntimeError("anchor end_text must be nonempty")

    start_body = strip_line_ending(lines[start_line - 1])
    end_body = strip_line_ending(lines[end_line - 1])
    if start_body.count(start_text) != 1:
        raise RuntimeError(
            f"start text is not unique on line {start_line}: {start_text!r}"
        )
    if end_body.count(end_text) != 1:
        raise RuntimeError(
            f"end text is not unique on line {end_line}: {end_text!r}"
        )

    start_offset = offsets[start_line - 1] + start_body.index(start_text)
    end_offset = offsets[end_line - 1] + end_body.index(end_text) + len(end_text)
    if end_offset <= start_offset:
        raise RuntimeError("anchor span is empty or reversed")

    after_offset = end_offset
    if raw_completion.startswith("\r\n", after_offset):
        after_offset += 2
    elif raw_completion.startswith(("\n", "\r"), after_offset):
        after_offset += 1

    anchor_text = raw_completion[start_offset:end_offset]
    before_prefix = raw_completion[:start_offset]
    after_prefix = raw_completion[:after_offset]
    inserted_text = after_prefix[len(before_prefix) :]
    if not inserted_text.startswith(anchor_text):
        raise RuntimeError("after prefix does not add the configured anchor")
    if not raw_completion.startswith(before_prefix) or not raw_completion.startswith(
        after_prefix
    ):
        raise RuntimeError("resolved prefix is not an exact source prefix")

    return {
        **spec,
        "start_char_0_based": start_offset,
        "end_char_0_based_exclusive": end_offset,
        "after_char_0_based_exclusive": after_offset,
        "anchor_text": anchor_text,
        "inserted_text": inserted_text,
        "before_prefix_char_count": len(before_prefix),
        "after_prefix_char_count": len(after_prefix),
    }


def verify_identity(record: dict[str, Any], selected: dict[str, Any]) -> None:
    source_id = selected["source_id"]
    for field in ("example_id", "condition"):
        if record.get(field) != selected.get(field):
            raise RuntimeError(f"source identity mismatch for {source_id}: {field}")
    generation_config = record.get("generation_config")
    if not isinstance(generation_config, dict):
        raise RuntimeError(f"missing generation configuration for {source_id}")
    if str(generation_config.get("seed")) != str(selected.get("seed")):
        raise RuntimeError(
            f"source identity mismatch for {source_id}: generation seed"
        )
    if selected.get("condition_alias") != "baseline_p5":
        raise RuntimeError(f"{source_id} is not labeled as baseline_p5")
    if not isinstance(record.get("prompt"), str) or not record["prompt"]:
        raise RuntimeError(f"missing prompt for {source_id}")
    if selected.get("role") != "anchor_source":
        raise RuntimeError(f"{source_id} is not an anchor source")
    if selected.get("original_result") not in {"PASS", "FAIL"}:
        raise RuntimeError(f"invalid original result for {source_id}")
    if not isinstance(selected.get("c_is_correct"), bool):
        raise RuntimeError(f"missing C correctness label for {source_id}")


def normalize_record(record: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    """Add the replay fields used by the runner without changing source content."""
    normalized = copy.deepcopy(record)
    prompt = normalized["prompt"]
    conversation = normalized.get("conversation")
    if conversation is None:
        normalized["conversation"] = [{"role": "user", "content": prompt}]
    elif conversation != [{"role": "user", "content": prompt}]:
        raise RuntimeError(
            f"unexpected conversation structure for {selected['source_id']}"
        )
    normalized["seed"] = normalized["generation_config"]["seed"]
    normalized["case_name"] = selected["case_name"]
    return normalized


def resolve_anchors(record: dict[str, Any], selected: dict[str, Any]) -> None:
    raw_completion = record.get("raw_completion")
    if not isinstance(raw_completion, str) or not raw_completion:
        raise RuntimeError(f"missing raw_completion for {selected['source_id']}")
    anchors = selected.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != set(ANCHORS):
        raise RuntimeError(
            f"{selected['source_id']} must define exactly {list(ANCHORS)}"
        )

    resolved = {
        anchor: locate_anchor(raw_completion, anchors[anchor]) for anchor in ANCHORS
    }
    intervals = sorted(
        (
            data["start_char_0_based"],
            data["end_char_0_based_exclusive"],
            anchor,
        )
        for anchor, data in resolved.items()
    )
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise RuntimeError(
                f"overlapping anchors for {selected['source_id']}: "
                f"{left[2]} and {right[2]}"
            )
    actual_order = [anchor for _, _, anchor in intervals]
    if actual_order != selected.get("anchor_order"):
        raise RuntimeError(
            f"anchor order mismatch for {selected['source_id']}: "
            f"configured {selected.get('anchor_order')}, actual {actual_order}"
        )
    selected["anchors"] = resolved


def build_package(selection_path: Path, output_path: Path) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("experiment") != "three_anchor_generalization_before_after":
        raise RuntimeError("unexpected experiment name in source selection")
    if selection.get("rollouts_per_cell") != 30:
        raise RuntimeError("source selection must specify 30 rollouts per cell")
    if (selection.get("sample_seed_start"), selection.get("sample_seed_end_inclusive")) != (
        4000,
        4029,
    ):
        raise RuntimeError("source selection must reserve seeds 4000 through 4029")
    selected_sources = selection.get("sources")
    if not isinstance(selected_sources, list) or len(selected_sources) != 6:
        raise RuntimeError("source selection must contain exactly six sources")

    cache: dict[Path, list[dict[str, Any]]] = {}
    checked_hashes: dict[Path, str] = {}
    packaged: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    family_ranks: set[tuple[str, int]] = set()
    for raw_selected in selected_sources:
        selected = copy.deepcopy(raw_selected)
        source_id = selected.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("every source must have a nonempty source_id")
        if source_id in source_ids:
            raise RuntimeError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        family_rank = (selected.get("story_key"), selected.get("source_rank"))
        if family_rank in family_ranks:
            raise RuntimeError(f"duplicate family/rank label: {family_rank}")
        family_ranks.add(family_rank)

        path = resolve_source_path(selected["source_path"])
        if not path.is_file():
            raise FileNotFoundError(f"source file not found: {path}")
        if path not in checked_hashes:
            checked_hashes[path] = sha256(path)
        actual_hash = checked_hashes[path]
        if actual_hash != selected["source_sha256"]:
            raise RuntimeError(
                f"source hash changed for {path}: expected "
                f"{selected['source_sha256']}, found {actual_hash}"
            )
        if path not in cache:
            cache[path] = load_jsonl(path)
        rows = cache[path]
        record_number = selected.get("source_record_1_based")
        if not isinstance(record_number, int) or not 1 <= record_number <= len(rows):
            raise RuntimeError(f"invalid record number for {source_id}")
        source_record = rows[record_number - 1]
        verify_identity(source_record, selected)
        record = normalize_record(source_record, selected)
        resolve_anchors(record, selected)
        packaged.append({"selection": selected, "record": record})

    story_keys = [row["selection"]["story_key"] for row in packaged]
    if len(set(story_keys)) != 6:
        raise RuntimeError("expected six distinct held-out source families")
    if {"madrid", "starry_night"} & set(story_keys):
        raise RuntimeError("discovery families cannot appear in the holdout set")
    if {row["selection"]["original_result"] for row in packaged} != {
        "PASS",
        "FAIL",
    }:
        raise RuntimeError("expected both passing and failing source traces")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packaged),
        encoding="utf-8",
    )
    return packaged


def main() -> int:
    args = parse_args()
    packaged = build_package(args.selection, args.output)
    print(f"Verified and packaged {len(packaged)} sources: {args.output}")
    for row in packaged:
        selected = row["selection"]
        cuts = ", ".join(
            f"{name}={selected['anchors'][name]['start_line_1_based']}–"
            f"{selected['anchors'][name]['end_line_1_based']}"
            for name in ANCHORS
        )
        print(
            f"- {selected['source_label']}: order "
            f"{' < '.join(selected['anchor_order'])}; {cuts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

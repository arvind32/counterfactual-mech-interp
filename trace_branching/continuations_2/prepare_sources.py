#!/usr/bin/env python3
"""Package the pinned source traces used by the Continuations 2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
DEFAULT_SELECTION = ROOT / "source_selection.json"
DEFAULT_OUTPUT = ROOT / "source_records.jsonl"
LOCATIONS = ("A_STAR", "A_STAR_STAR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and package the configured baseline source traces."
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


def verify_identity(record: dict[str, Any], selected: dict[str, Any]) -> None:
    for field in ("case_name", "condition"):
        if record.get(field) != selected.get(field):
            raise RuntimeError(
                f"source identity mismatch for {selected['source_id']}: {field}"
            )
    if str(record.get("seed")) != str(selected.get("seed")):
        raise RuntimeError(
            f"source identity mismatch for {selected['source_id']}: seed"
        )


def verify_anchors(record: dict[str, Any], selected: dict[str, Any]) -> None:
    raw_completion = record.get("raw_completion")
    if not isinstance(raw_completion, str) or not raw_completion:
        raise RuntimeError(f"missing raw_completion for {selected['source_id']}")
    lines = raw_completion.splitlines()
    previous = 0
    for location in LOCATIONS:
        anchor = selected.get("anchors", {}).get(location, {})
        line_number = anchor.get("raw_completion_line_1_based")
        if not isinstance(line_number, int) or not previous < line_number <= len(lines):
            raise RuntimeError(
                f"invalid {location} line number for {selected['source_id']}"
            )
        actual = lines[line_number - 1]
        if actual != anchor.get("text"):
            raise RuntimeError(
                f"{location} anchor mismatch for {selected['source_id']}\n"
                f"expected: {anchor.get('text')!r}\n"
                f"found:    {actual!r}"
            )
        previous = line_number


def build_package(selection_path: Path, output_path: Path) -> list[dict[str, Any]]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_sources = selection.get("sources")
    if not isinstance(selected_sources, list) or not selected_sources:
        raise RuntimeError("source selection must contain at least one source")

    cache: dict[Path, list[dict[str, Any]]] = {}
    checked_hashes: dict[Path, str] = {}
    packaged: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for selected in selected_sources:
        source_id = selected.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("every source must have a nonempty source_id")
        if source_id in source_ids:
            raise RuntimeError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)

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
        record = rows[record_number - 1]
        verify_identity(record, selected)
        verify_anchors(record, selected)
        packaged.append({"selection": selected, "record": record})

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
            f"{name}={selected['anchors'][name]['raw_completion_line_1_based']}"
            for name in LOCATIONS
        )
        print(f"- {selected['source_label']}: {cuts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

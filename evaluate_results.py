#!/usr/bin/env python3
"""Evaluate counterfactual story edits in the terminal or with an LLM judge.

Human evaluation in the terminal is the default. An OpenAI structured-output
judge is available as an explicit opt-in. The script computes the overall
result: PASS only for Yes / Yes / Yes / No.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RUBRIC_VERSION = "counterfactual_edit_v3"

QUESTION_LABELS = (
    (
        "q1_intervention_applied",
        "1. Requested intervention applied?",
    ),
    (
        "q2_downstream_consequences",
        "2. Downstream consequences updated correctly?",
    ),
    (
        "q3_invariants_preserved",
        "3. Background factors and unrelated details preserved?",
    ),
)

Q4_FIELD_NAME = "q4_other_incoherence"
Q4_LABEL = "4. Otherwise incoherent or inconsistent?"

SYSTEM_PROMPT = """You are a careful evaluator of counterfactual story edits.

Evaluate only the model's final edited story. Do not evaluate its hidden
reasoning or writing style. Treat the supplied expected edit and reasoning as
the reference interpretation of the item, while allowing semantically
equivalent wording.

Answer these four questions independently:

1. Requested intervention applied?
   - Yes: the edited story correctly and fully makes the change explicitly
     stipulated by the instruction.
   - Partial: the response attempts the intervention, but applies it only
     incompletely or ambiguously.
   - No: the intervention is absent or incorrectly applied.

2. Downstream consequences updated correctly?
   - Yes: all consequences of the intervention identified by the reference
     annotations are updated consistently, holding explicit and implicit
     background factors fixed.
   - Partial: some, but not all, required downstream consequences are updated
     correctly.
   - No: the required downstream consequences are not updated correctly.

3. Background factors and unrelated details preserved?
   - Yes: the response preserves facts that should remain fixed, makes no
     unsupported extra changes, and does not alter a background factor
     anti-causally merely to preserve the original outcome.
   - Partial: most relevant facts are preserved, but there is a limited
     omission, unsupported change, or ambiguity.
   - No: important upstream/background facts or unrelated details are changed
     incorrectly.

4. Otherwise incoherent or inconsistent?
   - Yes: the edited story has an additional internal contradiction,
     incoherence, or nonsensical content not adequately captured by questions
     1-3.
   - No: it has no such additional problem.
   Do not answer Yes merely because one of questions 1-3 failed.

Also report your confidence in the evaluation as exactly H, M, or L, where H
means high, M means medium, and L means low. Add concise free-text comments
only when useful; otherwise return an empty string.

Do not assign an overall grade or a single failure type. The caller computes
the overall result.
"""


class EditEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q1_intervention_applied: Literal["Yes", "No", "Partial"]
    q2_downstream_consequences: Literal["Yes", "No", "Partial"]
    q3_invariants_preserved: Literal["Yes", "No", "Partial"]
    q4_other_incoherence: Literal["Yes", "No"]
    confidence: Literal["H", "M", "L"]
    comments: str = Field(
        description="Optional comments about the evaluation; empty string if none."
    )


CSV_FIELDS = [
    "example_id",
    "row_number",
    "model",
    "revision",
    "condition",
    "generation_profile",
    "thinking_enabled",
    "thinking_status",
    "generation_tags",
    "story",
    "query",
    "expectation",
    "knowledge_reasoning",
    "response",
    "q1_intervention_applied",
    "q2_downstream_consequences",
    "q3_invariants_preserved",
    "q4_other_incoherence",
    "confidence",
    "comments",
    "failed_checks",
    "overall_result",
    "evaluation_status",
    "evaluation_error",
    "evaluator_type",
    "evaluator_model",
    "evaluator_response_id",
    "rubric_version",
    "evaluated_at_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen JSONL results and write one resumable CSV row "
            "per answer."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL results file produced by run_dataset.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Evaluation CSV. Defaults to evaluations/<input-stem>-evaluation.csv."
        ),
    )
    parser.add_argument(
        "--evaluator",
        choices=["human", "openai"],
        default="human",
        help=(
            "Evaluation method. Defaults to interactive human review in the "
            "terminal; use 'openai' to opt into API evaluation."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.6",
        help="OpenAI model used when --evaluator openai is selected.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many unfinished records.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API attempts per record.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first unfinished evaluation prompt without calling the API.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV instead of resuming it.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return Path("evaluations") / f"{input_path.stem}-evaluation.csv"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if "example_id" not in record:
                raise ValueError(
                    f"Missing example_id in {path} at line {line_number}."
                )
            records.append(record)
    return records


def record_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("example_id", "")),
        str(record.get("model", "")),
        str(record.get("condition", "")),
    )


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()

    completed = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                f"Existing CSV schema does not match {RUBRIC_VERSION}: {path}. "
                "Choose a new --output path or use --overwrite."
            )
        for row in reader:
            if row.get("evaluation_status") in {"evaluated", "no_response"}:
                completed.add(record_key(row))
    return completed


def build_user_prompt(record: dict) -> str:
    response = record.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Cannot build an evaluation prompt without a response.")

    return f"""Original story:
{record.get('story', '')}

Counterfactual editing instruction:
{record.get('query', '')}

Reference expected edit:
{record.get('expectation', '')}

Reference background knowledge and reasoning:
{record.get('knowledge_reasoning', '')}

Model's final edited story:
{response.strip()}
"""


class EvaluationAborted(Exception):
    """Stop interactive review without losing previously written rows."""


def print_case(record: dict, position: int, total: int) -> None:
    print("\n" + "=" * 88)
    print(
        f"CASE {position}/{total} | {record.get('example_id', '')} | "
        f"spreadsheet row {record.get('row_number', '')}"
    )
    print("=" * 88)
    print("\nORIGINAL STORY\n")
    print(record.get("story", ""))
    print("\nCOUNTERFACTUAL INSTRUCTION\n")
    print(record.get("query", ""))
    print("\nREFERENCE EXPECTED EDIT\n")
    print(record.get("expectation", ""))
    print("\nREFERENCE REASONING\n")
    print(record.get("knowledge_reasoning", ""))
    print("\nMODEL'S FINAL EDITED STORY\n")
    print(record.get("response", ""))
    print("\n" + "-" * 88)


def read_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt) as error:
        print()
        raise EvaluationAborted from error


def prompt_yes_no_partial(label: str) -> Literal["Yes", "No", "Partial"]:
    while True:
        value = read_input(
            f"\n{label} [y/n/p; q to save and quit]: "
        ).lower()
        if value in {"y", "yes"}:
            return "Yes"
        if value in {"n", "no"}:
            return "No"
        if value in {"p", "partial"}:
            return "Partial"
        if value in {"q", "quit", "exit"}:
            raise EvaluationAborted
        print("Please enter y, n, p, or q.")


def prompt_yes_no(label: str) -> Literal["Yes", "No"]:
    while True:
        value = read_input(
            f"\n{label} [y/n; q to save and quit]: "
        ).lower()
        if value in {"y", "yes"}:
            return "Yes"
        if value in {"n", "no"}:
            return "No"
        if value in {"q", "quit", "exit"}:
            raise EvaluationAborted
        print("Please enter y, n, or q.")


def prompt_confidence() -> Literal["H", "M", "L"]:
    while True:
        value = read_input("\nConfidence [h/m/l; q to save and quit]: ").lower()
        if value in {"h", "high"}:
            return "H"
        if value in {"m", "medium"}:
            return "M"
        if value in {"l", "low"}:
            return "L"
        if value in {"q", "quit", "exit"}:
            raise EvaluationAborted
        print("Please enter h, m, l, or q.")


def prompt_comments() -> str:
    return read_input("Comments (optional; press Enter to skip): ")


def call_human_judge(
    record: dict,
    position: int,
    total: int,
) -> tuple[EditEvaluation, str]:
    print_case(record, position, total)
    judgments: dict[str, str] = {}
    for field_name, label in QUESTION_LABELS:
        judgments[field_name] = prompt_yes_no_partial(label)

    judgments[Q4_FIELD_NAME] = prompt_yes_no(Q4_LABEL)
    judgments["confidence"] = prompt_confidence()
    judgments["comments"] = prompt_comments()
    return EditEvaluation(**judgments), ""


def compute_grade(evaluation: EditEvaluation) -> tuple[str, str]:
    answers = (
        evaluation.q1_intervention_applied,
        evaluation.q2_downstream_consequences,
        evaluation.q3_invariants_preserved,
        evaluation.q4_other_incoherence,
    )
    passed = answers == ("Yes", "Yes", "Yes", "No")

    failed = []
    if answers[0] != "Yes":
        failed.append("Q1_intervention_not_fully_applied")
    if answers[1] != "Yes":
        failed.append("Q2_downstream_consequences_not_fully_correct")
    if answers[2] != "Yes":
        failed.append("Q3_invariants_not_fully_preserved")
    if answers[3] != "No":
        failed.append("Q4_other_incoherence")

    return ("PASS" if passed else "FAIL"), "|".join(failed)


def base_csv_row(
    record: dict,
    evaluator_type: str,
    evaluator_model: str,
) -> dict[str, str]:
    return {
        "example_id": str(record.get("example_id", "")),
        "row_number": str(record.get("row_number", "")),
        "model": str(record.get("model", "")),
        "revision": str(record.get("revision", "")),
        "condition": str(record.get("condition", "")),
        "generation_profile": str(record.get("generation_profile", "")),
        "thinking_enabled": str(record.get("thinking_enabled", "")),
        "thinking_status": str(record.get("thinking_status", "")),
        "generation_tags": json.dumps(
            record.get("tags", []), ensure_ascii=False
        ),
        "story": str(record.get("story", "")),
        "query": str(record.get("query", "")),
        "expectation": str(record.get("expectation", "")),
        "knowledge_reasoning": str(record.get("knowledge_reasoning", "")),
        "response": "" if record.get("response") is None else str(record["response"]),
        "q1_intervention_applied": "",
        "q2_downstream_consequences": "",
        "q3_invariants_preserved": "",
        "q4_other_incoherence": "",
        "confidence": "",
        "comments": "",
        "failed_checks": "",
        "overall_result": "FAIL",
        "evaluation_status": "",
        "evaluation_error": "",
        "evaluator_type": evaluator_type,
        "evaluator_model": evaluator_model,
        "evaluator_response_id": "",
        "rubric_version": RUBRIC_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluated_csv_row(
    record: dict,
    evaluation: EditEvaluation,
    evaluator_type: str,
    evaluator_model: str,
    response_id: str,
) -> dict[str, str]:
    overall_result, failed_checks = compute_grade(evaluation)
    row = base_csv_row(record, evaluator_type, evaluator_model)
    row.update(
        {
            "q1_intervention_applied": evaluation.q1_intervention_applied,
            "q2_downstream_consequences": evaluation.q2_downstream_consequences,
            "q3_invariants_preserved": evaluation.q3_invariants_preserved,
            "q4_other_incoherence": evaluation.q4_other_incoherence,
            "confidence": evaluation.confidence,
            "comments": evaluation.comments,
            "failed_checks": failed_checks,
            "overall_result": overall_result,
            "evaluation_status": "evaluated",
            "evaluator_response_id": response_id,
        }
    )
    return row


def no_response_csv_row(
    record: dict,
    evaluator_type: str,
    evaluator_model: str,
) -> dict[str, str]:
    row = base_csv_row(record, evaluator_type, evaluator_model)
    row.update(
        {
            "failed_checks": "NO_COMPLETED_RESPONSE",
            "evaluation_status": "no_response",
            "evaluation_error": (
                "The generation record has no completed final response; "
                "no judge API call was made."
            ),
        }
    )
    return row


def error_csv_row(
    record: dict,
    evaluator_type: str,
    evaluator_model: str,
    error: Exception,
) -> dict[str, str]:
    row = base_csv_row(record, evaluator_type, evaluator_model)
    row.update(
        {
            "failed_checks": "EVALUATION_ERROR",
            "evaluation_status": "error",
            "evaluation_error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def append_csv_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def call_judge(client, judge_model: str, record: dict) -> tuple[EditEvaluation, str]:
    api_response = client.responses.parse(
        model=judge_model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        text_format=EditEvaluation,
    )
    evaluation = api_response.output_parsed
    if evaluation is None:
        raise RuntimeError("Judge returned no parsed evaluation.")
    return evaluation, str(api_response.id)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else default_output_path(input_path)
    )
    evaluator_model = (
        "human_terminal" if args.evaluator == "human" else args.judge_model
    )

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1.")
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input JSONL: {input_path}")

    if args.overwrite and output_path.exists():
        output_path.unlink()

    records = load_jsonl(input_path)
    already_completed = completed_keys(output_path)
    pending = [
        record
        for record in records
        if record_key(record) not in already_completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        print("No unfinished evaluations remain.")
        print("Evaluation CSV:", output_path)
        return 0

    print("Input:", input_path)
    print("Output:", output_path)
    print("Evaluator:", args.evaluator)
    if args.evaluator == "openai":
        print("Judge model:", args.judge_model)
    print("Rubric:", RUBRIC_VERSION)
    print("Pending records:", len(pending))

    if args.dry_run:
        first_with_response = next(
            (
                record
                for record in pending
                if isinstance(record.get("response"), str)
                and record["response"].strip()
            ),
            None,
        )
        if first_with_response is None:
            print("No pending record has a completed final response.")
            return 0
        if args.evaluator == "human":
            print_case(first_with_response, 1, 1)
        else:
            print("\nSYSTEM PROMPT\n")
            print(SYSTEM_PROMPT)
            print("\nUSER PROMPT\n")
            print(build_user_prompt(first_with_response))
        print("\nDry run only; no evaluation or CSV write was made.")
        return 0

    client = None
    if args.evaluator == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before using "
                "--evaluator openai. Never save the key in this repository."
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The OpenAI Python package is not installed. Run: "
                "python -m pip install --upgrade openai pydantic"
            ) from error
        client = OpenAI()

    stopped_early = False
    for position, record in enumerate(pending, start=1):
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            append_csv_row(
                output_path,
                no_response_csv_row(
                    record,
                    args.evaluator,
                    evaluator_model,
                ),
            )
            print(
                f"[{position}/{len(pending)}] {record['example_id']}: "
                "FAIL (no completed response; evaluation skipped)"
            )
            continue

        if args.evaluator == "human":
            try:
                evaluation, response_id = call_human_judge(
                    record,
                    position,
                    len(pending),
                )
            except EvaluationAborted:
                stopped_early = True
                print("\nStopped. Previously completed rows are saved.")
                break

            row = evaluated_csv_row(
                record,
                evaluation,
                args.evaluator,
                evaluator_model,
                response_id,
            )
            append_csv_row(output_path, row)
            print(f"\nRESULT: {row['overall_result']}")
            if row["failed_checks"]:
                print("FAILED CHECKS:", row["failed_checks"])
            continue

        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                evaluation, response_id = call_judge(
                    client,
                    args.judge_model,
                    record,
                )
                row = evaluated_csv_row(
                    record,
                    evaluation,
                    args.evaluator,
                    evaluator_model,
                    response_id,
                )
                append_csv_row(output_path, row)
                print(
                    f"[{position}/{len(pending)}] {record['example_id']}: "
                    f"{row['overall_result']}"
                )
                last_error = None
                break
            except Exception as error:  # API/transport/validation failures
                last_error = error
                if attempt < args.max_retries:
                    delay = 2 ** (attempt - 1)
                    print(
                        f"Attempt {attempt} failed for {record['example_id']}: "
                        f"{error}. Retrying in {delay}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)

        if last_error is not None:
            append_csv_row(
                output_path,
                error_csv_row(
                    record,
                    args.evaluator,
                    evaluator_model,
                    last_error,
                ),
            )
            print(
                f"[{position}/{len(pending)}] {record['example_id']}: "
                "evaluation error saved; it will be retried next run",
                file=sys.stderr,
            )

    if not stopped_early:
        print("\nFinished.")
    print("Evaluation CSV:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

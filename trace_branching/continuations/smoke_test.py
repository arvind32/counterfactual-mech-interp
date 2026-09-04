#!/usr/bin/env python3
"""Fail-fast validation for the premise-acceptance continuation experiment."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import traceback
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent
PROJECT_ROOT = TRACE_ROOT.parent
REPORT_PATH = ROOT / "smoke_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sanitize(value: object) -> str:
    """Remove machine-specific home and project paths from diagnostic text."""
    text = str(value)
    replacements = [
        (str(Path.home()), "~"),
        (str(PROJECT_ROOT), "<project>"),
    ]
    for original, replacement in replacements:
        if original:
            text = text.replace(original, replacement)
    return text


def main() -> int:
    report = {
        "result": "FAIL",
        "failed_stage": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "checks": [],
    }

    def check(name: str, detail: str) -> None:
        report["checks"].append({"name": name, "result": "PASS", "detail": detail})
        print(f"[PASS] {name}: {detail}", flush=True)

    def stage(name: str) -> None:
        report["failed_stage"] = name
        print(f"\n--- {name} ---", flush=True)

    print("PREMISE-ACCEPTANCE FAIL-FAST SMOKE TEST", flush=True)
    try:
        stage("1. Package integrity")
        required = [
            TRACE_ROOT / "run_ab.py",
            TRACE_ROOT / "history_presence.py",
            ROOT / "source_records.jsonl",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing required experiment files: {missing}")
        sources = load_jsonl(ROOT / "source_records.jsonl")
        if len(sources) != 6:
            raise RuntimeError(
                f"expected 6 pinned failed sources, found {len(sources)}"
            )
        check("package", "required files; 6 pinned failed sources across 3 stories")

        stage("2. Python, CUDA, cache paths, and exact vLLM version")
        cache_environment = {
            name: os.environ.get(name)
            for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
            if os.environ.get(name)
        }
        malformed = {
            name for name, value in cache_environment.items() if value.startswith("/~")
        }
        if malformed:
            raise RuntimeError(
                "malformed Hugging Face cache path in: " + ", ".join(sorted(malformed))
            )
        report["cache_variables_set"] = sorted(cache_environment)

        import run_acceptance_continuations as runner

        base_jobs, sources, jobs = runner.load_design(ROOT / "source_records.jsonl")
        runner.validate_design(base_jobs, jobs)
        check(
            "design",
            (
                "360 jobs: 240 premise-acceptance branches plus 120 "
                "cut-matched unaltered controls"
            ),
        )

        import torch
        from importlib.metadata import version

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see a CUDA GPU")
        vllm_version = version("vllm")
        report.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "vllm": vllm_version,
                "expected_vllm": runner.base_runner.EXPECTED_VLLM_VERSION,
            }
        )
        if vllm_version != runner.base_runner.EXPECTED_VLLM_VERSION:
            raise RuntimeError(
                "vLLM mismatch: expected "
                f"{runner.base_runner.EXPECTED_VLLM_VERSION}; found {vllm_version}"
            )
        check(
            "environment",
            f"vLLM {vllm_version}; torch {torch.__version__}; {report['gpu']}; "
            f"cache variables {sorted(cache_environment) or 'defaults'}",
        )

        stage("3. Load the pinned Qwen model and custom processor")
        llm, SamplingParams, _ = runner.base_runner.build_engine(0.90)
        tokenizer = llm.get_tokenizer()
        check(
            "engine",
            f"{runner.base_runner.MODEL} at revision "
            f"{runner.base_runner.REVISION[:12]}",
        )

        stage("4. Test presence-history restoration")
        runner.base_runner.runtime_smoke(llm, tokenizer, SamplingParams)
        check(
            "presence history",
            "GPU arithmetic and request-level processor dispatch are correct",
        )

        stage("5. Reconstruct all augmented branch prefixes")
        prepared = runner.prepare_jobs(jobs, sources, tokenizer)
        if len(prepared) != runner.EXPECTED_JOBS:
            raise RuntimeError(
                f"expected {runner.EXPECTED_JOBS} prepared jobs, found {len(prepared)}"
            )
        for item in prepared:
            prompt_ids = item["prompt_ids"]
            if not prompt_ids or not all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in prompt_ids
            ):
                raise TypeError(
                    f"non-integer prompt token in {item['job']['job_id']}"
                )
            if (
                item["job"]["intervention_type"] == "premise_acceptance"
                and not item["job"]["raw_prefix"].endswith(
                    item["job"]["insertion_line"]
                )
            ):
                raise RuntimeError(
                    f"candidate placement failed for {item['job']['job_id']}"
                )
        budgets = {}
        for item in prepared:
            budgets[item["job"]["story_family"]] = item["max_tokens"]
        report["continuation_budgets"] = budgets
        check("prefixes", f"360/360 exact round trips; budgets {budgets}")

        stage("6. Generate from an actual Harbor-A-M1 branch")
        item = next(
            item
            for item in prepared
            if item["job"]["case_name"] == "a_names"
            and item["job"]["cut"] == "A"
            and item["job"]["candidate_id"] == "M1"
        )
        params = runner.base_runner.sampling_params(
            SamplingParams,
            seed=909090,
            max_tokens=64,
            prefix_ids=item["prefix_ids"],
        )
        output = llm.generate(
            [{"prompt_token_ids": item["prompt_ids"]}],
            sampling_params=[params],
            use_tqdm=False,
        )[0].outputs[0]
        token_ids = list(output.token_ids)
        if not token_ids:
            raise RuntimeError("actual augmented branch produced zero continuation tokens")
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        report["actual_branch"] = {
            "job_id": item["job"]["job_id"],
            "generated_tokens": len(token_ids),
            "finish_reason": output.finish_reason,
            "text_preview": text[:160],
        }
        check(
            "actual branch",
            f"{len(token_ids)} tokens from {item['job']['job_id']}",
        )

        report["result"] = "PASS"
        report["failed_stage"] = None
    except BaseException as error:
        report["error_type"] = type(error).__name__
        report["error"] = sanitize(error)
        report["traceback_tail"] = [
            sanitize(line) for line in traceback.format_exc().splitlines()[-12:]
        ]
        print(
            f"\n[FAIL] {report['failed_stage']}: "
            f"{type(error).__name__}: {sanitize(error)}",
            flush=True,
        )
    finally:
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("\n--- BEGIN SMOKE REPORT ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("--- END SMOKE REPORT ---")
        print("Saved: trace_branching_pilot/continuations/smoke_report.json")

    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

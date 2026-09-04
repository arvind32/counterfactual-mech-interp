#!/usr/bin/env python3
"""Fail-fast validation for the Continuations 2 control-rollout experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import traceback


ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent
PROJECT_ROOT = TRACE_ROOT.parent
REPORT_PATH = ROOT / "smoke_report.json"


def sanitize(value: object) -> str:
    text = str(value)
    for original, replacement in (
        (str(Path.home()), "~"),
        (str(PROJECT_ROOT), "<project>"),
    ):
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

    def stage(name: str) -> None:
        report["failed_stage"] = name
        print(f"\n--- {name} ---", flush=True)

    def passed(name: str, detail: str) -> None:
        report["checks"].append(
            {"name": name, "result": "PASS", "detail": detail}
        )
        print(f"[PASS] {name}: {detail}", flush=True)

    print("CONTINUATIONS 2 FAIL-FAST SMOKE TEST", flush=True)
    try:
        stage("1. Package and source integrity")
        required = [
            TRACE_ROOT / "run_ab.py",
            TRACE_ROOT / "history_presence.py",
            ROOT / "source_selection.json",
            ROOT / "prepare_sources.py",
            ROOT / "run_control_rollouts.py",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing required files: {missing}")

        import prepare_sources

        packaged = prepare_sources.build_package(
            ROOT / "source_selection.json", ROOT / "source_records.jsonl"
        )
        passed(
            "sources",
            f"{len(packaged)} pinned baseline_p5 traces; exact files, records, hashes, and cuts",
        )

        stage("2. Complete rollout design")
        import run_control_rollouts as runner

        sources = runner.load_jsonl(ROOT / "source_records.jsonl")
        jobs = runner.build_jobs(sources)
        runner.validate_design(jobs)
        expected_jobs = (
            len(sources) * len(runner.LOCATIONS) * runner.ROLLOUTS_PER_CELL
        )
        passed(
            "design",
            f"{expected_jobs} controls: {len(sources)} traces × "
            f"{len(runner.LOCATIONS)} locations × "
            f"{runner.ROLLOUTS_PER_CELL} new continuations",
        )

        stage("3. Python, CUDA, cache paths, and exact vLLM version")
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
                "malformed Hugging Face cache path in: "
                + ", ".join(sorted(malformed))
            )
        report["cache_variables_set"] = sorted(cache_environment)

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
        passed(
            "environment",
            f"vLLM {vllm_version}; torch {torch.__version__}; {report['gpu']}; "
            f"cache variables {sorted(cache_environment) or 'defaults'}",
        )

        stage("4. Load the pinned Qwen model and custom processor")
        llm, SamplingParams, _ = runner.base_runner.build_engine(0.90)
        tokenizer = llm.get_tokenizer()
        passed(
            "engine",
            f"{runner.base_runner.MODEL} at revision "
            f"{runner.base_runner.REVISION[:12]}",
        )

        stage("5. Test presence-history restoration")
        runner.base_runner.runtime_smoke(llm, tokenizer, SamplingParams)
        passed(
            "presence history",
            "GPU arithmetic and request-level processor dispatch are correct",
        )

        stage("6. Reconstruct every branch prefix")
        prepared = runner.prepare_jobs(jobs, sources, tokenizer)
        if len(prepared) != expected_jobs:
            raise RuntimeError(
                f"expected {expected_jobs} prepared jobs, found {len(prepared)}"
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
            if not item["record"]["raw_completion"].startswith(
                item["job"]["raw_prefix"]
            ):
                raise RuntimeError(
                    f"prefix is not exact for {item['job']['job_id']}"
                )
        budgets: dict[str, int] = {}
        for item in prepared:
            budgets[item["job"]["story_family"]] = item["max_tokens"]
        report["continuation_budgets"] = budgets
        passed(
            "prefixes",
            f"{expected_jobs}/{expected_jobs} exact token round trips; budgets {budgets}",
        )

        stage("7. Generate from an actual A** branch")
        item = next(
            item
            for item in prepared
            if item["job"]["source_rank"] == 1
            and item["job"]["story_key"] == "madrid"
            and item["job"]["location"] == "A_STAR_STAR"
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
            raise RuntimeError("actual branch produced zero continuation tokens")
        preview = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        report["actual_branch"] = {
            "job_id": item["job"]["job_id"],
            "generated_tokens": len(token_ids),
            "finish_reason": output.finish_reason,
            "text_preview": preview[:160],
        }
        passed(
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
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("\n--- BEGIN SMOKE REPORT ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("--- END SMOKE REPORT ---")
        print("Saved: trace_branching_pilot/continuations_2/smoke_report.json")

    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-fast validation for the trace-branching package and GPU environment."""

from __future__ import annotations

import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "smoke_report.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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

    print("TRACE-BRANCHING FAIL-FAST SMOKE TEST", flush=True)
    try:
        stage("1. Package integrity")
        sources = load_jsonl(ROOT / "source_records.jsonl")
        jobs = load_jsonl(ROOT / "ab_jobs.jsonl")
        if len(sources) != 6 or len(jobs) != 60:
            raise RuntimeError(f"expected 6 sources/60 jobs; found {len(sources)}/{len(jobs)}")
        cells = {}
        for job in jobs:
            key = (job["story_family"], job["cut"])
            cells[key] = cells.get(key, 0) + 1
            source = sources[job["source_index"]]["record"]
            if not source["raw_completion"].startswith(job["raw_prefix"]):
                raise RuntimeError(f"stored prefix mismatch: {job['job_id']}")
            if job["raw_prefix"].splitlines()[-1] != job["cut_line_text"]:
                raise RuntimeError(f"stored cut-line mismatch: {job['job_id']}")
        if len(cells) != 6 or set(cells.values()) != {10}:
            raise RuntimeError(f"unexpected A/B cell counts: {cells}")
        check("package", "6 pinned sources; 3 stories × 2 cuts × 10 continuations")

        stage("2. Python, CUDA, and exact vLLM version")
        cache_environment = {
            name: os.environ.get(name)
            for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
            if os.environ.get(name)
        }
        report["cache_environment"] = cache_environment
        malformed = {
            name: value
            for name, value in cache_environment.items()
            if value.startswith("/~")
        }
        if malformed:
            raise RuntimeError(f"malformed Hugging Face cache path(s): {malformed}")
        import torch
        from importlib.metadata import version
        import run_ab
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see a CUDA GPU")
        vllm_version = version("vllm")
        report.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vllm": vllm_version,
            "expected_vllm": run_ab.EXPECTED_VLLM_VERSION,
        })
        if vllm_version != run_ab.EXPECTED_VLLM_VERSION:
            raise RuntimeError(
                f"vLLM mismatch: expected {run_ab.EXPECTED_VLLM_VERSION}; found {vllm_version}"
            )
        check(
            "environment",
            f"vLLM {vllm_version}; torch {torch.__version__}; {report['gpu']}; "
            f"cache {cache_environment or 'Hugging Face default'}",
        )

        stage("3. Load the pinned Qwen model and custom processor")
        llm, SamplingParams, _ = run_ab.build_engine(0.90)
        tokenizer = llm.get_tokenizer()
        check("engine", f"{run_ab.MODEL} at revision {run_ab.REVISION[:12]}")

        stage("4. Test presence-history restoration")
        run_ab.runtime_smoke(llm, tokenizer, SamplingParams)
        check(
            "presence history",
            "GPU arithmetic correct; empty history disables and nonempty history enables the processor",
        )

        stage("5. Reconstruct every real branch prefix")
        prepared = run_ab.prepare_jobs(jobs, sources, tokenizer)
        counts = {
            item["job"]["story_family"]: item["max_tokens"] for item in prepared
        }
        report["continuation_budgets"] = counts
        check("prefixes", f"60/60 round trips exact; budgets {counts}")

        stage("6. Generate from an actual Harbor A branch")
        item = prepared[0]
        params = run_ab.sampling_params(
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
        check("actual branch", f"{len(token_ids)} tokens from {item['job']['job_id']}")

        report["result"] = "PASS"
        report["failed_stage"] = None
    except BaseException as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        print(f"\n[FAIL] {report['failed_stage']}: {type(error).__name__}: {error}", flush=True)
    finally:
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("\n--- BEGIN SMOKE REPORT ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("--- END SMOKE REPORT ---")
        print(f"Saved: {REPORT_PATH}")

    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

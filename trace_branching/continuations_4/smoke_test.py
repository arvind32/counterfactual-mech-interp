#!/usr/bin/env python3
"""Fail-fast validation for the held-out-family three-anchor experiment."""

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
        (str(PROJECT_ROOT), "<project>"),
        (str(Path.home()), "~"),
    ):
        if original:
            text = text.replace(original, replacement)
    return text


def main() -> int:
    report: dict[str, object] = {
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
        checks = report["checks"]
        assert isinstance(checks, list)
        checks.append({"name": name, "result": "PASS", "detail": detail})
        print(f"[PASS] {name}: {detail}", flush=True)

    print("THREE-ANCHOR GENERALIZATION FAIL-FAST SMOKE TEST", flush=True)
    try:
        stage("1. Package and source integrity")
        required = [
            TRACE_ROOT / "run_ab.py",
            TRACE_ROOT / "history_presence.py",
            ROOT / "source_selection.json",
            ROOT / "prepare_sources.py",
            ROOT / "run_generalization_rollouts.py",
            ROOT / "inspect_results.py",
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
            "6 pinned baseline_p5 traces; exact files, records, hashes, and anchor spans",
        )

        stage("2. Complete paired rollout design")
        import run_generalization_rollouts as runner

        sources = runner.load_jsonl(ROOT / "source_records.jsonl")
        jobs = runner.build_jobs(sources)
        runner.validate_design(jobs)
        expected_jobs = 6 * 3 * 2 * 30
        expected_pairs = 6 * 3 * 30
        if len(jobs) != expected_jobs:
            raise RuntimeError(f"expected {expected_jobs} jobs, found {len(jobs)}")
        passed(
            "design",
            f"{expected_jobs} continuations in {expected_pairs} matched-seed "
            "before/after pairs; seeds 4000–4029",
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

        stage("6. Reconstruct and compare every branch prefix")
        prepared = runner.prepare_jobs(jobs, sources, tokenizer)
        if len(prepared) != expected_jobs:
            raise RuntimeError(
                f"expected {expected_jobs} prepared jobs, found {len(prepared)}"
            )
        states: dict[tuple[str, str, str], dict[str, object]] = {}
        for item in prepared:
            job = item["job"]
            key = (job["source_id"], job["anchor"], job["branch"])
            states.setdefault(key, item)
            prompt_ids = item["prompt_ids"]
            if not prompt_ids or not all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in prompt_ids
            ):
                raise TypeError(f"non-integer prompt token in {job['job_id']}")
            if not item["record"]["raw_completion"].startswith(job["raw_prefix"]):
                raise RuntimeError(f"prefix is not exact for {job['job_id']}")
        if len(states) != 36:
            raise RuntimeError(f"expected 36 unique branch states, found {len(states)}")
        for source in sources:
            source_id = source["selection"]["source_id"]
            for anchor in runner.ANCHORS:
                before = states[(source_id, anchor, "BEFORE")]["job"]
                after = states[(source_id, anchor, "AFTER")]["job"]
                added = after["raw_prefix"][len(before["raw_prefix"]) :]
                if added != before["inserted_text"]:
                    raise RuntimeError(
                        f"before/after difference is incorrect for {source_id}/{anchor}"
                    )
        budgets: dict[str, int] = {}
        for item in states.values():
            budgets[item["job"]["story_family"]] = item["max_tokens"]
        report["continuation_budgets"] = budgets
        passed(
            "prefixes",
            f"36/36 exact unique prefix round trips; before/after differences "
            f"match only their natural anchor spans; budgets {budgets}",
        )

        stage("7. Generate briefly from all 36 branch states")
        state_items = list(states.values())
        prompts = [{"prompt_token_ids": item["prompt_ids"]} for item in state_items]
        params = [
            runner.base_runner.sampling_params(
                SamplingParams,
                seed=930000 + index,
                max_tokens=16,
                prefix_ids=item["prefix_ids"],
            )
            for index, item in enumerate(state_items)
        ]
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
        if len(outputs) != len(state_items):
            raise RuntimeError("vLLM returned a different number of smoke outputs")
        previews: list[dict[str, object]] = []
        for item, output in zip(state_items, outputs):
            token_ids = list(output.outputs[0].token_ids)
            if not token_ids:
                raise RuntimeError(
                    f"zero continuation tokens for {item['job']['source_id']}/"
                    f"{item['job']['anchor']}/{item['job']['branch']}"
                )
            preview = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            previews.append(
                {
                    "source_id": item["job"]["source_id"],
                    "anchor": item["job"]["anchor"],
                    "branch": item["job"]["branch"],
                    "generated_tokens": len(token_ids),
                    "text_preview": preview[:80],
                }
            )
        report["actual_branches"] = previews
        passed("actual branches", "all 36 source/anchor/branch states generated tokens")

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
        print("Saved: trace_branching_pilot/continuation_4/smoke_report.json")

    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

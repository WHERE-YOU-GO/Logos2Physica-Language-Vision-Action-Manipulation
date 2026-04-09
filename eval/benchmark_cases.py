from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.metrics import task_success_rate

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class BenchmarkCase:
    case_id: str
    prompt: str
    expected_source: str
    expected_target: str | None = None


def load_benchmark_cases(path: str) -> list[BenchmarkCase]:
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"Benchmark case file does not exist: {path_obj}")

    if path_obj.suffix.lower() == ".json":
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load non-JSON benchmark cases.")
        payload = yaml.safe_load(path_obj.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError("Benchmark case file must contain a list of cases.")

    cases = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each benchmark case must be a mapping.")
        cases.append(
            BenchmarkCase(
                case_id=str(item["case_id"]),
                prompt=str(item["prompt"]),
                expected_source=str(item["expected_source"]),
                expected_target=None if item.get("expected_target") is None else str(item["expected_target"]),
            )
        )
    return cases


def run_benchmark(cases: list[BenchmarkCase], fsm) -> dict:
    results: list[dict[str, Any]] = []
    success_flags: list[bool] = []

    for case in cases:
        run_result = fsm.run_once(case.prompt)
        resolved = run_result.get("resolved_ids", {})
        source_ok = resolved.get("source_id") == case.expected_source
        target_ok = case.expected_target is None or resolved.get("target_id") == case.expected_target
        success = bool(run_result.get("status") == "success" and source_ok and target_ok)
        success_flags.append(success)
        results.append(
            {
                "case_id": case.case_id,
                "success": success,
                "status": run_result.get("status"),
                "resolved_ids": resolved,
            }
        )

    return {
        "num_cases": len(cases),
        "task_success_rate": task_success_rate(success_flags),
        "results": results,
    }

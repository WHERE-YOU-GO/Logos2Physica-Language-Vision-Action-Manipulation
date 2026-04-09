from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, save_dir: str) -> None:
        self._save_dir = Path(save_dir).expanduser()
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._current_run: dict[str, Any] | None = None

    def start_run(self, prompt: str) -> None:
        self._current_run = {
            "run_id": uuid.uuid4().hex,
            "prompt": prompt,
            "start_time_s": time.time(),
            "steps": [],
            "result": None,
        }

    def log_step(self, state: str, payload: dict) -> None:
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self._current_run["steps"].append(
            {"timestamp_s": time.time(), "state": state, "payload": payload}
        )

    def end_run(self, result: dict) -> None:
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self._current_run["end_time_s"] = time.time()
        self._current_run["result"] = result

    def flush(self) -> None:
        if self._current_run is None:
            return
        if self._current_run.get("result") is None:
            raise RuntimeError("Cannot flush a run before end_run() is called.")

        run_id = self._current_run["run_id"]
        output_path = self._save_dir / f"{run_id}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self._current_run, handle, indent=2, ensure_ascii=False)
        self._current_run = None

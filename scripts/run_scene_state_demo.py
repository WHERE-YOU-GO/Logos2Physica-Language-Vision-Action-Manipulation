from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from typing import Any

from common.logger import ProjectLogger
from perception.scene_state import SceneStateBuilder
from scripts._backend_factory import available_backend_names, build_backend
from scripts._demo_support import ColorBlockDemoDetector, SyntheticFrameProvider, default_demo_meta
from sensing.replay_frame_provider import ReplayFrameProvider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and inspect scene state from one frame.")
    parser.add_argument("--scene_dir", default=None)
    parser.add_argument("--backend", default="demo", choices=["demo", *available_backend_names()])
    parser.add_argument("--config", default="config/detector.yaml")
    return parser.parse_args()


def _candidate_labels(scene_meta: dict[str, Any]) -> list[str]:
    labels = [
        str(scene_meta.get("expected_source_label", "cube")).strip().lower(),
        str(scene_meta.get("expected_target_label", "block")).strip().lower(),
        "cube",
        "block",
    ]
    return [label for index, label in enumerate(labels) if label and label not in labels[:index]]


def _build_frame_provider(args: argparse.Namespace, logger: ProjectLogger):
    if args.scene_dir:
        return ReplayFrameProvider(args.scene_dir, logger=logger)
    return SyntheticFrameProvider(meta=default_demo_meta())


def _build_detector(args: argparse.Namespace, logger: ProjectLogger, scene_meta: dict[str, Any]):
    if args.backend == "demo":
        detector = ColorBlockDemoDetector(scene_meta=scene_meta, logger=logger)
        detector.warmup()
        return detector
    detector = build_backend(args.backend, args.config, logger)
    detector.warmup()
    return detector


def main() -> None:
    args = _parse_args()
    logger = ProjectLogger("logs/scene_state_demo")
    frame_provider = _build_frame_provider(args, logger)
    scene_meta = frame_provider.get_meta() if hasattr(frame_provider, "get_meta") else default_demo_meta()
    detector = _build_detector(args, logger, scene_meta)
    frame = frame_provider.get_current_frame()
    detections = detector.detect(frame.rgb, _candidate_labels(scene_meta))

    builder = SceneStateBuilder(logger=logger)
    scene_state = builder.build(frame, detections)

    payload = {
        "object_ids": [obj.object_id for obj in scene_state.objects],
        "objects": [
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "color": obj.color,
                "center_base_m": obj.center_base.tolist(),
            }
            for obj in scene_state.objects
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

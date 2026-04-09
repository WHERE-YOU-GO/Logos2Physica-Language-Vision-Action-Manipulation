# Logos2Physica — Language-Vision-Action Manipulation

A language-commanded pick-and-place system for the **UFactory Lite6** arm with
a **ZED 2i** RGB-D camera and **AprilTag** workspace calibration. Built for the
4-week 5551 Robotics group project (4 members).

You give the robot a sentence — *"Put the red cube on the blue block"* — and it:

1. **Parses** the sentence into `(pick_object, place_target, place_relation)`
   via an LLM API (OpenAI / Ollama / OpenRouter — no fine-tuning).
2. **Detects** each object in the ZED RGB image with zero-shot **OWL-ViT**
   (Plan A) or **HSV color thresholding** (Plan B fallback).
3. **Localizes** the object in 3D using the ZED point cloud and an AprilTag
   PnP base-frame calibration.
4. **Executes** a top-down grasp + place on the Lite6 via the xArm SDK.

## Quickstart

```bash
pip install -r prompt2pose/requirements.txt
# also install: ZED SDK Python bindings (pyzed) and xarm-python-sdk

# Edit API key + robot IP
$EDITOR prompt2pose/config.yaml

# Run a single command
python -m prompt2pose.main --prompt "Put the red cube on the blue block"
```

See [`prompt2pose/README.md`](prompt2pose/README.md) for the full guide,
including how to swap LLM backends and toggle the OWL-ViT / HSV detectors.

## Repository layout

| Path                  | Purpose                                                |
|-----------------------|--------------------------------------------------------|
| `prompt2pose/`        | **Concise 5-file working pipeline** (start here)       |
| `common/`             | Shared dataclasses, config loader, geometry utilities  |
| `perception/`         | Earlier multi-backend detector exploration             |
| `semantic_interface/` | Earlier LLM/regex parser exploration                   |
| `skill_planning/`     | Earlier pick/place planner with MoveIt fallback        |
| `control_actuation/`  | Earlier Lite6 adapter, IK solver, motion executor      |
| `fsm/`                | Earlier finite-state-machine orchestrator              |
| `config/`             | YAML configs for camera, robot, detector, workspace    |
| `tests/`              | Unit tests                                             |

`prompt2pose/` is the clean, working entry point. The other top-level packages
are the team's earlier modular draft and remain in the tree for reference.

## Plan A vs Plan B (course requirement)

| Stage         | Plan A                              | Plan B                          |
|---------------|-------------------------------------|---------------------------------|
| Command parse | LLM API (`gpt-4o-mini` etc.)        | Regex parser                    |
| Object detect | OWL-ViT zero-shot                   | HSV color thresholding          |
| Localization  | ZED point cloud (median patch)      | ZED point cloud (median patch)  |
| Motion        | xArm SDK Cartesian + RRT-Connect    | xArm SDK Cartesian + RRT-Connect|

The detector backend is selected in `prompt2pose/config.yaml` (`detector.backend:
owlvit | hsv`); both produce the same `Detection` dataclass downstream.

## Hardware

- UFactory **Lite6** (xArm SDK, TCP offset 67 mm for the Lite6 gripper)
- **ZED 2i** stereo camera (point cloud in METER units, set in `camera.py`)
- 4× AprilTags (`tag36h11`, 80 mm) at the workspace corners — see
  `camera.TAG_CENTER_COORDINATES`

# Logos2Physica — Language–Vision–Action Manipulation

A modular **language-conditioned tabletop manipulation** stack for a Lite6-class arm: one **natural-language command** drives **open-vocabulary detection**, **RGB-D geometry**, **IK / Cartesian motion planning**, **gripper execution**, and **post-grasp / post-place verification** (with bounded retries). The design follows the course proposal pipeline: *sense → parse → ground → project → plan → grasp → verify → place*.

---

## What this repository does

| Layer | Role |
|--------|------|
| **Language** | Turn a short English prompt into a structured `ParsedCommand` (source object, optional target, relation). |
| **Perception** | YOLO-World (or another detector backend) proposes 2D boxes; depth + intrinsics + extrinsics yield 3D `SceneObject`s; color/shape can be refined via HSV-style logic for disambiguation. |
| **Planning** | Top-down grasp pose, place pose, Cartesian waypoints; optional MoveIt-style fallback if waypoint generation fails. |
| **Control & safety** | Cartesian execution, gripper open/close, workspace guardrails. |
| **Verification** | Grasp and place checks using resensed scene state (and gripper state where available); FSM can retry up to `max_attempts`. |

The **primary user-facing task** is: *give the robot an instruction so it **picks up (grasps) the right object***, and optionally **places** it relative to another object or region.

---

## System inputs and outputs

**Inputs (as implemented / intended)**

- One **natural-language command** (see [Command format](#command-format-how-to-tell-it-to-grab-something) below).
- **RGB-D** from a calibrated camera (e.g. ZED 2i) — wired through `sensing/` and `RGBDFrame` with `T_base_cam`.
- **Robot / workspace** parameters in `config/robot.yaml`, `config/workspace.yaml`, `config/camera.yaml`, `config/detector.yaml`.

**Outputs**

- Parsed task structure (`ParsedCommand` / `ResolvedCommand` in `semantic_interface/command_schema.py`).
- A **pick–place plan** (grasp pose, place pose, motion segments in `PickPlacePlan`).
- **Execution result** from `Prompt2PoseFSM.run_once()`: status, state trace, resolved object IDs, retry decisions, failure reason if any.

---

## End-to-end pipeline (FSM)

The orchestrator is `fsm/main_fsm.py` (`Prompt2PoseFSM`). A single run walks these states (see `fsm/states.py`):

1. **Parse command** — regex parser first; on failure, optional LLM parser (`semantic_interface/llm_parser.py`) if configured.
2. **Sense scene** — grab frame → **detect** with candidate labels derived from the parse → **build** `SceneState`.
3. **Resolve targets** — map language queries to concrete `object_id`s (`semantic_interface/target_resolver.py`).
4. **Plan** — `PickPlacePlanner` builds grasp, place, and motions (`skill_planning/pick_place_plan.py`).
5. **Safety check** — `SafetyGuardrail` validates the plan.
6. **Execute pick** — Cartesian plan for approach / grasp; gripper handled by the execution stack.
7. **Verify grasp** — resense + `GraspVerifier` (`verification/grasp_verify.py`).
8. **Execute place** — if pick verified.
9. **Verify place** — `PlaceVerifier`; on failure, **retry** policy may replan (up to `max_attempts`).

This matches the proposal’s **Tier-1** semantic path (open-vocabulary detection + depth geometry). **Tier-2**-style robustness on colored primitives is supported in spirit by **regex color tokens** + **HSV-based color refinement** in `perception/color_shape_refiner.py` when detections are noisy; automatic tier switching by a single confidence threshold is a natural extension point, not a single `if` in one file.

---

## Command format: how to tell it to grab something

Parsing is implemented in `semantic_interface/regex_parser.py`. Supported **verbs** include: `pick up`, `pick`, `grab`, `place`, `put`, `move`.

### Pick-only (grasp without placement)

Use a **pick** or **grab** pattern **without** a second object:

```text
pick up <object description>
pick <object description>
grab <object description>
```

Examples:

```text
grab the red cube
pick the blue block
pick up the red cube
```

The parser extracts **color** (e.g. red, blue, green, yellow, orange, purple, black, white, gray), **shape** (cube, block, box), and **category** keywords from a fixed vocabulary so the detector and resolver can ground the right instance.

### Pick and place

```text
put <source> on <target>
place <source> onto <target>
move <source> to <target>
put <source> into <target>
```

Example:

```text
put the red cube on the blue block
```

For **`on` / `onto`**, the target must resolve to a **detected** object. Symbolic regions (`area`, `zone`, `region`) are only used with certain **`to`**-style relations in `TargetResolver`.

---

## Repository layout (high level)

| Path | Purpose |
|------|---------|
| `fsm/` | `Prompt2PoseFSM` state machine. |
| `semantic_interface/` | Parsers, `ParsedCommand` / `ResolvedCommand`, target resolution. |
| `sensing/` | Frame acquisition, calibration helpers (e.g. AprilTag-related utilities). |
| `perception/` | Detectors (YOLO-World, etc.), depth projection, grasp pose, scene state. |
| `skill_planning/` | Pick–place planning, waypoints, MoveIt fallback hooks. |
| `control_actuation/` | Motion, IK, gripper, safety guardrail, robot adapter. |
| `verification/` | Grasp / place verification, scene recheck. |
| `config/` | `robot.yaml`, `workspace.yaml`, `camera.yaml`, `detector.yaml`. |
| `scripts/` | Runnable demos (`run_fsm_once.py`, detector / depth demos, etc.). |
| `tests/` | Unit tests for parsers, geometry, safety, etc. |

---

## Running a quick demo

From the repository root (with Python dependencies installed: `numpy`, and for real runs `ultralytics` for YOLO-World, etc.):

**Dry run (synthetic frame + synthetic detections)** — no camera or robot:

```bash
python scripts/run_fsm_once.py "grab the red cube" --synthetic
```

**Default prompt** if you omit the argument:

```text
put the red cube on the blue block
```

**Planning-only demo** (no FSM, no hardware):

```bash
python scripts/run_pick_plan_demo.py
```

For a **real detector**, omit `--synthetic`; ensure `config/detector.yaml` and model paths are valid. If initialization fails, the script suggests using `--synthetic`.

**Optional LLM parsing** — configure an OpenAI-compatible API in the YAML used by `LLMCommandParser` (`provider`, `api_key`, `model`, optional `base_url`) and wire `llm_parser` into the FSM when regex parsing is insufficient.

---

## Configuration and hardware notes

- **Robot / motion**: `config/robot.yaml` (speeds, approach heights, IK backend flags, `robot_ip`, MoveIt toggles).
- **Workspace bounds**: `config/workspace.yaml` (used by safety and planning).
- **Camera**: `config/camera.yaml`; extrinsics typically come from calibration (e.g. AprilTag pipeline in `sensing/`).
- **Detector**: `config/detector.yaml` (e.g. YOLO-World weights path).

Align physical setup with the proposal: **static tabletop**, **camera-to-base** calibration, **Lite6 + parallel gripper** style assumptions in grasp and motion modules.

---

## Evaluation and logging (proposal alignment)

For **Manipulation-Net**-style reporting, log per trial:

- Grounding result (labels, scores, chosen `object_id`s).
- Planning and execution timing.
- Final **success / failure** and **failure stage** from `run_once()` JSON.
- Optional placement error metrics from `eval/` if you run benchmark scripts.

The proposal’s **one guarded retry** after failed grasp maps to the FSM **retry** path after `VERIFY_GRASP` (and similarly after place), subject to `RetryPolicy` and `max_attempts`.

---

## References (external components)

As in the course materials: **YOLO-World** (open-vocabulary detection), **AprilTag** (calibration), **ZED SDK** (RGB-D), **LLM API** (optional parsing), **Lite6 / xArm SDK** and optional **MoveIt**-style planning — cite and disclose per your report requirements.

---

## License / team

See project root for license if added. Course proposal credits team roles (perception, language, planning, integration); code ownership may follow the same split across `sensing/`, `semantic_interface/`, `skill_planning/` + `control_actuation/`, `fsm/`.

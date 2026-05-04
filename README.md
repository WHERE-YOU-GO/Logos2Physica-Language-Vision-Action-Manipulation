# Logos2Physica — Language–Vision–Action Manipulation

Logos2Physica is a modular language-conditioned tabletop manipulation stack for CSCI 5551. It connects short structured commands to RGB-D perception, AprilTag/world-frame calibration, geometric planning, Lite6/xArm execution, and run-time verification.

The final implemented path centers on `scripts/run_mini_task.py` with the validated `checkpoint8_style` backend. The stable hardware baseline is duplicate-aware AprilTag-assisted multi-cube placement; an opt-in preset-layout mode places cubes into predefined robot-base-frame slots without target AprilTags.

**Pipeline:** sense → parse → ground → project → plan → grasp → verify → place

## 1. Project Overview

The project goal is to translate tabletop manipulation commands into stable, measurable robot actions. The system is intentionally transparent rather than an end-to-end black box: language parsing, color/semantic grounding, RGB-D projection, AprilTag calibration, pose planning, grasp/place execution, and verification are separate layers that can be inspected and tested.

The proposal/poster frame the work as a two-tier system:

- **Tier 1 semantic path:** language parsing plus open-vocabulary or VLM-style grounding where available in the repository.
- **Tier 2 geometric fallback:** keyword/color extraction, deterministic HSV segmentation, depth projection, AprilTag or robot-base-frame targets, and checkpoint-style robot execution.

The final validated real-robot implementation emphasizes the robust Tier 2 / `checkpoint8_style` branch.

## 2. Implemented Operating Modes

- **Replay / fake-robot validation:** the broader repository contains replay scenes, demo detectors, and fake Lite6 adapters for safe software validation.
- **`checkpoint8_style` real robot execution:** `scripts/run_mini_task.py --execution_backend checkpoint8_style` is the main hardware entry point.
- **AprilTag-assisted duplicate-aware multi-cube placement:** detects multiple same-color cubes and duplicate same-ID target tags, then assigns concrete cube/tag pairs.
- **Preset-layout placement without target AprilTags:** detects source cubes from RGB-D, then places them into JSON-defined robot-base-frame slots.
- **Older standalone runner:** `scripts/run_checkpoint8_style_mini_task.py` still exists as an experimental checkpoint8-style runner, but `scripts/run_mini_task.py` is the primary documented entry point.

## 3. Hardware and Dependencies

Target lab setup:

- Lite6 / xArm robotic arm with the xArm Python SDK.
- ZED 2i or compatible ZED RGB-D camera with `pyzed`.
- OpenCV, NumPy, AprilTag detection, and RGB-D point-cloud processing.
- Colored tabletop cubes, AprilTag calibration landmarks, and a calibrated camera-to-robot transform.
- Lab checkpoint modules used by the real robot path, including `checkpoint0`, `checkpoint1`, `checkpoint6`, `utils.zed_camera`, and `utils.vis_utils`.

Recommended Linux lab setup:

```bash
bash scripts/setup_linux_lab.sh
source .venv_lab/bin/activate
bash scripts/install_optional_vision.sh
bash scripts/install_robot_stack.sh
```

Robot IP and other local hardware values should be configured locally, for example through `config/robot.yaml` or command-line overrides. Do not commit lab-specific private configuration.

## 4. Safety and Robot Readiness

Before any real robot motion, clear errors if needed and verify that the robot can home:

```bash
python scripts/run_mini_task.py --recover_robot
python scripts/run_mini_task.py --home_only
```

Expected ready status:

```text
state=(0,2)
err/warn=[0,0]
```

Safety rules:

- Run `--home_only` before real robot execution.
- Prefer `--dry_run` before every new real run.
- Do not run real motion after C22, C31, or `state=4` until `--recover_robot` and `--home_only` pass.
- Do not use `--auto_confirm` until the same command has succeeded with normal operator confirmation.
- The `checkpoint8_style` pose-plan child enforces start-home and final-home behavior.

## 5. AprilTag-Assisted Duplicate-Aware Multi-Cube Placement

This is the stable six-cube baseline:

- 2 red cubes -> 2 target detections of tag 6
- 2 green cubes -> 2 target detections of tag 8
- 2 blue cubes -> 2 target detections of tag 7

Dry run:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --duplicate_aware_multi_place \
  --duplicate_cube_tag_map "red cube:6:2,green cube:8:2,blue cube:7:2" \
  --target_tag_size_m 0.020 \
  --dry_run \
  --no_gui
```

Real run with operator confirmation:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --duplicate_aware_multi_place \
  --duplicate_cube_tag_map "red cube:6:2,green cube:8:2,blue cube:7:2" \
  --target_tag_size_m 0.020 \
  --no_gui
```

Implementation notes:

- `--duplicate_cube_tag_map` encodes cube prompt, target tag ID, and required count.
- Repeated target IDs are allowed when a count is provided.
- Raw HSV/point-cloud components are clustered into physical cube instances before assignment.
- Assignment uses global nearest matching in XY.
- For each assigned pair, the parent writes a raw pose-plan JSON.
- A refinement child opens the ZED, refines source/target poses, writes a refined pose-plan JSON, and performs no robot motion.
- A robot-only child loads the refined JSON, uses `--no_pose_plan_refine`, does not open ZED, and executes home -> `checkpoint1.grasp_cube` -> safety check -> `checkpoint1.place_cube` -> final home.
- This subprocess split keeps native ZED/OpenCV/pyzed cleanup failures isolated from robot motion.

`--target_tag_size_m 0.020` is the empirically used effective detected size for placement target tags in the validated setup. Calibration tags used by `checkpoint0` are handled separately; this value should not be interpreted as the physical size of every AprilTag in the lab.

## 6. Preset-Layout Placement Without Target AprilTags

Preset-layout mode is an opt-in extension for target-free layout generation. Source cubes are still detected using ZED/HSV/depth, but target locations come from a JSON layout in the robot base frame. The example layout is `config/layouts/hexagon_6_slots.json`.

Target AprilTags are not used for placement in this mode, but calibration tags and the camera-to-base transform are still required. Preset slot coordinates must be adjusted for the actual table and robot workspace.

Dry run:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --preset_layout_place \
  --preset_place_layout_json config/layouts/hexagon_6_slots.json \
  --preset_cube_counts "red cube:2,green cube:2,blue cube:2" \
  --preset_cube_slot_map "red cube:1,2;green cube:3,4;blue cube:5,6" \
  --dry_run \
  --no_gui
```

Real run with operator confirmation:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --preset_layout_place \
  --preset_place_layout_json config/layouts/hexagon_6_slots.json \
  --preset_cube_counts "red cube:2,green cube:2,blue cube:2" \
  --preset_cube_slot_map "red cube:1,2;green cube:3,4;blue cube:5,6" \
  --no_gui
```

Auto-confirmed run, only after a normal confirmed run succeeds:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --preset_layout_place \
  --preset_place_layout_json config/layouts/hexagon_6_slots.json \
  --preset_cube_counts "red cube:2,green cube:2,blue cube:2" \
  --preset_cube_slot_map "red cube:1,2;green cube:3,4;blue cube:5,6" \
  --auto_confirm \
  --no_gui
```

Useful preset options include `--preset_assignment_metric nearest`, `--preset_use_slot_yaw`, and `--allow_preset_slots_outside_workspace`.

## 7. Single-Color Diagnostic Runs

Use these to validate one color group at a time. Add `--dry_run` first when testing a new physical arrangement.

AprilTag-assisted red-only:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --duplicate_aware_multi_place \
  --duplicate_cube_tag_map "red cube:6:2" \
  --target_tag_size_m 0.020 \
  --no_gui
```

AprilTag-assisted green-only:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --duplicate_aware_multi_place \
  --duplicate_cube_tag_map "green cube:8:2" \
  --target_tag_size_m 0.020 \
  --no_gui
```

AprilTag-assisted blue-only:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --duplicate_aware_multi_place \
  --duplicate_cube_tag_map "blue cube:7:2" \
  --target_tag_size_m 0.020 \
  --no_gui
```

Preset red-only:

```bash
python scripts/run_mini_task.py \
  --execution_backend checkpoint8_style \
  --preset_layout_place \
  --preset_place_layout_json config/layouts/hexagon_6_slots.json \
  --preset_cube_counts "red cube:2" \
  --preset_cube_slot_map "red cube:1,2" \
  --no_gui
```

## 8. Logs and Outputs

Runtime reports, previews, and pose plans are saved under `logs/`, including:

- `logs/duplicate_assignment_report.json`
- `logs/run_mini_task_duplicate_assignment_preview.png`
- `logs/duplicate_pose_plans/`
- `logs/preset_layout_assignment_report.json`
- `logs/run_mini_task_preset_layout_preview.png`
- `logs/preset_pose_plans/`

These outputs support experiment review, success/failure summaries, task timing notes, placement-error measurement, and repeatability checks. `logs/` is ignored by `.gitignore`; do not commit raw lab logs or generated previews unless intentionally curated.

## 9. Development Tests

Run the focused syntax and parser/assignment tests:

```bash
python -m py_compile scripts/run_mini_task.py
python -m unittest tests/test_run_mini_task_parser.py -v
```

Broader software sanity checks are available through:

```bash
bash scripts/run_sanity_checks.sh
```

## 10. Known Limitations

- Real robot execution requires the lab hardware setup, vendor SDKs, and valid local robot configuration.
- The ZED camera stream must be available and not occupied by another process.
- The prompt parser is structured; it is not open-ended dialogue.
- The robust validated hardware path is the geometric / `checkpoint8_style` branch. Semantic Tier 1 components should be treated as modular project framing or partial support unless verified for the current setup.
- Duplicate-aware and preset modes currently require supported color prompts containing exactly one of `red`, `green`, or `blue`.
- Duplicate-aware assignment depends on clean segmentation, visible cubes/tags, and reasonable workspace bounds.
- Preset layout coordinates are lab-specific and must be calibrated for the physical table.
- Target-free preset layout still requires camera-to-base calibration.
- Dry runs and normal operator-confirmed runs should precede any `--auto_confirm` execution.

## 11. Relation to Proposal and Poster

The proposal/poster frame Logos2Physica as Lite6 language-conditioned tabletop manipulation with an AprilTag-centered reference frame and a Tier 1 semantic path plus Tier 2 geometric fallback. The final implementation emphasizes the robust Tier 2 / `checkpoint8_style` real-robot branch and scales it to duplicate-aware multi-cube placement.

AprilTag-assisted placement is the repeatable baseline for measurable placement. Preset-layout mode demonstrates target-free layout generation by placing detected cubes into predefined robot-base-frame slots.

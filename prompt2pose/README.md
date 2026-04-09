# Prompt2Pose

A concise language-commanded pick-and-place stack for the UFactory **Lite6** arm
+ **ZED 2i** RGB-D camera + **AprilTag** workspace calibration. Built for the
4-week 5551 Robotics group project.

You give the robot a sentence — *"Put the red cube on the blue block"* — and it:

1. Calls an LLM API (OpenAI / Ollama / OpenRouter) to parse the sentence into a
   structured `(pick_object, place_target, place_relation)` triple.
2. Detects each object in the ZED RGB image (zero-shot OWL-ViT, or HSV fallback).
3. Reads the 3D position from the ZED point cloud and transforms it into the
   robot base frame using the AprilTag PnP calibration from `checkpoint0`.
4. Executes a top-down grasp + place sequence on the Lite6.

The package is intentionally **5 source files, ~700 lines total** and mirrors
the conventions of `robotics5551/` (the official course example) so it runs on
the same hardware without modification.

## Layout

```
prompt2pose/
├── parser.py        # LLM API command parser (OpenAI-compatible)
├── camera.py        # ZED 2i wrapper + AprilTag base calibration
├── detector.py      # OWL-ViT (Plan A) and HSV (Plan B) detectors
├── robot.py         # Lite6 wrapper with grasp / place primitives
├── main.py          # PromptPipeline + CLI entry point
├── config.yaml      # API key, robot IP, detector backend
└── requirements.txt
```

## Quickstart

```bash
# 1. Install Python deps (also install ZED SDK + xarm-python-sdk separately)
pip install -r prompt2pose/requirements.txt

# 2. Edit prompt2pose/config.yaml — set your OpenAI API key and robot IP
$EDITOR prompt2pose/config.yaml

# 3. Run a command
python -m prompt2pose.main --prompt "Put the red cube on the blue block"
```

## Choosing an LLM backend

`config.yaml -> llm` uses an OpenAI-compatible client. Three common choices:

| Backend     | api_key      | model                          | base_url                              |
|-------------|--------------|--------------------------------|---------------------------------------|
| OpenAI      | `sk-...`     | `gpt-4o-mini`                  | `null`                                |
| Ollama      | `ollama`     | `qwen2.5`, `llama3.1`          | `http://localhost:11434/v1`           |
| OpenRouter  | `sk-or-...`  | `anthropic/claude-3.5-sonnet`  | `https://openrouter.ai/api/v1`        |

No fine-tuning needed — the parser is just a system prompt that asks the LLM
for strict JSON. See `parser.py:SYSTEM_PROMPT`.

## Plan A vs Plan B

- **Plan A** (default): `detector.backend: owlvit`. Zero-shot OWL-ViT from
  HuggingFace. Handles arbitrary text prompts like "blue brick" or "wooden
  cylinder". Needs a GPU and the `transformers` + `torch` packages.
- **Plan B** (fallback): `detector.backend: hsv`. Pure OpenCV color thresholding
  for `red / green / blue / yellow`. Use this when network or GPU is unavailable
  or for the live demo backup.

## How it lines up with the course checkpoints

| Checkpoint            | Prompt2Pose equivalent                              |
|-----------------------|-----------------------------------------------------|
| `checkpoint0.py`      | `camera.compute_T_cam_robot()`                      |
| `checkpoint1.py`      | `robot.Lite6.grasp_at()`                            |
| `checkpoint2.py`      | `robot.Lite6.place_at()`                            |
| `checkpoint3/8.py`    | `detector.OWLViTDetector` / `detector.HSVDetector`  |
| `checkpoint4/5.py`    | Stack-on logic in `main._resolve_place_pose()`      |
| Top-level integration | `main.PromptPipeline`                               |

## Notes

- Coordinates: all public APIs use **meters** in the robot base frame. The
  Lite6 wrapper converts to mm for the xArm SDK internally.
- The ZED point cloud is configured to METER units in `camera.ZedCamera` so it
  matches the world frame solved by `cv2.solvePnP` with the AprilTag layout.
- The grasp primitive uses a top-down orientation (`roll=180, pitch=0`); change
  `yaw_deg` in `Lite6.grasp_at()` if you add a yaw estimator later.
- Hardware execution is gated by the comment in `main.execute()` — comment the
  `self.robot.grasp_at(...)` lines for a perception-only dry run.

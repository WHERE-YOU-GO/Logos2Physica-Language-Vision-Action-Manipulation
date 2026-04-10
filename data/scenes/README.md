# Replay Scenes

This directory stores replayable RGB-D scenes for detector, scene-state, planning, and FSM dry-run demos.

## Minimum Layout

Each replay scene directory must contain:

```text
data/scenes/scene_01/
    rgb.png
    depth.npy
    meta.json
```

## File Roles

- `rgb.png`: The RGB image for the scene.
- `depth.npy`: A depth array aligned with `rgb.png`. Depth values must be in meters.
- `meta.json`: Scene metadata and test expectations.

## Minimum `meta.json` Example

```json
{
  "prompt": "put the red cube on the blue block",
  "expected_source_color": "red",
  "expected_source_label": "cube",
  "expected_target_color": "blue",
  "expected_target_label": "block"
}
```

## Optional Calibration Fields

The replay frame loader can also use optional calibration values from `meta.json`:

```json
{
  "intrinsics": {
    "fx": 700.0,
    "fy": 700.0,
    "cx": 640.0,
    "cy": 360.0,
    "width": 1280,
    "height": 720
  },
  "T_base_cam": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

If these fields are missing, replay tools may fall back to `config/camera.yaml` when possible.

## Validation

To validate one replay scene directory:

```bash
python -m scripts.validate_scene_dir --scene_dir data/scenes/scene_01
```

The validator checks:

- required files exist
- RGB size
- depth array shape
- RGB/depth alignment
- required `meta.json` fields

## Demo Asset Generation

To generate the minimal reproducible demo scene used by the dry-run scripts:

```bash
python -m scripts.generate_demo_scene --scene_dir data/scenes/scene_01 --overwrite
```

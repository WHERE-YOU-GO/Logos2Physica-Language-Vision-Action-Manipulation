# Linux-Lab Setup

Linux-Lab is the target mode for the full native Linux project. Use it on Ubuntu or another supported lab Linux image when you need the full stack, including optional vision backends and robot bring-up.

## Recommended Installation Order

Install in layers:

1. base/demo workflow
2. optional vision layer
3. optional robot stack layer

This staged approach makes failures easier to diagnose.

## Base Setup

```bash
bash scripts/setup_linux_lab.sh
source .venv_lab/bin/activate
```

## Sanity Checks

```bash
bash scripts/run_sanity_checks.sh
```

This validates:

- Python imports
- tests
- replay scene tooling
- fake robot planning/demo flows

## Optional Vision Layer

Install this only after the base/demo workflow is stable:

```bash
bash scripts/install_optional_vision.sh
```

The default detector backend in the platform config remains `demo` because it is the safest initial state. After the vision layer is validated, you can switch to `yolo_world` or another real backend that matches your lab machine, driver, and wheel compatibility.

## Optional Robot Stack

Install the Python part of the robot stack:

```bash
bash scripts/install_robot_stack.sh
```

System-level components such as ROS 2, ZED SDK, vendor USB rules, and lab-specific drivers must be installed separately.

## Common Failure Modes

- Python is not 3.11.
- Torch wheels do not match the machine, CUDA, or NumPy version.
- The environment is polluted by user site-packages.
- Vendor SDKs are missing from the lab image.
- USB permissions or udev rules are incomplete.

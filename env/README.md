# Environment Templates

These files provide example environment variables for the three supported platform modes.

## Common Variables

- `OPENAI_API_KEY`: Optional API key for LLM-enabled flows.
- `PROJECT_ROOT`: Absolute path to the repository root on the current machine.
- `USE_FAKE_ROBOT`: `1` for fake robot mode, `0` for hardware mode.
- `DETECTOR_BACKEND`: Recommended detector backend for the current platform mode.
- `SCENE_DIR`: Default replay scene directory for demo and sanity-check flows.
- `PLATFORM_MODE`: One of `windows_dev`, `wsl_sim`, or `linux_lab`.

## Recommended Defaults

- Windows-Dev: fake robot enabled, `demo` backend, replay-first workflow.
- WSL-Sim: fake robot enabled, `demo` backend, Linux-style development and simulation workflow.
- Linux-Lab: hardware disabled only during early smoke tests; switch to robot hardware after the base/demo stack is stable.

Copy the file that matches your platform mode, rename it if needed, and load the variables with your preferred shell tooling.

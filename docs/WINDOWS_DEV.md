# Windows-Dev

Windows-Dev is the safest starting point for this project. It is intended for local development, regression testing, replay scenes, fake robot workflows, and demo backends.

## Scope

Recommended:

- `pytest`
- replay scene validation
- fake robot FSM runs
- demo detector backend
- lightweight code editing and refactoring

Not recommended as the default path:

- final robot bring-up
- USB hardware debugging
- ZED SDK integration
- production Linux vision stack validation

## Setup

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_dev.ps1
```

Or from Command Prompt:

```bat
scripts\setup_windows_dev.bat
```

## Activate the Environment

```powershell
.\.venv_win\Scripts\Activate.ps1
```

## Run the Minimum Workflow

```powershell
python -m pytest tests -vv
python -m scripts.run_fsm_once --use_fake_robot --scene_dir data/scenes/scene_01 --backend demo
```

## Full Sanity Check

```powershell
.\scripts\run_sanity_checks.ps1
```

Windows should be treated as a strong development and regression platform, not as the preferred final real-hardware environment.

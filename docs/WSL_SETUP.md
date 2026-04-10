# WSL Setup

WSL-Sim is the recommended next step after Windows-Dev if you want Linux-style development, replay, fake robot workflows, and future simulation integration.

## Why Use WSL

WSL gives you:

- Linux shell tooling
- closer parity with Ubuntu development
- an easier path to future ROS 2 and simulation tooling

It is still not the preferred final environment for robot hardware bring-up.

## Install WSL 2 and Ubuntu

Use the standard Microsoft installation flow for WSL 2 and Ubuntu. After Ubuntu is installed, clone this repository inside the WSL Linux filesystem, for example:

```bash
cd /home/<user>
git clone <repo-url> Logos2Physica
cd Logos2Physica
```

Avoid running the project from `/mnt/c/...` or `/mnt/d/...` when possible.

## Setup

```bash
bash scripts/setup_wsl_dev.sh
```

## Activate the Environment

```bash
source .venv_wsl/bin/activate
```

## Run the Minimum Workflow

```bash
python -m pytest tests -vv
python -m scripts.run_fsm_once --use_fake_robot --scene_dir data/scenes/scene_01 --backend demo
```

## Run the Full Sanity Check

```bash
bash scripts/run_sanity_checks.sh
```

## Notes

- WSL is well suited for replay, fake robot, simulation-style workflows, and Linux-oriented development.
- WSL is not the preferred final environment for robot hardware bring-up.

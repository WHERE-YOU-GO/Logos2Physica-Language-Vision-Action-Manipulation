# Cross-Platform Setup

This project supports three reproducible operating modes:

1. Windows-Dev
2. WSL-Sim
3. Linux-Lab

## Recommended Progression

Follow this order when bringing up a new developer or lab machine:

1. Windows-Dev
2. WSL-Sim
3. Linux-Lab

This order reduces uncertainty. Start with replay, fake robot, and tests. Move to Linux-style simulation next. Reserve full vision and robot bring-up for native Linux lab machines.

## Why Not Copy a Windows Environment to Linux

Do not treat a Windows Python environment as a template for Linux. Wheel availability, binary compatibility, GPU stack requirements, USB access, ROS 2 packaging, and vendor SDKs differ across platforms. A package set that works on Windows can still fail on Ubuntu because of ABI, system library, or driver differences.

## Why Requirements Are Layered

The requirements are split so you can install only the level you need:

- `base.txt`: core logic, tests, and config
- `dev.txt`: linting and developer tooling
- `demo.txt`: replay and fake-robot workflow
- `vision.txt`: heavy vision backends
- `robot.txt`: robot-side Python packages

Platform overlays such as `windows.txt`, `wsl.txt`, and `linux.txt` are intentionally lightweight. They describe the platform mode rather than forcing large platform-specific package stacks.

## Quick Start

- Windows-Dev: run `scripts/setup_windows_dev.ps1`
- WSL-Sim: run `bash scripts/setup_wsl_dev.sh`
- Linux-Lab: run `bash scripts/setup_linux_lab.sh`

After setup, run the matching sanity-check script before installing vision or robot-specific layers.

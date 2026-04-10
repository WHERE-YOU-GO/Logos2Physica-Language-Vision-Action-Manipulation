# Cross-Platform Development

This project is designed to support three practical development modes:

1. Windows-Dev
2. WSL-Dev
3. Linux-Lab

## Windows Development

Use Windows for:

- editing code
- running `pytest`
- replay scenes
- fake robot workflows
- lightweight demo validation

Windows is the most convenient place to stabilize pure Python logic and regression tests, but it is not the preferred long-term environment for full hardware bring-up.

## WSL Development

Use WSL 2 + Ubuntu for:

- Linux-style shell workflows
- `pytest`
- replay scenes
- fake robot runs
- detector/demo iteration

WSL is a strong bridge between Windows editing and native Linux bring-up. It is still better to keep the repository inside the WSL Linux filesystem instead of `/mnt/c/...` or `/mnt/d/...`.

## Linux Lab Development

Use a native Linux lab machine for:

- the full Linux project
- heavy vision dependencies
- robot-side integrations
- staged real hardware bring-up

Start with the base/demo workflow, then install the vision layer, and only then enable robot and hardware-specific components.

## Why Not Develop Long-Term Under /mnt/d/...

Running large Python projects from `/mnt/c/...` or `/mnt/d/...` inside WSL usually causes:

- slower filesystem performance
- more fragile file watching and caching
- occasional permission and path behavior mismatches

For daily WSL development, clone the repository under `/home/<user>/...`.

## Recommended Workflow by Mode

- Windows: `pytest` + fake robot
- WSL: `pytest` + replay + fake robot + detector/demo
- Linux-Lab: full stack bring-up

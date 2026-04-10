# Platform Modes

This repository supports three operating modes with different goals and risk levels.

## Windows-Dev

Use this mode for:

- editing code
- running tests
- replay scenes
- fake robot workflows
- demo backend validation

Avoid using this mode as the default for:

- final robot bring-up
- Linux-only SDK validation
- USB hardware debugging

Defaults:

- fake robot enabled
- detector backend `demo`
- replay scene `data/scenes/scene_01`
- GUI simulation disabled
- robot hardware disabled

## WSL-Sim

Use this mode for:

- Linux-style development on a Windows host
- replay scenes
- fake robot workflows
- simulation-oriented tooling
- preparing for ROS 2 or Webots integration

Avoid using this mode as the final hardware bring-up target.

Defaults:

- fake robot enabled
- detector backend `demo`
- GUI simulation allowed
- robot hardware disabled
- USB hardware disabled

## Linux-Lab

Use this mode for:

- native Linux development
- full vision stack validation
- staged robot bring-up
- USB and hardware access
- future production-like lab workflows

Defaults:

- fake robot disabled by default, but still useful during early smoke tests
- detector backend `demo` at first, then `yolo_world` after the vision layer is validated
- GUI simulation allowed
- robot hardware allowed
- USB hardware allowed

## Boundary Summary

- Windows-Dev: best for local development and regression checks
- WSL-Sim: best for Linux-style development and simulation preparation
- Linux-Lab: best for full-stack validation and lab bring-up

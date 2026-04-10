#!/usr/bin/env bash
set -euo pipefail

run_step() {
  local title="$1"
  shift
  echo
  echo "== ${title} =="
  "$@"
}

run_step "Pytest" python -m pytest tests -vv
run_step "Verify Python Environment" python -m scripts.verify_python_env
run_step "Runtime Environment Report" python -m scripts.check_runtime_env
run_step "Platform Report" python -m scripts.check_platform
run_step "Depth Projection Demo" python -m scripts.run_depth_projection_demo
run_step "Validate Replay Scene" python -m scripts.validate_scene_dir --scene_dir data/scenes/scene_01
run_step "Scene State Demo" python -m scripts.run_scene_state_demo --scene_dir data/scenes/scene_01 --backend demo
run_step "Pick Plan Demo" python -m scripts.run_pick_plan_demo --scene_dir data/scenes/scene_01 --backend demo
run_step "FSM Dry Run" python -m scripts.run_fsm_once --use_fake_robot --scene_dir data/scenes/scene_01 --backend demo

echo
echo "All WSL/Linux sanity checks passed."

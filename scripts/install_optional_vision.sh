#!/usr/bin/env bash
set -euo pipefail

trap 'echo "Vision install failed. Check CUDA, driver, wheel, torch, and NumPy compatibility." >&2' ERR

echo "== Installing optional vision layer =="
python -m pip install -r requirements/vision.txt

echo
echo "== Environment report after vision install =="
python -m scripts.print_env_report

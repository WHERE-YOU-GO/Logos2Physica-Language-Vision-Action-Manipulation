#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:"
  echo "source scripts/activate_project_env.sh"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f .venv_lab/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv_lab/bin/activate
  echo "Activated .venv_lab"
elif [[ -f .venv_wsl/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv_wsl/bin/activate
  echo "Activated .venv_wsl"
else
  echo "No Linux virtual environment was found."
  echo "Run bash scripts/setup_wsl_dev.sh or bash scripts/setup_linux_lab.sh first."
fi

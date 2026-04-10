#!/usr/bin/env bash
set -euo pipefail

print_step() {
  echo
  echo "== $1 =="
}

is_wsl() {
  if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    return 0
  fi
  if [[ -f /proc/version ]] && grep -qi microsoft /proc/version; then
    return 0
  fi
  return 1
}

pick_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    local version
    version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${version}" == "3.11" ]]; then
      echo "python3"
      return 0
    fi
  fi
  return 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is intended for native Linux."
  exit 1
fi

if is_wsl; then
  echo "This script is for native Linux-Lab machines, not WSL."
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

print_step "Detecting Python 3.11"
if ! PYTHON_BIN="$(pick_python)"; then
  echo "Python 3.11 was not found. Install python3.11 and rerun this script."
  exit 1
fi
echo "Using ${PYTHON_BIN}"

if [[ -d .venv_lab ]]; then
  print_step "Reusing existing .venv_lab"
else
  print_step "Creating .venv_lab"
  "${PYTHON_BIN}" -m venv .venv_lab
fi

source .venv_lab/bin/activate

print_step "Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

print_step "Installing layered requirements"
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/dev.txt
python -m pip install -r requirements/demo.txt
python -m pip install -r requirements/linux.txt

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

print_step "Linux-Lab base environment is ready"
echo "Activate:"
echo "source .venv_lab/bin/activate"
echo
echo "Recommended commands:"
echo "bash scripts/run_sanity_checks.sh"
echo "bash scripts/install_optional_vision.sh"
echo "bash scripts/install_robot_stack.sh"

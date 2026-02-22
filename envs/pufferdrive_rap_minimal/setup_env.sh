#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${1:-${ROOT_DIR}/.venv-pufferdrive-rap}"
MAP_DIR="${2:-resources/drive/binaries}"

echo "[setup] repo root: ${ROOT_DIR}"
echo "[setup] venv path: ${VENV_PATH}"
echo "[setup] map dir for Drive validation: ${MAP_DIR}"

python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

# Ensure pip exists even in minimal/system images where venv lacks it.
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing pufferlib in editable mode (minimal train deps)..."
NO_TRAIN=1 python -m pip install -e "${ROOT_DIR}"

echo "[setup] compiling native extensions in-place..."
NO_TRAIN=1 python "${ROOT_DIR}/setup.py" build_ext --inplace --force

echo "[setup] installing RAP bridge extras..."
python -m pip install -r "${ROOT_DIR}/envs/pufferdrive_rap_minimal/requirements.txt"

echo "[setup] running enforced validation (imports + renderer + Drive reset/step/state/map)..."
python "${ROOT_DIR}/envs/pufferdrive_rap_minimal/validate_env.py" --check-drive --map-dir "${MAP_DIR}"

echo "[setup] done. Drive functionality validation passed."
echo
echo "Activate with:"
echo "  source \"${VENV_PATH}/bin/activate\""
echo
echo "Quick validation:"
echo "  python \"${ROOT_DIR}/envs/pufferdrive_rap_minimal/validate_env.py\" --check-drive"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH_INPUT="${1:-${ROOT_DIR}/.venv-pufferdrive-rap}"
MAP_DIR_INPUT="${2:-resources/drive/binaries}"
PIP_RETRIES="${PIP_RETRIES:-3}"
PIP_RETRY_DELAY_SEC="${PIP_RETRY_DELAY_SEC:-5}"
BUILD_RETRIES="${BUILD_RETRIES:-2}"
BUILD_RETRY_DELAY_SEC="${BUILD_RETRY_DELAY_SEC:-5}"

if [[ "${VENV_PATH_INPUT}" = /* ]]; then
  VENV_PATH="${VENV_PATH_INPUT}"
else
  VENV_PATH="${ROOT_DIR}/${VENV_PATH_INPUT}"
fi

if [[ "${MAP_DIR_INPUT}" = /* ]]; then
  MAP_DIR="${MAP_DIR_INPUT}"
else
  MAP_DIR="${ROOT_DIR}/${MAP_DIR_INPUT}"
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[setup][error] python not found in PATH"
  exit 1
fi

if ! command -v gcc >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
  echo "[setup][error] C compiler not found (need gcc or cc for native extensions)."
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/setup.py" ]]; then
  echo "[setup][error] setup.py not found at ${ROOT_DIR}. Is this repo checkout complete?"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/third_party/RAP/process_data/helpers/renderer.py" ]]; then
  echo "[setup][error] RAP renderer not found at third_party/RAP/process_data/helpers/renderer.py"
  echo "[setup][error] Verify submodules/vendor files are present."
  exit 1
fi

if [[ ! -f "${MAP_DIR}/map_000.bin" ]]; then
  echo "[setup][error] map_000.bin not found under: ${MAP_DIR}"
  echo "[setup][error] Pass map dir explicitly: bash envs/pufferdrive_rap_minimal/setup_env.sh <venv> <map_dir>"
  exit 1
fi

read -r -a PIP_EXTRA_ARGS_ARR <<< "${PIP_EXTRA_ARGS:-}"

run_pip() {
  local attempt=1
  while true; do
    if python -m pip --disable-pip-version-check --no-input "${PIP_EXTRA_ARGS_ARR[@]}" "$@"; then
      return 0
    fi
    if (( attempt >= PIP_RETRIES )); then
      echo "[setup][error] pip command failed after ${attempt} attempts: python -m pip $*"
      echo "[setup][error] If your node requires a custom index, set PIP_EXTRA_ARGS."
      echo "[setup][error] Example: PIP_EXTRA_ARGS='-i https://<your-index>/simple --trusted-host <your-index-host>'"
      return 1
    fi
    echo "[setup] pip command failed. retry ${attempt}/${PIP_RETRIES} in ${PIP_RETRY_DELAY_SEC}s..."
    sleep "${PIP_RETRY_DELAY_SEC}"
    attempt=$((attempt + 1))
  done
}

run_native_build() {
  local attempt=1
  while true; do
    if NO_TRAIN=1 python "${ROOT_DIR}/setup.py" build_ext --inplace --force; then
      return 0
    fi
    if (( attempt >= BUILD_RETRIES )); then
      echo "[setup][error] native build failed after ${attempt} attempts."
      echo "[setup][error] setup.py downloads native assets (raylib/box2d/inih) when missing."
      echo "[setup][error] Verify network/proxy access to github.com or rerun with cached assets present."
      return 1
    fi
    echo "[setup] native build failed. retry ${attempt}/${BUILD_RETRIES} in ${BUILD_RETRY_DELAY_SEC}s..."
    sleep "${BUILD_RETRY_DELAY_SEC}"
    attempt=$((attempt + 1))
  done
}

echo "[setup] repo root: ${ROOT_DIR}"
echo "[setup] python: ${PYTHON_BIN}"
echo "[setup] venv path: ${VENV_PATH}"
echo "[setup] map dir for Drive validation: ${MAP_DIR}"
echo "[setup] pip retries: ${PIP_RETRIES}"
echo "[setup] native build retries: ${BUILD_RETRIES}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("[setup][error] python>=3.9 is required")
print(f"[setup] python version: {sys.version.split()[0]}")
PY

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

# Ensure pip exists even in minimal/system images where venv lacks it.
python -m ensurepip --upgrade
run_pip install --upgrade pip setuptools wheel

echo "[setup] installing pufferlib in editable mode (minimal train deps)..."
NO_TRAIN=1 run_pip install --no-build-isolation -e "${ROOT_DIR}"

echo "[setup] compiling native extensions in-place..."
run_native_build

echo "[setup] installing RAP bridge extras..."
run_pip install -r "${ROOT_DIR}/envs/pufferdrive_rap_minimal/requirements.txt"

echo "[setup] running enforced validation (imports + renderer + Drive reset/step/state/map)..."
python "${ROOT_DIR}/envs/pufferdrive_rap_minimal/validate_env.py" --check-drive --map-dir "${MAP_DIR}"

echo "[setup] done. Drive functionality validation passed."
echo
echo "Activate with:"
echo "  source \"${VENV_PATH}/bin/activate\""
echo
echo "Quick validation:"
echo "  python \"${ROOT_DIR}/envs/pufferdrive_rap_minimal/validate_env.py\" --check-drive"

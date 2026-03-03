#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH_INPUT="${1:-${ROOT_DIR}/.venv-pufferdrive-rap}"
MAP_DIR_INPUT="${2:-resources/drive/binaries}"

# Keep setup linear and manual-friendly.
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_JAX="${INSTALL_JAX:-1}"
JAX_WHEEL="${JAX_WHEEL:-jax[cuda12]}"
ENFORCE_JAX_GPU="${ENFORCE_JAX_GPU:-1}"
BUILD_EXT="${BUILD_EXT:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"

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

if [[ ! -f "${ROOT_DIR}/setup.py" ]]; then
  echo "[setup][error] setup.py not found at ${ROOT_DIR}"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/third_party/RAP/process_data/helpers/renderer.py" ]]; then
  echo "[setup][error] missing RAP renderer at third_party/RAP/process_data/helpers/renderer.py"
  exit 1
fi

read -r -a PIP_EXTRA_ARGS_ARR <<< "${PIP_EXTRA_ARGS:-}"
run_pip() {
  python -m pip --disable-pip-version-check --no-input "${PIP_EXTRA_ARGS_ARR[@]}" "$@"
}

echo "[setup] repo root: ${ROOT_DIR}"
echo "[setup] python: ${PYTHON_BIN}"
echo "[setup] venv path: ${VENV_PATH}"
echo "[setup] map dir: ${MAP_DIR}"
echo "[setup] install torch: ${INSTALL_TORCH}"
echo "[setup] install jax: ${INSTALL_JAX}"
echo "[setup] jax wheel: ${JAX_WHEEL}"
echo "[setup] enforce jax gpu: ${ENFORCE_JAX_GPU}"
echo "[setup] build native extensions: ${BUILD_EXT}"
echo "[setup] run validation: ${RUN_VALIDATE}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("[setup][error] python>=3.9 is required")
print(f"[setup] python version: {sys.version.split()[0]}")
PY

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m ensurepip --upgrade
run_pip install --upgrade pip setuptools wheel

echo "[setup] installing core deps..."
run_pip install \
  "numpy<2.0" \
  "opencv-python-headless==4.9.0.80" \
  "tqdm" \
  "shimmy[gym-v21]" \
  "gym==0.23" \
  "gymnasium==0.29.1" \
  "pettingzoo==1.24.1"

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  echo "[setup] installing torch..."
  run_pip install torch
fi

if [[ "${INSTALL_JAX}" == "1" ]]; then
  echo "[setup] installing jax package: ${JAX_WHEEL}"
  run_pip install --upgrade "${JAX_WHEEL}"
  python - <<'PY'
import os
import jax
import jaxlib

backend = jax.default_backend()
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == "gpu"]
print(f"[setup] jax version: {jax.__version__}, jaxlib version: {jaxlib.__version__}")
print(f"[setup] jax backend: {backend}")
print(f"[setup] jax devices: {devices}")

if os.environ.get("ENFORCE_JAX_GPU", "1") == "1" and not gpu_devices:
    raise SystemExit(
        "[setup][error] JAX GPU enforcement is enabled but no GPU devices were detected. "
        "Ensure NVIDIA driver/CUDA runtime is available and JAX CUDA wheels are installed."
    )
PY
fi

echo "[setup] installing local package (editable, NO_TRAIN=1)..."
NO_TRAIN=1 run_pip install --no-build-isolation -e "${ROOT_DIR}"

if [[ "${BUILD_EXT}" == "1" ]]; then
  echo "[setup] building native extensions..."
  NO_TRAIN=1 python "${ROOT_DIR}/setup.py" build_ext --inplace --force
fi

echo "[setup] installing RAP extras from requirements.txt..."
run_pip install -r "${ROOT_DIR}/envs/pufferdrive_rap_minimal/requirements.txt"

if [[ "${RUN_VALIDATE}" == "1" ]]; then
  if [[ -f "${MAP_DIR}/map_000.bin" ]]; then
    echo "[setup] running full validation (with Drive check)..."
    python "${ROOT_DIR}/envs/pufferdrive_rap_minimal/validate_env.py" --check-drive --map-dir "${MAP_DIR}"
  else
    echo "[setup][warn] map_000.bin not found in ${MAP_DIR}; running import/renderer validation only."
    python "${ROOT_DIR}/envs/pufferdrive_rap_minimal/validate_env.py"
  fi
fi

echo
echo "[setup] done."
echo "Activate with:"
echo "  source \"${VENV_PATH}/bin/activate\""

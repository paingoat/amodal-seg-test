#!/usr/bin/env bash
# Build GroundingDINO's ms_deform_attn C++/CUDA extension for the active GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

eval "$(conda shell.bash hook)"
conda activate "$AMODAL_ENV_NAME"

GDINO_REPO="$GROUNDED_SAM_REPO/GroundingDINO"
if [ ! -f "$GDINO_REPO/setup.py" ]; then
  echo "[fix_groundingdino_ops] ERROR: setup.py missing: $GDINO_REPO"
  echo "Run: bash scripts/model.sh"
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[fix_groundingdino_ops] ERROR: nvcc not found."
  echo "Install the CUDA toolkit matching torch.version.cuda; the NVIDIA driver alone is insufficient."
  exit 1
fi

if [ -z "${CUDA_HOME:-}" ]; then
  NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
  export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# RTX 5090 is compute capability 12.0. This also forces a CUDA build when
# setup.py runs in an environment where torch.cuda.is_available() is false.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "[fix_groundingdino_ops] CUDA_HOME=$CUDA_HOME"
echo "[fix_groundingdino_ops] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
nvcc --version
python -c "import torch; print(f'torch={torch.__version__} torch.version.cuda={torch.version.cuda}'); print(f'cuda_available={torch.cuda.is_available()}')"

python -m pip install -q ninja
cd "$GDINO_REPO"
python setup.py clean --all || true
python setup.py build_ext --inplace
python -m pip install -e . --no-build-isolation --no-deps

cd "$AMODAL_ROOT"
python "$SCRIPT_DIR/verify_groundingdino_ops.py"

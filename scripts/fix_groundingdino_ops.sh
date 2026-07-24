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

TORCH_CUDA_VERSION="$(python -c "import torch; print(torch.version.cuda or '')")"
if [ -z "$TORCH_CUDA_VERSION" ]; then
  echo "[fix_groundingdino_ops] ERROR: installed PyTorch has no CUDA runtime."
  exit 1
fi

nvcc_version() {
  "$1" --version 2>/dev/null |
    sed -n '/release /{s/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p;q;}'
}

# Prefer a toolkit matching torch.version.cuda. `nvidia-smi` only reports the
# driver's maximum CUDA version; compiling _C requires a matching local nvcc.
NVCC_PATH=""
DETECTED_NVCC=""
CANDIDATES=()
[ -n "${CUDA_HOME:-}" ] && CANDIDATES+=("$CUDA_HOME/bin/nvcc")
CANDIDATES+=("/usr/local/cuda-$TORCH_CUDA_VERSION/bin/nvcc")
CANDIDATES+=("/usr/local/cuda/bin/nvcc")
if command -v nvcc >/dev/null 2>&1; then
  CANDIDATES+=("$(command -v nvcc)")
fi

for candidate in "${CANDIDATES[@]}"; do
  [ -x "$candidate" ] || continue
  candidate="$(readlink -f "$candidate")"
  version="$(nvcc_version "$candidate")"
  DETECTED_NVCC="${DETECTED_NVCC}${candidate} (CUDA ${version:-unknown}); "
  if [ "$version" = "$TORCH_CUDA_VERSION" ]; then
    NVCC_PATH="$candidate"
    break
  fi
done

if [ -z "$NVCC_PATH" ]; then
  echo "[fix_groundingdino_ops] ERROR: no nvcc matching PyTorch CUDA $TORCH_CUDA_VERSION."
  echo "Detected: ${DETECTED_NVCC:-none}"
  echo "Install the matching toolkit (Ubuntu with NVIDIA CUDA repo configured):"
  echo "  sudo apt update && sudo apt install cuda-toolkit-${TORCH_CUDA_VERSION/./-}"
  echo "Then rerun this script. Do not use the generic 'cuda' package to replace the driver."
  exit 1
fi

export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# RTX 5090 is compute capability 12.0. This also forces a CUDA build when
# setup.py runs in an environment where torch.cuda.is_available() is false.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "[fix_groundingdino_ops] CUDA_HOME=$CUDA_HOME"
echo "[fix_groundingdino_ops] torch CUDA=$TORCH_CUDA_VERSION, nvcc CUDA=$(nvcc_version "$NVCC_PATH")"
echo "[fix_groundingdino_ops] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
"$NVCC_PATH" --version
python -c "import torch; print(f'torch={torch.__version__} torch.version.cuda={torch.version.cuda}'); print(f'cuda_available={torch.cuda.is_available()}')"

python -m pip install -q ninja
cd "$GDINO_REPO"
python setup.py clean --all || true
python setup.py build_ext --inplace
python -m pip install -e . --no-build-isolation --no-deps

cd "$AMODAL_ROOT"
python "$SCRIPT_DIR/verify_groundingdino_ops.py"

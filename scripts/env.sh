#!/usr/bin/env bash
# Create dual conda envs (amodal + lisa) for RTX 5090 / CUDA 13.2.
# Intentionally does NOT pass -y to conda create (newer conda may block it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

echo "============================================================"
echo " Amodal dual-env setup (Python 3.11, torch ${TORCH_VERSION}+cu132)"
echo "============================================================"
echo "AMODAL_ROOT = $AMODAL_ROOT"
echo "MODELS_DIR  = $MODELS_DIR"
echo "HF_HOME     = $HF_HOME"
echo "Envs        = $AMODAL_ENV_NAME , $LISA_ENV_NAME"
echo
if [ ! -f "$AMODAL_ROOT/.env" ]; then
  echo "NOTE: $AMODAL_ROOT/.env not found."
  echo "      cp .env.example .env  # then set HF_HOME (default /backup/data/art-gen)"
  echo
fi
read -r -p "Press Enter to continue (Ctrl+C to abort)..."

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

# Make conda available in non-interactive shells
eval "$(conda shell.bash hook)"

create_env_if_needed() {
  local name="$1"
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "[env] Conda env '$name' already exists — skipping create."
  else
    echo "[env] Creating conda env '$name' with Python 3.11 ..."
    echo "      (Confirm any conda prompts interactively — no -y flag.)"
    conda create -n "$name" python=3.11
  fi
}

install_torch() {
  local name="$1"
  echo "[env] Installing torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} into '$name' ..."
  conda run -n "$name" python -m pip install --upgrade pip
  conda run -n "$name" python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
}

create_env_if_needed "$AMODAL_ENV_NAME"
create_env_if_needed "$LISA_ENV_NAME"

install_torch "$AMODAL_ENV_NAME"
install_torch "$LISA_ENV_NAME"

echo "[env] Installing amodal requirements ..."
conda run -n "$AMODAL_ENV_NAME" python -m pip install -r "$AMODAL_ROOT/requirements.txt"

# OpenAI CLIP: setup.py imports pkg_resources; pip build isolation breaks on modern setuptools.
# Install into the env without isolation after setuptools is present.
echo "[env] Installing OpenAI CLIP (no build isolation) ..."
conda run -n "$AMODAL_ENV_NAME" python -m pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"

echo "[env] Installing lisa requirements ..."
conda run -n "$LISA_ENV_NAME" python -m pip install -r "$AMODAL_ROOT/requirements-lisa.txt"

echo
echo "[env] GPU smoke check (amodal env) ..."
conda run -n "$AMODAL_ENV_NAME" python "$AMODAL_ROOT/scripts/check_gpu.py" || true

echo
echo "Done. Next steps:"
echo "  1) source $SCRIPT_DIR/paths.env"
echo "  2) bash $SCRIPT_DIR/model.sh"
echo "  3) conda activate $AMODAL_ENV_NAME   # or $LISA_ENV_NAME for LISA server"

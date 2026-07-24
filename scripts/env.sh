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

# Gradio 3.39 cluster — must be installed as one locked set (see requirements-lisa.txt).
install_lisa_gradio_stack() {
  echo "[env] Locking Gradio 3.39 / Pydantic v1 / gradio_client 0.5.0 into '$LISA_ENV_NAME' ..."
  conda run -n "$LISA_ENV_NAME" python -m pip install \
    "setuptools>=68,<81" wheel \
    "pydantic==1.10.13" \
    "fastapi==0.100.1" \
    "starlette==0.27.0" \
    "gradio_client==0.5.0" \
    "gradio==3.39.0" \
    "websockets>=10,<12" \
    "huggingface_hub>=0.16,<0.23" \
    "uvicorn==0.23.2" \
    "markupsafe>=2,<3" \
    "httpx>=0.24,<0.28"
}

verify_lisa_imports() {
  echo "[env] Verifying LISA Gradio imports ..."
  conda run -n "$LISA_ENV_NAME" python - <<'PY'
import pydantic, fastapi, gradio, gradio_client
from gradio_client.serializing import JSONSerializable  # noqa: F401
print("pydantic", pydantic.__version__)
print("fastapi", fastapi.__version__)
print("gradio", gradio.__version__)
print("gradio_client", gradio_client.__version__)
assert pydantic.__version__.startswith("1."), pydantic.__version__
assert gradio.__version__.startswith("3.39"), gradio.__version__
assert gradio_client.__version__.startswith("0.5"), gradio_client.__version__
print("OK: LISA Gradio stack imports")
PY
}

install_amodal_hf_stack() {
  echo "[env] Locking amodal HF stack (diffusers 0.32.2 + transformers 4.46.3) ..."
  conda run -n "$AMODAL_ENV_NAME" python -m pip install \
    "setuptools>=68,<81" \
    "transformers==4.46.3" \
    "tokenizers>=0.19,<0.21" \
    "diffusers==0.32.2" \
    "accelerate==1.2.1" \
    "huggingface_hub>=0.26,<0.30" \
    "safetensors>=0.4.0"
}

install_amodal_clip() {
  echo "[env] Installing OpenAI CLIP (no build isolation) ..."
  conda run -n "$AMODAL_ENV_NAME" python -m pip install ftfy regex tqdm
  conda run -n "$AMODAL_ENV_NAME" python -m pip install --no-build-isolation \
    "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"
}

verify_amodal_imports() {
  echo "[env] Verifying amodal imports via scripts/verify_amodal.py ..."
  # Sibling packages may be missing until model.sh — allow soft fail here, hard-check pip core.
  conda run -n "$AMODAL_ENV_NAME" python - <<'PY'
import clip  # noqa: F401
import transformers, diffusers, accelerate, gradio, cv2, torch
from diffusers import StableDiffusionInpaintPipeline  # noqa: F401
assert transformers.__version__.startswith("4.46.")
assert diffusers.__version__.startswith("0.32.")
import addict, pycocotools  # sibling runtime deps
print("OK: torch/cv2/clip/diffusers/transformers/gradio/addict/pycocotools")
PY
}

create_env_if_needed "$AMODAL_ENV_NAME"
create_env_if_needed "$LISA_ENV_NAME"

install_torch "$AMODAL_ENV_NAME"
install_torch "$LISA_ENV_NAME"

echo "[env] Installing amodal requirements ..."
conda run -n "$AMODAL_ENV_NAME" python -m pip install -r "$AMODAL_ROOT/requirements.txt"
install_amodal_hf_stack
install_amodal_clip
verify_amodal_imports

echo "[env] Installing lisa requirements ..."
conda run -n "$LISA_ENV_NAME" python -m pip install -r "$AMODAL_ROOT/requirements-lisa.txt"
# Re-lock Gradio cluster AFTER the rest (prevents huggingface_hub etc. from upgrading client/pydantic)
install_lisa_gradio_stack
verify_lisa_imports

echo
echo "[env] GPU smoke check (amodal env) ..."
conda run -n "$AMODAL_ENV_NAME" python "$AMODAL_ROOT/scripts/check_gpu.py" || true

echo
echo "Done. Next steps:"
echo "  1) source $SCRIPT_DIR/paths.env"
echo "  2) bash $SCRIPT_DIR/model.sh"
echo "  3) python $SCRIPT_DIR/verify_amodal.py   # after model.sh (checks siblings too)"
echo "  4) conda activate $AMODAL_ENV_NAME   # or $LISA_ENV_NAME for LISA server"
echo "Repair later: bash $SCRIPT_DIR/fix_amodal_deps.sh | bash $SCRIPT_DIR/fix_lisa_deps.sh"

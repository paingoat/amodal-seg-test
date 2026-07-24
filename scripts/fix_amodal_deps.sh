#!/usr/bin/env bash
# Repair amodal env: HF stack + OpenAI CLIP + smoke-import core packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

eval "$(conda shell.bash hook)"
conda activate "$AMODAL_ENV_NAME"

echo "[fix_amodal_deps] Repairing env: $AMODAL_ENV_NAME"

python -m pip install "setuptools>=68,<81" wheel ftfy regex tqdm

echo "[fix_amodal_deps] Locking HF stack ..."
python -m pip install \
  "transformers==4.46.3" \
  "tokenizers>=0.19,<0.21" \
  "diffusers==0.32.2" \
  "accelerate==1.2.1" \
  "huggingface_hub>=0.26,<0.30" \
  "safetensors>=0.4.0"

echo "[fix_amodal_deps] Installing OpenAI CLIP (no build isolation) ..."
python -m pip install --no-build-isolation \
  "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"

echo "[fix_amodal_deps] Ensuring Gradio UI deps ..."
# Gradio 4.44.1 stack: pydantic<2.11 (API schema) + starlette<1 (TemplateResponse)
python -m pip install \
  "gradio==4.44.1" \
  "gradio_client==1.3.0" \
  "pydantic>=2.0,<2.11" \
  "starlette>=0.37,<1.0" \
  "fastapi>=0.111,<0.116"

echo "[fix_amodal_deps] Sibling runtime deps (GroundingDINO / InstaOrder / RAM) ..."
python -m pip install \
  "addict>=2.4.0" "yapf>=0.40.0" "pycocotools>=2.0.6" "supervision>=0.22.0" \
  "fairscale>=0.4.4"

echo "[fix_amodal_deps] Reinstalling segment_anything (broken stub → SamPredictor missing) ..."
# A bad/empty pip package shadows Grounded-SAM's local copy ("unknown location").
python -m pip uninstall -y segment-anything segment_anything 2>/dev/null || true
if [ -d "$GROUNDED_SAM_REPO/segment_anything" ]; then
  python -m pip install -e "$GROUNDED_SAM_REPO/segment_anything"
else
  echo "WARN: $GROUNDED_SAM_REPO/segment_anything missing — run bash scripts/model.sh first"
  python -m pip install "git+https://github.com/facebookresearch/segment-anything.git"
fi

if [ -d "$GROUNDED_SAM_REPO/GroundingDINO" ]; then
  echo "[fix_amodal_deps] Building GroundingDINO CUDA ops ..."
  bash "$SCRIPT_DIR/fix_groundingdino_ops.sh"
fi
if [ -d "$RAM_REPO" ]; then
  echo "[fix_amodal_deps] Ensuring RAM editable install ..."
  # Do not pip -r RAM requirements.txt (pins ancient timm / reinstalls CLIP).
  python -m pip install "fairscale>=0.4.4"
  python -m pip install -e "$RAM_REPO" || echo "WARN: RAM editable install failed"
fi

if [ -d "$INSTAORDER_REPO" ] || [ -d "$AMODAL_ROOT/InstaOrder" ]; then
  echo "[fix_amodal_deps] Patching InstaOrder for NumPy>=1.24 ..."
  python "$SCRIPT_DIR/patch_instaorder_numpy.py" || true
fi

echo "[fix_amodal_deps] Verifying imports ..."
python "$AMODAL_ROOT/scripts/verify_amodal.py"

echo
echo "Next:"
echo "  source scripts/paths.env"
echo "  bash scripts/start_gradio.sh"

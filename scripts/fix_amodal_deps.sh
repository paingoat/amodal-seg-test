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
python -m pip install "gradio==4.44.1" "gradio_client>=1.3.0,<1.5.0"

echo "[fix_amodal_deps] Verifying imports ..."
python "$AMODAL_ROOT/scripts/verify_amodal.py"

echo
echo "Next:"
echo "  source scripts/paths.env"
echo "  bash scripts/start_gradio.sh"

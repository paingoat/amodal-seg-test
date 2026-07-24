#!/usr/bin/env bash
# Start LISA Gradio server on port 7860 (lisa conda env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

eval "$(conda shell.bash hook)"
conda activate "$LISA_ENV_NAME"

cd "$LISA_REPO"
export LISA_SERVER_PORT
export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

MODEL_PATH="${LISA_MODEL_PATH:-$LISA_REPO/LISA-13B-llama2-v1}"
echo "[start_lisa] repo=$LISA_REPO model=$MODEL_PATH port=$LISA_SERVER_PORT"
echo "[start_lisa] Tip: after masks are cached, run scripts/stop_lisa.sh to free VRAM."

# Keep fp16 for quality parity with the original paper code.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python app.py \
  --version "$MODEL_PATH" \
  --precision fp16 \
  --vis_save_path "${LISA_VIS_PATH:-$AMODAL_ROOT/cache/lisa_vis}"

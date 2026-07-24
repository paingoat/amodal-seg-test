#!/usr/bin/env bash
# Launch the full-pipeline Gradio UI on port 7861 (amodal env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

eval "$(conda shell.bash hook)"
conda activate "$AMODAL_ENV_NAME"
cd "$AMODAL_ROOT"

export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE
export LISA_OUTPUT_PATH LISA_SERVER_URL AMODAL_ROOT OUTPUT_DIR
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "[start_gradio] Checking checkpoints ..."
python "$SCRIPT_DIR/verify_checkpoints.py" || {
  echo "[start_gradio] Abort: fix checkpoints first (0-byte files are common after a bad download)."
  echo "  rm -f -- \"$INSTAORDER_CKPT\""
  echo "  bash scripts/model.sh"
  exit 1
}

echo "[start_gradio] Checking GroundingDINO CUDA ops ..."
python "$SCRIPT_DIR/verify_groundingdino_ops.py" || {
  echo "[start_gradio] Abort: GroundingDINO _C extension is unavailable."
  echo "  bash scripts/fix_groundingdino_ops.sh"
  exit 1
}

echo "[start_gradio] http://0.0.0.0:${GRADIO_PIPELINE_PORT}  (LISA must use ${LISA_SERVER_PORT})"
python gradio_app.py --port "$GRADIO_PIPELINE_PORT"

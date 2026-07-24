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

echo "[start_gradio] Checking GroundingDINO attention ops ..."
# Prefer compiled _C; if missing, auto-enable pure-PyTorch MSDeformAttn fallback.
if ! python "$SCRIPT_DIR/verify_groundingdino_ops.py"; then
  echo "[start_gradio] Compiled _C missing — enabling PyTorch attention fallback ..."
  python "$SCRIPT_DIR/patch_groundingdino_pytorch_attn.py" || {
    echo "[start_gradio] Abort: cannot patch GroundingDINO attention fallback."
    echo "  Preferred: sudo bash scripts/install_cuda_toolkit_13_2.sh && bash scripts/fix_groundingdino_ops.sh"
    exit 1
  }
  export GDINO_ALLOW_PYTORCH_ATTN=1
  python "$SCRIPT_DIR/verify_groundingdino_ops.py" || {
    echo "[start_gradio] Abort: GroundingDINO attention still unavailable."
    exit 1
  }
fi

echo "[start_gradio] Patching InstaOrder for NumPy>=1.24 ..."
python "$SCRIPT_DIR/patch_instaorder_numpy.py" || {
  echo "[start_gradio] WARN: InstaOrder numpy patch failed (continuing)."
}

echo "[start_gradio] http://0.0.0.0:${GRADIO_PIPELINE_PORT}  (LISA must use ${LISA_SERVER_PORT})"
python gradio_app.py --port "$GRADIO_PIPELINE_PORT"

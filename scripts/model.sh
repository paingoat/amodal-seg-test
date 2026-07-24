#!/usr/bin/env bash
# Clone sibling repos and download all checkpoints needed by the pipeline.
# Does NOT download LaMa (removed from this fork).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

echo "============================================================"
echo " Download / prepare models under $MODELS_DIR"
echo "============================================================"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN is empty. Add HF_TOKEN to $AMODAL_ROOT/.env or create $AMODAL_ROOT/.hf_token"
  echo "      to reduce Hugging Face rate limits."
fi

mkdir -p "$MODELS_DIR" "$LISA_OUTPUT_PATH" "$HF_HOME"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

hf_download() {
  local repo="$1"
  local dest="$2"
  shift 2
  if have_cmd hf; then
    hf download "$repo" --local-dir "$dest" "$@"
  elif have_cmd huggingface-cli; then
    huggingface-cli download "$repo" --local-dir "$dest" "$@"
  else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' found."
    echo "Install: conda run -n $AMODAL_ENV_NAME python -m pip install -U 'huggingface_hub[cli]'"
    exit 1
  fi
}

clone_if_needed() {
  local url="$1"
  local dest="$2"
  if [ -d "$dest/.git" ] || [ -d "$dest" ]; then
    echo "[model] Exists: $dest"
  else
    echo "[model] Cloning $url -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "[model] Exists: $out"
    return
  fi
  mkdir -p "$(dirname "$out")"
  echo "[model] Downloading $url"
  if have_cmd wget; then
    wget -O "$out" "$url"
  else
    curl -L -o "$out" "$url"
  fi
}

# --- Sibling repos ---
clone_if_needed "https://github.com/dvlab-research/LISA.git" "$LISA_REPO"
clone_if_needed "https://github.com/POSTECH-CVLab/InstaOrder.git" "$INSTAORDER_REPO"
clone_if_needed "https://github.com/IDEA-Research/Grounded-Segment-Anything.git" "$GROUNDED_SAM_REPO"
clone_if_needed "https://github.com/xinyu1205/recognize-anything.git" "$RAM_REPO"

# Patch LISA app.py from this repo (returns raw mask for pipeline)
if [ -f "$AMODAL_ROOT/LISA/app.py" ]; then
  echo "[model] Installing modified LISA/app.py into $LISA_REPO"
  cp "$AMODAL_ROOT/LISA/app.py" "$LISA_REPO/app.py"
fi

# Symlink sibling repos into AMODAL_ROOT so main.py sys.path defaults work when cwd=AMODAL_ROOT.
# Note: do NOT symlink LISA here — repo-root LISA/app.py is the patch source; full clone is third_party/LISA.
link_sibling() {
  local name="$1"
  local src="$2"
  local dst="$AMODAL_ROOT/$name"
  if [ -L "$dst" ]; then
    echo "[model] Symlink exists: $dst"
  elif [ -e "$dst" ]; then
    echo "[model] Path exists (not linking): $dst"
  else
    ln -s "$src" "$dst"
    echo "[model] Symlinked $dst -> $src"
  fi
}
link_sibling "InstaOrder" "$INSTAORDER_REPO"
link_sibling "Grounded-Segment-Anything" "$GROUNDED_SAM_REPO"
link_sibling "recognize-anything" "$RAM_REPO"

# --- Checkpoints ---
# GroundingDINO
download_file \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
  "$GDINO_CKPT"

# SAM ViT-H
download_file \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
  "$SAM_CKPT"

# RAM++
download_file \
  "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth" \
  "$RAM_CKPT"

# InstaOrder checkpoint (official release)
mkdir -p "$(dirname "$INSTAORDER_CKPT")"
if [ ! -f "$INSTAORDER_CKPT" ]; then
  echo "[model] Downloading InstaOrder checkpoint ..."
  # Common mirror path used by community; if this fails, see GUIDE.md
  download_file \
    "https://github.com/POSTECH-CVLab/InstaOrder/releases/download/v1.0/InstaOrder_InstaOrderNet_od.pth.tar" \
    "$INSTAORDER_CKPT" || true
  if [ ! -f "$INSTAORDER_CKPT" ]; then
    echo "WARN: InstaOrder ckpt auto-download failed."
    echo "      Place InstaOrder_InstaOrderNet_od.pth.tar at:"
    echo "      $INSTAORDER_CKPT"
  fi
fi

# LISA-13B weights
if [ ! -d "$LISA_MODEL_PATH" ] || [ -z "$(ls -A "$LISA_MODEL_PATH" 2>/dev/null || true)" ]; then
  echo "[model] Downloading xinlai/LISA-13B-llama2-v1 -> $LISA_MODEL_PATH"
  hf_download "xinlai/LISA-13B-llama2-v1" "$LISA_MODEL_PATH"
else
  echo "[model] Exists: $LISA_MODEL_PATH"
fi

# SD2 inpainting community mirror (into HF cache)
echo "[model] Prefetching $SD_MODEL_ID into HF_HOME=$HF_HOME"
if have_cmd hf; then
  hf download "$SD_MODEL_ID"
elif have_cmd huggingface-cli; then
  huggingface-cli download "$SD_MODEL_ID"
else
  conda run -n "$AMODAL_ENV_NAME" python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(os.environ.get("SD_MODEL_ID", "sd2-community/stable-diffusion-2-inpainting"))
print("SD model downloaded")
PY
fi

# Install sibling packages into amodal env (best-effort)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  echo "[model] Installing GroundingDINO / segment_anything / RAM into env $AMODAL_ENV_NAME ..."
  # Install declared runtime deps first (editable install often skips them if already partially present)
  conda run -n "$AMODAL_ENV_NAME" python -m pip install \
    "addict>=2.4.0" "yapf>=0.40.0" "pycocotools>=2.0.6" "supervision>=0.22.0" || true
  if [ -f "$GROUNDED_SAM_REPO/GroundingDINO/requirements.txt" ]; then
    conda run -n "$AMODAL_ENV_NAME" python -m pip install -r "$GROUNDED_SAM_REPO/GroundingDINO/requirements.txt" || true
  fi
  conda run -n "$AMODAL_ENV_NAME" python -m pip install -e "$GROUNDED_SAM_REPO/GroundingDINO" || \
    echo "WARN: GroundingDINO editable install failed (see GUIDE.md)."
  conda run -n "$AMODAL_ENV_NAME" python -m pip install -e "$GROUNDED_SAM_REPO/segment_anything" || \
    conda run -n "$AMODAL_ENV_NAME" python -m pip install git+https://github.com/facebookresearch/segment-anything.git || true
  conda run -n "$AMODAL_ENV_NAME" python -m pip install -e "$RAM_REPO" || true
  # InstaOrder is imported via sys.path; ensure pycocotools present
  conda run -n "$AMODAL_ENV_NAME" python -m pip install "pycocotools>=2.0.6" || true
fi

echo
echo "[model] Done."
echo "  LISA model : $LISA_MODEL_PATH"
echo "  GDINO ckpt : $GDINO_CKPT"
echo "  SAM ckpt   : $SAM_CKPT"
echo "  RAM ckpt   : $RAM_CKPT"
echo "  InstaOrder : $INSTAORDER_CKPT"
echo "  SD model   : $SD_MODEL_ID"

if command -v conda >/dev/null 2>&1; then
  echo
  echo "[model] Full amodal import check (pip + siblings) ..."
  conda run -n "$AMODAL_ENV_NAME" python "$AMODAL_ROOT/scripts/verify_amodal.py" || \
    echo "WARN: verify_amodal reported missing packages — run fix_amodal_deps.sh / re-check installs."
fi

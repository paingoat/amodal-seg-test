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

# download_file URL OUT [MIN_BYTES]
# Rejects empty / tiny files (failed downloads often leave 0-byte stubs).
download_file() {
  local url="$1"
  local out="$2"
  local min_bytes="${3:-1048576}"  # 1 MiB default
  if [ -f "$out" ]; then
    local sz
    sz="$(wc -c <"$out" | tr -d '[:space:]')"
    if [ "$sz" -ge "$min_bytes" ]; then
      echo "[model] Exists: $out ($sz bytes)"
      return 0
    fi
    echo "[model] Removing undersized stub ($sz < $min_bytes bytes): $out"
    rm -f "$out"
  fi
  mkdir -p "$(dirname "$out")"
  echo "[model] Downloading $url"
  local tmp="${out}.partial"
  rm -f "$tmp"
  if have_cmd wget; then
    wget -O "$tmp" "$url"
  else
    curl -L --fail -o "$tmp" "$url"
  fi
  local sz
  sz="$(wc -c <"$tmp" | tr -d '[:space:]')"
  if [ ! -f "$tmp" ] || [ "$sz" -lt "$min_bytes" ]; then
    echo "ERROR: download too small ($sz bytes, need ≥ $min_bytes): $url"
    rm -f "$tmp"
    return 1
  fi
  mv -f "$tmp" "$out"
  echo "[model] Saved: $out ($sz bytes)"
}

# InstaOrderNet_od — GitHub "releases/v1.0" URL 404s; use Drive (SNU mirror of od weights).
download_instaorder_ckpt() {
  local out="$INSTAORDER_CKPT"
  local min_bytes=52428800  # 50 MiB
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    local sz
    sz="$(wc -c <"$out" | tr -d '[:space:]')"
    if [ "$sz" -ge "$min_bytes" ]; then
      echo "[model] Exists: $out ($sz bytes)"
      return 0
    fi
    echo "[model] Removing undersized InstaOrder stub ($sz bytes): $out"
    rm -f "$out"
  fi

  echo "[model] Downloading InstaOrderNet_od via gdown (Google Drive) ..."
  # InstaOrderNet o,d — pass file id (gdown>=6 removed --fuzzy)
  local gdown_id="1QLikFxNOEW1Ld2oAZff8mL26FO4Mwwpv"
  if conda run -n "$AMODAL_ENV_NAME" python -m pip show gdown >/dev/null 2>&1 || \
     conda run -n "$AMODAL_ENV_NAME" python -m pip install -q "gdown>=5.0"; then
    if conda run -n "$AMODAL_ENV_NAME" python -m gdown "$gdown_id" -O "$out"; then
      local sz
      sz="$(wc -c <"$out" | tr -d '[:space:]')"
      if [ "$sz" -ge "$min_bytes" ]; then
        echo "[model] Saved: $out ($sz bytes)"
        return 0
      fi
      rm -f "$out"
    fi
  fi

  echo "WARN: InstaOrder auto-download failed."
  echo "      Manual options:"
  echo "        1) source scripts/paths.env"
  echo "           python -m gdown $gdown_id -O \"\$INSTAORDER_CKPT\""
  echo "        2) Full pack (3.5G): https://drive.google.com/file/d/1_GEmCmofLSkJZnidfp4vsQb2Nqq5aqBU/view"
  echo "           unzip and copy InstaOrder_InstaOrderNet_od.pth.tar → $out"
  return 1
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

# --- Checkpoints (min sizes reject 0-byte / HTML error pages) ---
# GroundingDINO ~694 MiB
download_file \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
  "$GDINO_CKPT" \
  104857600

# SAM ViT-H ~2.4 GiB
download_file \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
  "$SAM_CKPT" \
  1073741824

# RAM++
download_file \
  "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth" \
  "$RAM_CKPT" \
  524288000

# InstaOrderNet_od (Drive; old GitHub release URL is 404 → 0-byte stub)
download_instaorder_ckpt || true

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
  bash "$SCRIPT_DIR/fix_groundingdino_ops.sh" || \
    echo "WARN: GroundingDINO CUDA ops build failed (run scripts/fix_groundingdino_ops.sh)."
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

echo
echo "[model] Verifying checkpoint file sizes ..."
python3 "$AMODAL_ROOT/scripts/verify_checkpoints.py" || \
  echo "WARN: verify_checkpoints failed — fix downloads above before Gradio/main."

if command -v conda >/dev/null 2>&1; then
  echo
  echo "[model] Full amodal import check (pip + siblings) ..."
  conda run -n "$AMODAL_ENV_NAME" python "$AMODAL_ROOT/scripts/verify_amodal.py" || \
    echo "WARN: verify_amodal reported missing packages — run fix_amodal_deps.sh / re-check installs."
fi

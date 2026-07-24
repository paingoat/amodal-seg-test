#!/usr/bin/env bash
# Full two-stage batch: LISA masks -> stop LISA -> main pipeline batches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

cd "$AMODAL_ROOT"

IMG_FILENAMES_TXT="${IMG_FILENAMES_TXT:-./img_filenames_example.txt}"
JSON_LABEL_PATH="${JSON_LABEL_PATH:-./example_annotation.json}"
INPUT_DIR="${INPUT_DIR:-$AMODAL_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$AMODAL_ROOT/output}"
ROUND_NUMBER="${ROUND_NUMBER:-5}"

eval "$(conda shell.bash hook)"

echo "============================================================"
echo " Stage A: ensure LISA server is up, then batch masks"
echo "============================================================"
echo "If LISA is not running, start it in another terminal:"
echo "  bash $SCRIPT_DIR/start_lisa.sh"
echo
read -r -p "Press Enter once LISA is ready on port $LISA_SERVER_PORT ..."

conda activate "$AMODAL_ENV_NAME"
python "$SCRIPT_DIR/run_lisa_batch.py" \
  --input_dir "$INPUT_DIR" \
  --img_filenames_txt "$IMG_FILENAMES_TXT" \
  --json_label_path "$JSON_LABEL_PATH" \
  --lisa_mask_dir "$LISA_OUTPUT_PATH" \
  --lisa_server_url "$LISA_SERVER_URL"

echo "============================================================"
echo " Stop LISA to free ~VRAM before main pipeline"
echo "============================================================"
bash "$SCRIPT_DIR/stop_lisa.sh"

echo "============================================================"
echo " Stage B: main amodal pipeline (batches of $ROUND_NUMBER)"
echo "============================================================"
line_count=$(wc -l < "$IMG_FILENAMES_TXT")
for line_num in $(seq 0 "$ROUND_NUMBER" "$line_count"); do
  echo "Processing batch starting at line: $line_num"
  python main.py \
    --input_dir "$INPUT_DIR" \
    --img_filenames_txt "$IMG_FILENAMES_TXT" \
    --json_label_path "$JSON_LABEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lisa_mask_dir "$LISA_OUTPUT_PATH" \
    --skip_lisa_server \
    --require_lisa_cache \
    --line_num "$line_num" \
    --round_number "$ROUND_NUMBER"
done

echo "[run_full_batch] Done. Outputs under $OUTPUT_DIR"

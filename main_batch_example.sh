#!/bin/bash
# Example CLI batch runner (main pipeline only).
# Prefer scripts/run_full_batch.sh for LISA → stop → main on a single GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If this file still lives at repo root, scripts/ is ./scripts
if [ -f "$SCRIPT_DIR/scripts/paths.env" ]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/scripts/paths.env"
  ROOT="$SCRIPT_DIR"
else
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/paths.env"
  ROOT="$AMODAL_ROOT"
fi

cd "$ROOT"

img_filenames_txt="./img_filenames_example.txt"
INPUT_DIR="${INPUT_DIR:-$ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-$AMODAL_ROOT/output}"
JSON_LABEL_PATH="${JSON_LABEL_PATH:-./example_annotation.json}"
ROUND_NUMBER="${ROUND_NUMBER:-5}"

line_count=$(wc -l < "$img_filenames_txt")

for line_num in $(seq 0 "$ROUND_NUMBER" "$line_count"); do
    echo "Processing batch starting from line: $line_num"

    python main.py \
        --input_dir "$INPUT_DIR" \
        --img_filenames_txt "$img_filenames_txt" \
        --json_label_path "$JSON_LABEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --lisa_mask_dir "$LISA_OUTPUT_PATH" \
        --skip_lisa_server \
        --require_lisa_cache \
        --line_num "$line_num" \
        --round_number "$ROUND_NUMBER"

    echo "Batch starting at $line_num finished."
done

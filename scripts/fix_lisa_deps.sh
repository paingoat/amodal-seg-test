#!/usr/bin/env bash
# Repair Gradio 3.39 stack inside the existing `lisa` conda env.
# Use when start_lisa.sh fails on pkg_resources / FieldInfo.in_ / gradio_client.serializing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/paths.env"

eval "$(conda shell.bash hook)"
conda activate "$LISA_ENV_NAME"

echo "[fix_lisa_deps] Repairing Gradio 3.39 cluster in env: $LISA_ENV_NAME"

python -m pip install \
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

python - <<'PY'
import pydantic, fastapi, gradio, gradio_client
from gradio_client.serializing import JSONSerializable  # noqa: F401
print("pydantic", pydantic.__version__)
print("fastapi", fastapi.__version__)
print("gradio", gradio.__version__)
print("gradio_client", gradio_client.__version__)
assert pydantic.__version__.startswith("1.")
assert gradio.__version__.startswith("3.39")
assert gradio_client.__version__.startswith("0.5")
print("OK: LISA Gradio stack repaired")
PY

echo
echo "Next:"
echo "  source scripts/paths.env"
echo "  bash scripts/start_lisa.sh"

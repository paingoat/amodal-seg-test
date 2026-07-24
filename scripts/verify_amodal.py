#!/usr/bin/env python3
"""Smoke-check packages needed by main.py / gradio_app.py (amodal env)."""
from __future__ import annotations

import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Sibling import paths (same as main.py)
for p in (
    "InstaOrder",
    "Grounded-Segment-Anything",
    "Grounded-Segment-Anything/GroundingDINO",
):
    full = os.path.join(REPO_ROOT, p)
    if os.path.isdir(full) and full not in sys.path:
        sys.path.append(p if not os.path.isabs(p) else full)
    # Prefer appending relative names like main.py
sys.path.append("InstaOrder")
sys.path.append("Grounded-Segment-Anything")
sys.path.append("Grounded-Segment-Anything/GroundingDINO")


def _must(name: str, import_name: str | None = None) -> None:
    mod = import_name or name
    importlib.import_module(mod)
    print(f"  OK  {name}")


def main() -> int:
    print("[verify_amodal] core pip packages")
    errors: list[str] = []

    checks = [
        ("torch", "torch"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("PIL", "PIL"),
        ("skimage", "skimage"),
        ("tqdm", "tqdm"),
        ("gradio", "gradio"),
        ("gradio_client", "gradio_client"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("accelerate", "accelerate"),
        ("clip (OpenAI)", "clip"),
    ]
    for label, mod in checks:
        try:
            _must(label, mod)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            print(f"  FAIL {label}: {exc}")

    try:
        from diffusers import StableDiffusionInpaintPipeline  # noqa: F401

        import diffusers
        import transformers

        assert transformers.__version__.startswith("4.46."), transformers.__version__
        assert diffusers.__version__.startswith("0.32."), diffusers.__version__
        print(
            f"  OK  StableDiffusionInpaintPipeline "
            f"(transformers={transformers.__version__}, diffusers={diffusers.__version__})"
        )
    except Exception as exc:
        errors.append(f"StableDiffusionInpaintPipeline: {exc}")
        print(f"  FAIL StableDiffusionInpaintPipeline: {exc}")

    print("[verify_amodal] sibling packages (need scripts/model.sh)")
    sibling_checks = [
        ("InstaOrder.models", "models"),
        ("GroundingDINO", "GroundingDINO.groundingdino.models"),
        ("segment_anything", "segment_anything"),
        ("ram", "ram"),
    ]
    for label, mod in sibling_checks:
        try:
            _must(label, mod)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            print(f"  FAIL {label}: {exc}")

    if errors:
        print("\n[verify_amodal] FAILED:")
        for e in errors:
            print(f"  - {e}")
        print("\nHints:")
        print("  pip/core:  bash scripts/fix_amodal_deps.sh")
        print("  siblings:  bash scripts/model.sh")
        return 1

    print("\n[verify_amodal] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

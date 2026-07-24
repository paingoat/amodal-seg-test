#!/usr/bin/env python3
"""Fail fast when GroundingDINO's compiled C++/CUDA extension is unavailable."""
from __future__ import annotations

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "third_party"))
GROUNDED_SAM_REPO = os.environ.get(
    "GROUNDED_SAM_REPO",
    os.path.join(MODELS_DIR, "Grounded-Segment-Anything"),
)
GROUNDING_DINO_REPO = os.path.join(GROUNDED_SAM_REPO, "GroundingDINO")

if os.path.isdir(GROUNDING_DINO_REPO):
    sys.path.insert(0, GROUNDING_DINO_REPO)


def main() -> int:
    try:
        import torch

        print(
            "[verify_groundingdino_ops] "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"cuda_available={torch.cuda.is_available()}"
        )
        if torch.cuda.is_available():
            print(
                "[verify_groundingdino_ops] "
                f"device={torch.cuda.get_device_name(0)} "
                f"capability={torch.cuda.get_device_capability(0)}"
            )

        from groundingdino import _C

        required = ("ms_deform_attn_forward", "ms_deform_attn_backward")
        missing = [name for name in required if not hasattr(_C, name)]
        if missing:
            raise ImportError(f"compiled _C is missing symbols: {', '.join(missing)}")

        print(f"[verify_groundingdino_ops] OK: {_C.__file__}")
        return 0
    except Exception as exc:
        print(f"[verify_groundingdino_ops] FAIL: {type(exc).__name__}: {exc}")
        print("GroundingDINO custom CUDA ops were not built or cannot be loaded.")
        print("Fix: bash scripts/fix_groundingdino_ops.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())

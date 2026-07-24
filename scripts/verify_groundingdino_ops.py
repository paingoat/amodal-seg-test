#!/usr/bin/env python3
"""Fail fast when GroundingDINO cannot run deformable attention."""
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
MS_ATTN = os.path.join(
    GROUNDING_DINO_REPO,
    "groundingdino",
    "models",
    "GroundingDINO",
    "ms_deform_attn.py",
)

if os.path.isdir(GROUNDING_DINO_REPO):
    sys.path.insert(0, GROUNDING_DINO_REPO)


def _pytorch_fallback_ready() -> bool:
    # Env alone is not enough — the source file must actually fall back when _C is None.
    if not os.path.isfile(MS_ATTN):
        return False
    text = open(MS_ATTN, "r", encoding="utf-8").read()
    return "_C = None" in text and "use_cuda_ops" in text


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

        try:
            from groundingdino import _C

            required = ("ms_deform_attn_forward", "ms_deform_attn_backward")
            missing = [name for name in required if not hasattr(_C, name)]
            if missing:
                raise ImportError(f"compiled _C is missing symbols: {', '.join(missing)}")
            print(f"[verify_groundingdino_ops] OK: compiled ops @ {_C.__file__}")
            return 0
        except Exception as cuda_exc:
            if _pytorch_fallback_ready():
                print(
                    "[verify_groundingdino_ops] WARN: compiled _C unavailable "
                    f"({type(cuda_exc).__name__}: {cuda_exc})"
                )
                print(
                    "[verify_groundingdino_ops] OK: pure-PyTorch MSDeformAttn fallback enabled"
                )
                return 0
            raise
    except Exception as exc:
        print(f"[verify_groundingdino_ops] FAIL: {type(exc).__name__}: {exc}")
        print("GroundingDINO custom CUDA ops were not built and no PyTorch fallback is active.")
        print("Preferred fix (matching toolkit):")
        print("  sudo bash scripts/install_cuda_toolkit_13_2.sh")
        print("  export CUDA_HOME=/usr/local/cuda-13.2")
        print("  bash scripts/fix_groundingdino_ops.sh")
        print("Quick workaround (no toolkit compile):")
        print("  python scripts/patch_groundingdino_pytorch_attn.py")
        print("  export GDINO_ALLOW_PYTORCH_ATTN=1")
        return 1


if __name__ == "__main__":
    sys.exit(main())

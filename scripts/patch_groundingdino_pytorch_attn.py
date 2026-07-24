#!/usr/bin/env python3
"""Patch GroundingDINO to fall back to pure-PyTorch MSDeformAttn when _C is missing.

This unblocks RTX 5090 / CUDA 13.2 machines that only have an older system nvcc
and cannot yet compile groundingdino._C. Slightly slower than the CUDA op, but
correct for inference.
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "third_party"))
GROUNDED_SAM_REPO = os.environ.get(
    "GROUNDED_SAM_REPO",
    os.path.join(MODELS_DIR, "Grounded-Segment-Anything"),
)
TARGET = os.path.join(
    GROUNDED_SAM_REPO,
    "GroundingDINO",
    "groundingdino",
    "models",
    "GroundingDINO",
    "ms_deform_attn.py",
)

OLD_IMPORT = """try:
    from groundingdino import _C
except:
    warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")
"""

NEW_IMPORT = """try:
    from groundingdino import _C
except Exception:
    _C = None
    warnings.warn(
        "Failed to load custom C++ ops. Falling back to pure-PyTorch MSDeformAttn."
    )
"""

OLD_BRANCH = """        if torch.cuda.is_available() and value.is_cuda:
            halffloat = False
            if value.dtype == torch.float16:
                halffloat = True
                value = value.float()
                sampling_locations = sampling_locations.float()
                attention_weights = attention_weights.float()

            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

            if halffloat:
                output = output.half()
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
"""

NEW_BRANCH = """        use_cuda_ops = (
            _C is not None and torch.cuda.is_available() and value.is_cuda
        )
        if use_cuda_ops:
            halffloat = False
            if value.dtype == torch.float16:
                halffloat = True
                value = value.float()
                sampling_locations = sampling_locations.float()
                attention_weights = attention_weights.float()

            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

            if halffloat:
                output = output.half()
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
"""

MARKER = "use_cuda_ops = ("


def main() -> int:
    if not os.path.isfile(TARGET):
        print(f"[patch_gdino_attn] ERROR: missing {TARGET}")
        print("Run: bash scripts/model.sh")
        return 1

    text = open(TARGET, "r", encoding="utf-8").read()
    if MARKER in text and "_C = None" in text:
        print(f"[patch_gdino_attn] Already patched: {TARGET}")
        return 0

    updated = text
    if OLD_IMPORT in updated:
        updated = updated.replace(OLD_IMPORT, NEW_IMPORT, 1)
    elif "from groundingdino import _C" in updated and "_C = None" not in updated:
        # tolerant replace for slight whitespace differences
        updated = updated.replace(
            "except:\n    warnings.warn(\"Failed to load custom C++ ops. Running on CPU mode Only!\")",
            "except Exception:\n    _C = None\n    warnings.warn(\n"
            "        \"Failed to load custom C++ ops. Falling back to pure-PyTorch MSDeformAttn.\"\n"
            "    )",
            1,
        )

    if OLD_BRANCH in updated:
        updated = updated.replace(OLD_BRANCH, NEW_BRANCH, 1)
    elif "if torch.cuda.is_available() and value.is_cuda:" in updated and MARKER not in updated:
        updated = updated.replace(
            "if torch.cuda.is_available() and value.is_cuda:",
            "use_cuda_ops = (_C is not None and torch.cuda.is_available() and value.is_cuda)\n"
            "        if use_cuda_ops:",
            1,
        )

    if updated == text:
        print("[patch_gdino_attn] ERROR: expected patterns not found; file layout changed.")
        print(f"Inspect: {TARGET}")
        return 1

    bak = TARGET + ".bak_pre_pytorch_attn"
    if not os.path.isfile(bak):
        open(bak, "w", encoding="utf-8").write(text)
    open(TARGET, "w", encoding="utf-8").write(updated)
    print(f"[patch_gdino_attn] Patched: {TARGET}")
    print("[patch_gdino_attn] Backup :", bak)
    print("[patch_gdino_attn] GroundingDINO can now run without compiled _C.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

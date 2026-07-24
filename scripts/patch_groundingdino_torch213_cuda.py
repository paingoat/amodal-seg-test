#!/usr/bin/env python3
"""Patch GroundingDINO CUDA sources for PyTorch >=2.1 (value.type() removed)."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "third_party"))
GROUNDED_SAM_REPO = os.environ.get(
    "GROUNDED_SAM_REPO",
    os.path.join(MODELS_DIR, "Grounded-Segment-Anything"),
)
CSRC = os.path.join(
    GROUNDED_SAM_REPO,
    "GroundingDINO",
    "groundingdino",
    "models",
    "GroundingDINO",
    "csrc",
    "MsDeformAttn",
)
CUDA_CU = os.path.join(CSRC, "ms_deform_attn_cuda.cu")
HDR = os.path.join(CSRC, "ms_deform_attn.h")


def _patch_file(path: str, replacements: list[tuple[str, str]]) -> bool:
    if not os.path.isfile(path):
        print(f"[patch_gdino_cuda] ERROR: missing {path}")
        return False
    text = open(path, "r", encoding="utf-8").read()
    original = text
    for old, new in replacements:
        if new in text and old not in text:
            continue
        if old not in text:
            # Already patched or unexpected layout.
            if new.split("(")[0] in text:
                continue
            print(f"[patch_gdino_cuda] WARN: pattern not found in {path}: {old[:60]}...")
            continue
        text = text.replace(old, new)
    if text == original:
        print(f"[patch_gdino_cuda] Already patched or unchanged: {path}")
        return True
    bak = path + ".bak_pre_torch213"
    if not os.path.isfile(bak):
        open(bak, "w", encoding="utf-8").write(original)
    open(path, "w", encoding="utf-8").write(text)
    print(f"[patch_gdino_cuda] Patched: {path}")
    return True


def main() -> int:
    ok = True
    ok &= _patch_file(
        CUDA_CU,
        [
            (
                'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"',
                'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda"',
            ),
            (
                'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"',
                'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda"',
            ),
        ],
    )
    ok &= _patch_file(
        HDR,
        [
            ("if (value.type().is_cuda())", "if (value.is_cuda())"),
        ],
    )
    if not ok:
        return 1
    print("[patch_gdino_cuda] Done. Rebuild with: bash scripts/fix_groundingdino_ops.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())

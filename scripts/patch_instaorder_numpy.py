#!/usr/bin/env python3
"""Patch InstaOrder for NumPy>=1.24 (np.int / np.float / np.bool removed)."""
from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "third_party"))
INSTAORDER_REPO = os.environ.get(
    "INSTAORDER_REPO", os.path.join(MODELS_DIR, "InstaOrder")
)

# Safe replacements for inference/runtime code.
REPLACEMENTS = [
    (re.compile(r"\bnp\.int\b"), "int"),
    (re.compile(r"\bnp\.float\b"), "float"),
    (re.compile(r"\bnp\.complex\b"), "complex"),
    (re.compile(r"\bnp\.bool\b"), "bool"),
    (re.compile(r"\bnp\.object\b"), "object"),
    (re.compile(r"\bnp\.str\b"), "str"),
]


def patch_file(path: str) -> bool:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    original = text
    for pattern, repl in REPLACEMENTS:
        text = pattern.sub(repl, text)
    if text == original:
        return False
    bak = path + ".bak_pre_numpy124"
    if not os.path.isfile(bak):
        open(bak, "w", encoding="utf-8").write(original)
    open(path, "w", encoding="utf-8").write(text)
    print(f"[patch_instaorder_numpy] Patched {path}")
    return True


def main() -> int:
    roots = []
    for candidate in (
        INSTAORDER_REPO,
        os.path.join(REPO_ROOT, "InstaOrder"),
    ):
        if os.path.isdir(candidate) and candidate not in roots:
            roots.append(candidate)

    if not roots:
        print("[patch_instaorder_numpy] ERROR: InstaOrder not found.")
        print("Run: bash scripts/model.sh")
        return 1

    changed = 0
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if any(skip in dirpath for skip in (".git", "__pycache__", ".bak")):
                continue
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                if patch_file(os.path.join(dirpath, name)):
                    changed += 1

    if changed == 0:
        print("[patch_instaorder_numpy] No changes needed (already patched or clean).")
    else:
        print(f"[patch_instaorder_numpy] Updated {changed} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

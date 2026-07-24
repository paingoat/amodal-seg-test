#!/usr/bin/env python3
"""Verify local model checkpoints are present and not empty/truncated.

Run (after source scripts/paths.env, or from repo root):
  python scripts/verify_checkpoints.py

Exit 0 if all required weights look sane; exit 1 otherwise.
"""
from __future__ import annotations

import os
import sys

MB = 1024 * 1024
GB = 1024 * MB

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip().lstrip("\ufeff")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _human(n: int) -> str:
    if n >= GB:
        return f"{n / GB:.2f} GiB"
    if n >= MB:
        return f"{n / MB:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _check_file(label: str, path: str, min_bytes: int) -> str | None:
    if not path:
        return f"{label}: path empty"
    if not os.path.isfile(path):
        return f"{label}: missing → {path}"
    size = os.path.getsize(path)
    if size < min_bytes:
        return (
            f"{label}: too small ({_human(size)} < {_human(min_bytes)}) → {path}\n"
            f"         (0-byte / HTML error page from a failed download is common)"
        )
    print(f"  OK  {label}: {_human(size)}  ({path})")
    return None


def _check_dir(label: str, path: str, min_total_bytes: int, min_file_bytes: int) -> str | None:
    if not path:
        return f"{label}: path empty"
    if not os.path.isdir(path):
        return f"{label}: missing dir → {path}"
    total = 0
    largest = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            total += sz
            largest = max(largest, sz)
    if total < min_total_bytes or largest < min_file_bytes:
        return (
            f"{label}: incomplete ({_human(total)} total, largest {_human(largest)}) → {path}\n"
            f"         need ≥ {_human(min_total_bytes)} total and a file ≥ {_human(min_file_bytes)}"
        )
    print(f"  OK  {label}: {_human(total)} total  ({path})")
    return None


def main() -> int:
    _load_dotenv(os.path.join(REPO_ROOT, ".env"))
    models = _env("MODELS_DIR", os.path.join(REPO_ROOT, "third_party"))
    grounded = _env(
        "GROUNDED_SAM_REPO", os.path.join(models, "Grounded-Segment-Anything")
    )
    insta = _env("INSTAORDER_REPO", os.path.join(models, "InstaOrder"))
    ram = _env("RAM_REPO", os.path.join(models, "recognize-anything"))
    lisa_repo = _env("LISA_REPO", os.path.join(models, "LISA"))

    gdino = _env("GDINO_CKPT", os.path.join(grounded, "groundingdino_swint_ogc.pth"))
    sam = _env("SAM_CKPT", os.path.join(grounded, "sam_vit_h_4b8939.pth"))
    ram_ckpt = _env("RAM_CKPT", os.path.join(ram, "ram_plus_swin_large_14m.pth"))
    insta_ckpt = _env(
        "INSTAORDER_CKPT",
        os.path.join(insta, "InstaOrder_ckpt", "InstaOrder_InstaOrderNet_od.pth.tar"),
    )
    lisa_model = _env("LISA_MODEL_PATH", os.path.join(lisa_repo, "LISA-13B-llama2-v1"))
    gdino_cfg = _env(
        "GDINO_CONFIG",
        os.path.join(
            grounded,
            "GroundingDINO",
            "groundingdino",
            "config",
            "GroundingDINO_SwinT_OGC.py",
        ),
    )

    lisa_only = "--lisa-only" in sys.argv
    require_lisa = "--require-lisa" in sys.argv or lisa_only
    errors: list[str] = []

    if not lisa_only:
        print("[verify_checkpoints] pipeline weight files")
        # Mins catch empty/truncated downloads; real files are larger.
        checks = [
            ("GroundingDINO ckpt", gdino, 100 * MB),  # ~694 MiB
            ("SAM ViT-H ckpt", sam, 1 * GB),  # ~2.4 GiB
            ("RAM++ ckpt", ram_ckpt, 500 * MB),  # ~3 GiB class
            ("InstaOrderNet_od ckpt", insta_ckpt, 50 * MB),  # ResNet-50 ~100+ MiB
        ]
        for label, path, vmin in checks:
            err = _check_file(label, path, vmin)
            if err:
                errors.append(err)
                print(f"  FAIL {err.split(chr(10))[0]}")

        if not os.path.isfile(gdino_cfg):
            errors.append(f"GroundingDINO config: missing → {gdino_cfg}")
            print(f"  FAIL GroundingDINO config: missing → {gdino_cfg}")
        else:
            print(f"  OK  GroundingDINO config  ({gdino_cfg})")

    # LISA is only required for the mask-server step; warn for Gradio main unless flagged.
    print(
        "[verify_checkpoints] LISA weights"
        + (" (required)" if require_lisa else " (optional for Gradio main)")
    )
    lisa_err = _check_dir(
        "LISA-13B", lisa_model, min_total_bytes=5 * GB, min_file_bytes=100 * MB
    )
    if lisa_err:
        if require_lisa:
            errors.append(lisa_err)
            print(f"  FAIL {lisa_err.split(chr(10))[0]}")
        else:
            print(f"  WARN {lisa_err.split(chr(10))[0]}")

    if errors:
        print("\n[verify_checkpoints] FAILED:")
        for e in errors:
            print(f"  - {e}")
        print("\nFix:")
        print("  # remove broken 0-byte stubs, then re-download")
        print("  rm -f -- \"$INSTAORDER_CKPT\"   # or the failing path above")
        print("  source scripts/paths.env && bash scripts/model.sh")
        print("  python scripts/verify_checkpoints.py")
        return 1

    print("\n[verify_checkpoints] ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

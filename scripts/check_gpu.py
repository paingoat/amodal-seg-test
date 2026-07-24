#!/usr/bin/env python3
"""Verify CUDA + Blackwell (sm_120) support for RTX 5090."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: torch not importable: {exc}")
        return 1

    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"device={name}")
    print(f"capability=({major}, {minor})")
    print(f"arch_list={arch_list}")

    if "sm_120" not in arch_list and not any(a.startswith("sm_120") for a in arch_list):
        print(
            "WARN: sm_120 not in torch.cuda.get_arch_list(). "
            "Reinstall torch from cu128/cu130/cu132 index."
        )
        return 2

    x = torch.randn(64, 64, device="cuda")
    y = x @ x.T
    print(f"matmul_ok=True sum={float(y.sum()):.4f}")
    print("OK: GPU ready for RTX 5090 / sm_120")
    return 0


if __name__ == "__main__":
    sys.exit(main())

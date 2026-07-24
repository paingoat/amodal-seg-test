#!/usr/bin/env python3
"""
Batch-extract LISA visible masks via the Gradio server and cache them as pickles.

Run with the *amodal* or *lisa* env that has gradio_client installed.
Requires the LISA server to already be running (scripts/start_lisa.sh).

After this finishes, run scripts/stop_lisa.sh before main.py / gradio_app.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from gradio_client import Client
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


_load_dotenv(os.path.join(_REPO_ROOT, ".env"))


def read_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]


def stem_path(img_filename: str) -> str:
    return img_filename.split(".")[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch LISA mask extraction")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--img_filenames_txt", type=str, required=True)
    parser.add_argument("--json_label_path", type=str, required=True)
    parser.add_argument(
        "--lisa_mask_dir",
        type=str,
        default=os.environ.get(
            "LISA_OUTPUT_PATH",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cache",
                "lisa_masks",
            ),
        ),
    )
    parser.add_argument(
        "--lisa_server_url",
        type=str,
        default=os.environ.get("LISA_SERVER_URL", "http://127.0.0.1:7860/"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--line_num", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1, help="-1 = all remaining")
    args = parser.parse_args()

    os.makedirs(args.lisa_mask_dir, exist_ok=True)

    with open(args.json_label_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    prompt_map = {}
    for ann in label_data["annotations"]:
        base = ann["filename"].split("/")[-1].split(".")[0]
        prompt_map[base] = ann["labels"][0]

    filenames = read_txt(args.img_filenames_txt)
    if args.line_num:
        filenames = filenames[args.line_num :]
    if args.limit is not None and args.limit >= 0:
        filenames = filenames[: args.limit]

    print(f"[lisa_batch] server={args.lisa_server_url}")
    print(f"[lisa_batch] cache={args.lisa_mask_dir}")
    print(f"[lisa_batch] images={len(filenames)}")

    # Wait briefly for server readiness
    client = None
    for attempt in range(30):
        try:
            client = Client(args.lisa_server_url)
            break
        except Exception as exc:
            print(f"[lisa_batch] waiting for server ({attempt+1}/30): {exc}")
            time.sleep(2)
    if client is None:
        print("ERROR: could not connect to LISA server. Start it with scripts/start_lisa.sh")
        return 1

    manifest = {"items": []}
    for img_filename in filenames:
        base = img_filename.split("/")[-1].split(".")[0]
        if base not in prompt_map:
            print(f"[skip] no label for {img_filename}")
            continue

        out_pkl = os.path.join(args.lisa_mask_dir, f"{stem_path(img_filename)}.pkl")
        os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
        if os.path.isfile(out_pkl) and not args.overwrite:
            print(f"[skip] exists {out_pkl}")
            manifest["items"].append({"filename": img_filename, "pkl": out_pkl, "status": "cached"})
            continue

        img_path = os.path.join(args.input_dir, img_filename)
        text_query = prompt_map[base]
        print(f"[run] {img_filename} | query={text_query}")
        try:
            result = client.predict(text_query, img_path, api_name="/predict")
            with open(result[1], "r", encoding="utf-8") as jf:
                mask_json = json.load(jf)
            mask = np.array(mask_json["data"])
            with open(out_pkl, "wb") as pf:
                pickle.dump(mask, pf)

            # Save overlay if provided
            overlay_path = out_pkl.replace(".pkl", "_overlay.png")
            try:
                Image.open(result[0]).save(overlay_path)
            except Exception:
                Image.fromarray((mask.astype(np.uint8) * 255)).save(overlay_path)

            manifest["items"].append(
                {
                    "filename": img_filename,
                    "pkl": out_pkl,
                    "overlay": overlay_path,
                    "status": "ok",
                    "query": text_query,
                }
            )
            print(f"[ok] {out_pkl} shape={mask.shape}")
        except Exception as exc:
            print(f"[fail] {img_filename}: {exc}")
            manifest["items"].append(
                {"filename": img_filename, "status": "fail", "error": str(exc)}
            )

    manifest_path = os.path.join(args.lisa_mask_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[lisa_batch] wrote {manifest_path}")
    print("[lisa_batch] Next: bash scripts/stop_lisa.sh && run main.py / gradio_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

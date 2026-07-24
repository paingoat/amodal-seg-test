#!/usr/bin/env python3
"""
Gradio UI for the open-world amodal appearance completion pipeline.

Port: GRADIO_PIPELINE_PORT (default 7861) — do not use 7860 (reserved for LISA).

Recommended VRAM flow on a single 32GB GPU:
  1) Start LISA (scripts/start_lisa.sh) in another terminal
  2) Use "Cache LISA mask" here (or scripts/run_lisa_batch.py)
  3) Stop LISA (scripts/stop_lisa.sh)
  4) Run "Run main pipeline" in this UI
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import gradio as gr
import numpy as np
from PIL import Image

# Ensure repo root is importable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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


_load_dotenv(os.path.join(REPO_ROOT, ".env"))

import main as pipeline  # noqa: E402


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


AMODAL_ROOT = _env("AMODAL_ROOT", REPO_ROOT)
LISA_MASK_DIR = _env("LISA_OUTPUT_PATH", os.path.join(AMODAL_ROOT, "cache", "lisa_masks"))
LISA_SERVER_URL = _env("LISA_SERVER_URL", "http://127.0.0.1:7860/")
DEFAULT_OUTPUT = _env("OUTPUT_DIR", os.path.join(AMODAL_ROOT, "output", "gradio_runs"))
SCRIPT_DIR = os.path.join(REPO_ROOT, "scripts")


def _build_args(
    input_dir: str,
    filenames_txt: str,
    json_path: str,
    output_dir: str,
    max_iter: int,
    sd_cpu_offload: bool,
    require_cache: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_dir=input_dir,
        img_filenames_txt=filenames_txt,
        json_label_path=json_path,
        output_dir=output_dir,
        gdino_config=os.environ.get(
            "GDINO_CONFIG",
            "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        gdino_ckpt=os.environ.get(
            "GDINO_CKPT", "Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
        ),
        sam_ckpt=os.environ.get(
            "SAM_CKPT", "Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
        ),
        instaorder_ckpt=os.environ.get(
            "INSTAORDER_CKPT",
            "InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar",
        ),
        ram_ckpt=os.environ.get(
            "RAM_CKPT", "./recognize-anything/ram_plus_swin_large_14m.pth"
        ),
        sd_model_id=os.environ.get(
            "SD_MODEL_ID", "sd2-community/stable-diffusion-2-inpainting"
        ),
        sd_cpu_offload=sd_cpu_offload,
        lisa_server_url=LISA_SERVER_URL,
        lisa_mask_dir=LISA_MASK_DIR,
        skip_lisa_server=True,
        require_lisa_cache=require_cache,
        save_interm=True,
        max_iter_id=int(max_iter),
        mc_clean_bkgd_img="images/gray_wallpaper.jpeg",
        text_query="main object",
        inpaint_prompt="main object",
        line_num=0,
        round_number=1,
    )


def cache_lisa_mask(image, text_query, status=None):
    if image is None:
        return None, "", "Please upload an image."
    if not text_query or not str(text_query).strip():
        return None, "", "Please enter a text query."

    os.makedirs(LISA_MASK_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="lisa_gradio_")
    img_path = os.path.join(tmp_dir, "input.png")
    if isinstance(image, str):
        shutil.copy(image, img_path)
        pil = Image.open(img_path).convert("RGB")
    else:
        pil = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        pil.save(img_path)

    stem = f"gradio_{int(time.time())}"
    pkl_path = os.path.join(LISA_MASK_DIR, f"{stem}.pkl")

    try:
        from gradio_client import Client

        client = Client(LISA_SERVER_URL)
        result = client.predict(str(text_query).strip(), img_path, api_name="/predict")
        with open(result[1], "r", encoding="utf-8") as jf:
            mask_json = json.load(jf)
        mask = np.array(mask_json["data"])
        with open(pkl_path, "wb") as pf:
            pickle.dump(mask, pf)
        overlay = Image.open(result[0])
        meta = {
            "stem": stem,
            "pkl": pkl_path,
            "query": str(text_query).strip(),
            "img_path": img_path,
            "tmp_dir": tmp_dir,
        }
        meta_path = os.path.join(LISA_MASK_DIR, f"{stem}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        msg = (
            f"Cached mask at {pkl_path}\n"
            f"Stem auto-filled: {stem}\n"
            "Next: Stop LISA server, then Run main pipeline."
        )
        return overlay, stem, msg
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, "", f"LISA call failed (is server up on 7860?): {exc}"


def stop_lisa_server():
    script = os.path.join(SCRIPT_DIR, "stop_lisa.sh")
    try:
        out = subprocess.check_output(["bash", script], stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as exc:
        return exc.output or str(exc)
    except Exception as exc:
        return str(exc)


def run_main_on_upload(image, text_query, lisa_stem, max_iter, sd_cpu_offload, output_dir):
    if image is None:
        return None, "Upload an image first."
    if not text_query or not str(text_query).strip():
        return None, "Enter a text query."

    output_dir = output_dir or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LISA_MASK_DIR, exist_ok=True)

    work = tempfile.mkdtemp(prefix="amodal_gradio_", dir=output_dir)
    rel_name = "input.png"
    img_path = os.path.join(work, rel_name)

    if isinstance(image, str):
        shutil.copy(image, img_path)
    else:
        pil = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image.convert("RGB")
        pil.save(img_path)

    # Resolve LISA mask: explicit stem, or latest matching meta, or require user to cache first
    stem = (lisa_stem or "").strip()
    if not stem:
        return None, "Provide the LISA cache stem from the previous step (e.g. gradio_1719...)."

    src_pkl = os.path.join(LISA_MASK_DIR, f"{stem}.pkl")
    if not os.path.isfile(src_pkl):
        return None, f"Mask pickle not found: {src_pkl}. Cache LISA mask first, then stop LISA."

    # Place pickle where main expects: lisa_mask_dir / <filename_stem>.pkl
    # filename is "input.png" -> stem "input"
    dst_pkl = os.path.join(LISA_MASK_DIR, "input.pkl")
    shutil.copy(src_pkl, dst_pkl)

    filenames_txt = os.path.join(work, "files.txt")
    with open(filenames_txt, "w", encoding="utf-8") as f:
        f.write(rel_name + "\n")

    json_path = os.path.join(work, "ann.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "annotations": [
                    {"filename": rel_name, "labels": [str(text_query).strip()]}
                ]
            },
            f,
        )

    run_out = os.path.join(work, "out")
    os.makedirs(run_out, exist_ok=True)

    # main joins input_dir + filename; keep cwd = REPO_ROOT for sibling imports
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        args = _build_args(
            input_dir=work,
            filenames_txt=filenames_txt,
            json_path=json_path,
            output_dir=run_out,
            max_iter=max_iter,
            sd_cpu_offload=bool(sd_cpu_offload),
            require_cache=True,
        )
        pipeline.run_pipeline(args, ["input.png"], 0, 1)
    except Exception as exc:
        os.chdir(prev)
        return None, f"Pipeline failed: {exc}"
    finally:
        os.chdir(prev)

    result_path = os.path.join(run_out, "input", "amodal_completion.png")
    if os.path.isfile(result_path):
        return result_path, f"Success.\n{result_path}"
    # Fallback: any amodal_completion under out
    for root, _, files in os.walk(run_out):
        if "amodal_completion.png" in files:
            p = os.path.join(root, "amodal_completion.png")
            return p, f"Success.\n{p}"
    return None, f"Finished but amodal_completion.png not found under {run_out}"


def build_ui():
    with gr.Blocks(title="Amodal Appearance Completion") as demo:
        gr.Markdown(
            """
# Open-World Amodal Appearance Completion
**Ports:** LISA server `7860` · this UI `7861`

### Single-GPU workflow
1. In a terminal: `bash scripts/start_lisa.sh`
2. Here: upload image + query → **Cache LISA mask**
3. **Stop LISA server** (frees VRAM)
4. Paste the returned stem → **Run main pipeline**
            """
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Input image")
                text_query = gr.Textbox(
                    label="Text query",
                    placeholder="e.g. What is the cat in this image? Please output segmentation mask.",
                )
                lisa_stem = gr.Textbox(label="LISA cache stem (from step 2)")
                max_iter = gr.Slider(1, 5, value=3, step=1, label="max_iter_id")
                sd_offload = gr.Checkbox(label="SD CPU offload (lower VRAM)", value=False)
                output_dir = gr.Textbox(label="Output dir", value=DEFAULT_OUTPUT)
            with gr.Column():
                lisa_overlay = gr.Image(type="pil", label="LISA overlay")
                result_img = gr.Image(type="filepath", label="Amodal completion")
                log = gr.Textbox(label="Status", lines=8)

        with gr.Row():
            btn_lisa = gr.Button("1) Cache LISA mask", variant="primary")
            btn_stop = gr.Button("2) Stop LISA server")
            btn_main = gr.Button("3) Run main pipeline", variant="primary")

        btn_lisa.click(
            cache_lisa_mask,
            inputs=[image, text_query],
            outputs=[lisa_overlay, lisa_stem, log],
        )
        btn_stop.click(stop_lisa_server, outputs=[log])
        btn_main.click(
            run_main_on_upload,
            inputs=[image, text_query, lisa_stem, max_iter, sd_offload, output_dir],
            outputs=[result_img, log],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=int(_env("GRADIO_PIPELINE_PORT", "7861")),
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

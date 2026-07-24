"""
Acknowledgment:
This code is inspired by "Amodal Completion via Progressive Mixed Context Diffusion"
(https://github.com/k8xu/amodal). 
Our code is built upon and extensively utilizes their work.
"""

import os
import sys
import argparse
import json
import torch
import cv2
import numpy as np
import shutil
import gc
import time
import pickle
import re
import math
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from skimage.morphology import erosion, disk
from PIL import Image
from gradio_client import Client
from diffusers import StableDiffusionInpaintPipeline
import clip

# InstaOrder
# https://github.com/POSTECH-CVLab/InstaOrder
sys.path.append('InstaOrder')
import models
import inference as infer

# Grounding DINO
# https://github.com/IDEA-Research/GroundingDINO
sys.path.append("Grounded-Segment-Anything")
sys.path.append("Grounded-Segment-Anything/GroundingDINO")
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
# https://github.com/facebookresearch/segment-anything
from segment_anything import build_sam, SamPredictor

# RAM
# https://github.com/xinyu1205/recognize-anything
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

# ---------------------------------------------------------------------------
# Paths / HF cache (repo-root layout + optional .env)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: str) -> None:
    """Minimal KEY=VALUE loader; does not override existing os.environ keys."""
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

AMODAL_ROOT = os.environ.get("AMODAL_ROOT", _REPO_ROOT)
LISA_SERVER_URL = os.environ.get("LISA_SERVER_URL", "http://127.0.0.1:7860/")
LISA_OUTPUT_PATH = os.environ.get(
    "LISA_OUTPUT_PATH", os.path.join(AMODAL_ROOT, "cache", "lisa_masks")
)
# Large-disk HF cache (set HF_HOME in .env; default matches lab backup volume)
HF_HOME = os.environ.get("HF_HOME", "/backup/data/art-gen")
SD_MODEL_ID_DEFAULT = os.environ.get(
    "SD_MODEL_ID", "sd2-community/stable-diffusion-2-inpainting"
)

os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.makedirs(LISA_OUTPUT_PATH, exist_ok=True)

# Reused SAM predictor (avoids reloading ViT-H every call)
_SAM_PREDICTOR = None
_SAM_CKPT_LOADED = None


class QueryObject:
    def __init__(self, img_path, img, img_pil, mask_id, query_mask, output_img_dir):
        self.img_path = img_path
        self.img = img
        self.img_pil = img_pil
        self.mask_id = mask_id
        self.query_mask = query_mask
        self.output_img_dir = output_img_dir
        self.run_iter = True
        self.iter_id = 0
        self.first_sd_occ_mask = None 
        self.skipflag = False


def parse_args():
    parser = argparse.ArgumentParser(description='Run the pipeline')
    parser.add_argument('--input_dir',         type=str,  help="Folder path to images")
    parser.add_argument('--img_filenames_txt', type=str,  default="./img_filenames.txt", help='Text file with image filenames in input_dir that you want to use')
    parser.add_argument('--json_label_path',   type=str,  default="./img_annotation.json")
    parser.add_argument('--output_dir',        type=str,  default="./output")
    
    # Grounding DINO, SAM, InstaOrder, RAM
    parser.add_argument('--gdino_config',      type=str,  default=os.environ.get("GDINO_CONFIG", "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument('--gdino_ckpt',        type=str,  default=os.environ.get("GDINO_CKPT", "Grounded-Segment-Anything/groundingdino_swint_ogc.pth"))
    parser.add_argument('--sam_ckpt',          type=str,  default=os.environ.get("SAM_CKPT", "Grounded-Segment-Anything/sam_vit_h_4b8939.pth"))
    parser.add_argument('--instaorder_ckpt',   type=str,  default=os.environ.get("INSTAORDER_CKPT", "InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar"))
    parser.add_argument('--ram_ckpt',          type=str,  default=os.environ.get("RAM_CKPT", "./recognize-anything/ram_plus_swin_large_14m.pth"))

    parser.add_argument('--sd_model_id',       type=str,  default=SD_MODEL_ID_DEFAULT, help='HF repo id for SD2 inpainting')
    parser.add_argument('--sd_cpu_offload',    action='store_true', help='Enable Diffusers model CPU offload (lower VRAM, slower)')
    
    parser.add_argument('--lisa_server_url',   type=str,  default=LISA_SERVER_URL)
    parser.add_argument('--lisa_mask_dir',     type=str,  default=LISA_OUTPUT_PATH, help='Directory of cached LISA mask pickles')
    parser.add_argument('--skip_lisa_server',  action='store_true', help='Only use cached LISA masks; never call Gradio server')
    parser.add_argument('--require_lisa_cache', action='store_true', help='Fail if mask pickle missing (recommended after LISA batch stage)')

    parser.add_argument('--save_interm',       type=bool, default=True, help='Whether to save intermediate images')
    parser.add_argument('--max_iter_id',       type=int,  default=3,    help='Maximum number of pipeline iterations')
    parser.add_argument('--mc_clean_bkgd_img', type=str,  default="images/gray_wallpaper.jpeg", help='Path to clean background image')
    parser.add_argument('--text_query',        type=str,  default="main object", help='Text query input from user.')
    parser.add_argument('--inpaint_prompt',    type=str,  default="main object", help='Inpainting prompt')
    parser.add_argument('--line_num',          type=int,  default=0, help='Starting line index in filename list')
    parser.add_argument('--round_number',      type=int,  default=5, help='Images per process launch')
    return parser.parse_args()


def read_txt(file_path: str):
    with open(file_path, 'r') as f:
        files = f.read().splitlines()
    return files


def find_mask_sides(mask, val=1):
    """
    Determine the bounding box of a given value
    """
    mask[mask > 0] = 1
    x_arr, y_arr = np.where(mask == val)
    x_min, x_max = min(x_arr), max(x_arr)
    y_min, y_max = min(y_arr), max(y_arr)
    return x_min, x_max, y_min, y_max


def load_models(gdino_config, gdino_ckpt, instaorder_ckpt=None, sd_model_id=None, sd_cpu_offload=False, device="cuda"):
    """
    Load Grounding DINO, Stable Diffusion inpainter, and InstaOrder.
    LaMa is intentionally omitted (unused by the original pipeline at runtime).
    """
    loaded_models = []

    # Grounding DINO
    gdino_args = SLConfig.fromfile(gdino_config)
    gdino_args.device = device
    gdino_model = build_model(gdino_args)
    gdino_ckpt = torch.load(gdino_ckpt, map_location="cpu", weights_only=False)
    gdino_model.load_state_dict(clean_state_dict(gdino_ckpt["model"]), strict=False)
    gdino_model.eval()
    loaded_models.append(gdino_model)

    sd_id = sd_model_id or SD_MODEL_ID_DEFAULT
    sd_inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
        sd_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    sd_inpaint_model.enable_attention_slicing()
    if sd_cpu_offload:
        sd_inpaint_model.enable_model_cpu_offload()
    else:
        sd_inpaint_model = sd_inpaint_model.to("cuda")
    loaded_models.append(sd_inpaint_model)

    # InstaOrder, default parameters from InstaOrderNet_od
    # https://github.com/POSTECH-CVLab/InstaOrder/blob/dd88eceea300f60722b763f66335ca0099007aea/experiments/InstaOrder/InstaOrderNet_od/config.yaml
    if instaorder_ckpt is not None:
        instaorder_model_params = {
            'algo': 'InstaOrderNet_od',
            'total_iter': 60000,
            'lr_steps': [32000, 48000],
            'lr_mults': [0.1, 0.1],
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'optim': 'SGD',
            'warmup_lr': [],
            'warmup_steps': [],
            'use_rgb': True,
            'backbone_arch': 'resnet50_cls',
            'backbone_param': {'in_channels': 5, 'num_classes': [2, 3]},
            'overlap_weight': 0.1, 'distinct_weight': 0.9
        }
        instaorder_model = models.__dict__['InstaOrderNet_od'](instaorder_model_params)
        instaorder_model.load_state(instaorder_ckpt)
        instaorder_model.switch_to('eval')
        loaded_models.append(instaorder_model)

    return loaded_models


def get_sam_predictor(sam_ckpt, device="cuda"):
    """Reuse a single SAM predictor across calls to avoid repeated ViT-H loads."""
    global _SAM_PREDICTOR, _SAM_CKPT_LOADED
    if _SAM_PREDICTOR is None or _SAM_CKPT_LOADED != sam_ckpt:
        _SAM_PREDICTOR = SamPredictor(build_sam(checkpoint=sam_ckpt).to(device))
        _SAM_CKPT_LOADED = sam_ckpt
    return _SAM_PREDICTOR


def transform_image(img_pil, save_interm=False, output_img_dir=None):
    """
    Transform PIL image to tensor
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(img_pil, None)
    return img_tensor


def run_gdino(gdino_model, img, caption, box_thresh=0.35, text_thresh=0.35, with_logits=True, device="cuda"):
    gdino_model = gdino_model.to(device)
    img = img.to(device)

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = gdino_model(img[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # Filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_thresh
    logits_filt = logits_filt[filt_mask]  # (num_filt, 256)
    boxes_filt = boxes_filt[filt_mask]  # (num_filt, 4)
    logits_filt.shape[0]

    # Get predicted objects
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_thresh, gdino_model.tokenizer(caption), gdino_model.tokenizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def run_sam(img_pil, sam_ckpt, boxes_filt, pred_phrases=None, device="cuda"):
    img = np.array(img_pil)
    predictor = get_sam_predictor(sam_ckpt, device=device)
    predictor.set_image(img)

    # Predict SAM masks
    size = img_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt = boxes_filt.cpu()
    try:
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, img.shape[:2]).to(device)
        masks, iou_predictions, _ = predictor.predict_torch(  # masks: [1, 1, 512, 512]
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
    except: return None, None  # If there is an error, then skip to the next image
    masks = masks.cpu().numpy().squeeze(1)  # Convert from torch tensor to numpy array
    return img, masks


def segment(gdino_model, run_sam, sam_ckpt, img_pil, img_tensor, classes, device="cuda",target_class=None):
    """
    Run Grounding DINO on image
    """
    classes_query = target_class

    classes = ". ".join(classes)
    boxes_filt, pred_phrases = run_gdino(gdino_model, img_tensor, classes, device=device)

    target_flag = False
    boxes_filt_query, pred_phrases_query = run_gdino(gdino_model, img_tensor, classes_query, device=device)
    for pred_phrase in pred_phrases:
        class_name, pred_score, _ = re.split("\(|\)", pred_phrase)
        if class_name == classes_query:
            target_flag = True
            break
    if not target_flag:
        boxes_filt = torch.cat((boxes_filt, boxes_filt_query), dim=0)
        pred_phrases = np.concatenate((pred_phrases, pred_phrases_query), axis=0)

    img, masks = run_sam(img_pil, sam_ckpt, boxes_filt, pred_phrases=pred_phrases)
    # Separate predicted phrases into class names and prediction scores
    class_names = []
    pred_scores = []
    for pred_phrase in pred_phrases:
        class_name, pred_score, _ = re.split("\(|\)", pred_phrase)
        class_names.append(class_name)
        pred_scores.append(float(pred_score))

    return img, masks, class_names, pred_scores


def check_valid_query(
    img, mask_id, query_mask, class_names, pred_scores, classes,
    query_pred_score_thresh=0.3, query_mask_size_thresh=0.01, save_interm=False, output_img_dir=None,
):
    """
    Check whether the query object is suitable for amodal completion
    """
    query_mask = query_mask.astype(np.uint8)
    query_class = class_names[mask_id]
    pred_score = pred_scores[mask_id]
    if pred_score < query_pred_score_thresh or query_mask.sum() < query_mask_size_thresh * img.shape[0] * img.shape[1]: 
        print(f"Query object {query_class} {mask_id} does not meet the minimum score threshold or size threshold, skipping")
        return

    if query_class not in classes: return

    return query_mask, query_class


def expand_bbox(bboxes):
    """
    Expand bbox in InstaOrder network
    """
    new_bboxes = []
    for bbox in bboxes:
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * 3.0), bbox[2] * 1.1, bbox[3] * 1.1])
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        new_bboxes.append(new_bbox)
    return np.array(new_bboxes)


def find_expand_bboxes_instaorder(masks):
    bboxes = np.zeros((len(masks), 4))
    for i, mask in enumerate(masks):
        mask[mask > 0] = 1
        x_min_obj, x_max_obj, y_min_obj, y_max_obj = find_mask_sides(mask)
        w = y_max_obj - y_min_obj; h = x_max_obj - x_min_obj
        bboxes[i, 0] = y_min_obj
        bboxes[i, 1] = x_min_obj
        bboxes[i, 2] = w
        bboxes[i, 3] = h
    bboxes = expand_bbox(bboxes)
    return bboxes


def analyze_masks(instaorder_model, img, masks, mask_id):
    """
    Analyze occlusion order using InstaOrder

    Occlusion order and depth order matrices
        occ_order[i, j] = 1 if j is over i (due to transpose)
        depth_order[i, j] = 1 if i is closer than j
        depth_order[i, j] = 2 if i is equal to j
    """
    modal = np.zeros((len(masks), masks[0].shape[0], masks[0].shape[1]))
    for i, mask in enumerate(masks):
        modal[i] = mask
    bboxes = find_expand_bboxes_instaorder(masks)

    pcolor_occ_order, pcolor_depth_order = infer.infer_order_sup_occ_depth(
        instaorder_model, img, modal, bboxes, pairs="all", method="InstaOrderNet_od",
        patch_or_image="resize", input_size=384, disp_select_method="")
    pcolor_occ_order = pcolor_occ_order.transpose()

    all_occluder_masks = []
    occ_indices = pcolor_occ_order[mask_id]
    occ_mask_indices = np.where(occ_indices == 1)[0]
    for occ_mask_index in occ_mask_indices:
        if occ_mask_index == mask_id: continue  # Skip occluder mask if it's the same as the query mask
        all_occluder_masks.append(masks[occ_mask_index])

    del modal, bboxes, pcolor_occ_order 
    gc.collect() 

    return all_occluder_masks


def aggregate_occluders(query_mask, all_occluder_masks, query_class, mask_id, iter_id, save_interm=False, output_img_dir=None):
    """
    Aggregate all occluders into a single mask
    """
    agg_occluder_mask = np.zeros((query_mask.shape))
    for occluder_mask in all_occluder_masks:
        agg_occluder_mask += occluder_mask
    agg_occluder_mask[agg_occluder_mask > 0] = 1

    # Ensure new occluders do not contain query mask
    query_occ_overlap = query_mask + agg_occluder_mask
    agg_occluder_mask[query_occ_overlap > 1] = 0  # Prevent occluder from containing query mask
    
    return agg_occluder_mask


def create_canvas(input_arr, size_multiplier, canvas_val):
    """
    Preprocess input image or mask by placing on blank canvas
    """
    input_height, input_width = input_arr.shape[0], input_arr.shape[1]
    canvas_shape = list(input_arr.shape)
    canvas_shape[0] = int(input_height * size_multiplier)
    canvas_shape[1] = int(input_width * size_multiplier)
    canvas_shape = tuple(canvas_shape)

    assert canvas_val >= 0
    if canvas_val > 0:
        canvas = np.ones(canvas_shape) * canvas_val
    else:
        canvas = np.zeros(canvas_shape)

    # Place input on canvas
    input_height_start = (canvas_shape[0] // 2) - (input_height // 2)
    input_height_end = input_height_start + input_height
    input_width_start = (canvas_shape[1] // 2) - (input_width // 2)
    input_width_end = input_width_start + input_width
    canvas[input_height_start : input_height_end, input_width_start : input_width_end] = input_arr
    return canvas


def check_touch_boundary(mask, gap_pixels=10):
    """
    Check whether the mask touches image boundary
    """
    H, W = mask.shape[0], mask.shape[1]
    x_min_obj, x_max_obj, y_min_obj, y_max_obj = find_mask_sides(mask)

    sides_touched = set()
    if (x_max_obj >= H - gap_pixels):
        sides_touched.add("bottom")
    if (x_min_obj <= gap_pixels):
        sides_touched.add("top")
    if (y_max_obj >= W - gap_pixels):
        sides_touched.add("right")
    if (y_min_obj <= gap_pixels):
        sides_touched.add("left")

    return sides_touched


def find_crop_region(query_mask, query_mask_canvas, pad_pixels=150, crop_buffer=60):
    """
    Apply conditional padding and determine cropping region
    """
    query_mask_canvas_height, query_mask_canvas_width = query_mask_canvas.shape
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = find_mask_sides(query_mask_canvas)
    crop_x_min = max(0, crop_x_min - crop_buffer)
    crop_x_max = min(query_mask_canvas_height, crop_x_max + crop_buffer)
    crop_y_min = max(0, crop_y_min - crop_buffer)
    crop_y_max = min(query_mask_canvas_width, crop_y_max + crop_buffer)

    # Conditional padding
    sides_touched = check_touch_boundary(query_mask)
    #print('sides_touched:',sides_touched)
    if "top" in sides_touched:
        crop_x_min -= pad_pixels
    if "bottom" in sides_touched:
        crop_x_max += pad_pixels
    if "left" in sides_touched:
        crop_y_min -= pad_pixels
    if "right" in sides_touched:
        crop_y_max += pad_pixels
    
    # Compute target cropped region size
    crop_height = crop_x_max - crop_x_min
    crop_width = crop_y_max - crop_y_min
    crop_target_size = max(crop_height, crop_width)

    # Update cropped region to square with padding only on sides touched
    if crop_width < crop_target_size:
        crop_target_size_diff = (crop_target_size - crop_width)
        if "left" in sides_touched:
            crop_y_min -= crop_target_size_diff
        elif "right" in sides_touched:
            crop_y_max += crop_target_size_diff
        else:
            crop_y_min -= math.floor(crop_target_size_diff / 2)
            crop_y_max += math.ceil(crop_target_size_diff / 2)

    if crop_height < crop_target_size:
        crop_target_size_diff = (crop_target_size - crop_height)
        if "top" in sides_touched:
            crop_x_min -= crop_target_size_diff
        elif "bottom" in sides_touched:
            crop_x_max += crop_target_size_diff
        else:
            crop_x_min -= math.floor(crop_target_size_diff / 2)
            crop_x_max += math.ceil(crop_target_size_diff / 2)

    # Ensure that the crop boundaries do not exceed the canvas dimensions
    crop_x_min = max(0, crop_x_min)
    crop_x_max = min(query_mask_canvas_height, crop_x_max)
    crop_y_min = max(0, crop_y_min)
    crop_y_max = min(query_mask_canvas_width, crop_y_max)

    return crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_target_size


def crop(inputs, query_mask, query_mask_canvas):
    """
    Apply crop region to all inputs
    """
    crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_target_size = find_crop_region(query_mask, query_mask_canvas)
    return [input_arr[crop_x_min : crop_x_max, crop_y_min : crop_y_max].astype(np.uint8) for input_arr in inputs], crop_x_min, crop_x_max, crop_y_min, crop_y_max


def compute_iou(mask1, mask2):
    overlap = mask1 + mask2
    overlap[overlap < 2] = 0; overlap[overlap == 2] = 1
    intersection = overlap.sum()
    union = mask1 + mask2
    union[union == 0] = 0; union[union > 0] = 1
    union = union.sum()
    return intersection / union


def filter_out_amodal_segmentation(crop_query_mask, amodal_masks):
    """
    Filter out the amodal segmentation from instance segmentation mask candidates
    """
    # When no seg masks detected, treat the original modal mask as the amodal mask
    amodal_segmentation = crop_query_mask
    max_iou = 0
    amodal_i = 0
    for i, amodal_mask in enumerate(amodal_masks):
        iou = compute_iou(crop_query_mask, amodal_mask.astype(np.uint8))
        if iou > max_iou:
            amodal_segmentation = amodal_mask
            max_iou = iou
            amodal_i = i
    return amodal_i, amodal_segmentation


def check_occlusion(
    amodal_completion,
    crop_query_mask,
    query_class,
    mask_id,
    gdino_model,
    sam_ckpt,
    instaorder_model,
    classes,
    query_obj,
    crop_x_min,
    crop_x_max,
    crop_y_min,
    crop_y_max,
    save_interm=False,
):
    """
    Check whether the query object remains occluded
    """
    amodal_completion_tensor = transform_image(amodal_completion)
    amodal_completion, amodal_segmentations, seg_class_name, _ = segment(gdino_model, run_sam, sam_ckpt, amodal_completion, amodal_completion_tensor, classes, target_class= query_class)

    #Save the pickle flie to the path query_obj.output_img_dir
    if query_obj.iter_id == 0:
        pickle_path = os.path.join(query_obj.output_img_dir, "amodal_segmentationsiter"+".pickle")
        with open(pickle_path, 'wb') as handle:
            pickle.dump((amodal_completion, amodal_segmentations,seg_class_name), handle)

    # If no masks are detected, then proceed to the next object
    if amodal_segmentations is None:
        query_obj.run_iter = False
        query_obj.amodal_segmentation = None
        return query_obj  

    amodal_i, amodal_segmentation = filter_out_amodal_segmentation(crop_query_mask, amodal_segmentations)
    query_obj.amodal_segmentation = amodal_segmentation.astype(np.uint8)

    new_occluder_masks = analyze_masks(instaorder_model, amodal_completion, amodal_segmentations, amodal_i)
    new_occluder_mask = aggregate_occluders(amodal_segmentation, new_occluder_masks, query_class, amodal_i, query_obj.iter_id, save_interm=save_interm, output_img_dir=query_obj.output_img_dir)
    
    # Update canvas with new amodal completion
    query_obj.img_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = amodal_completion
    query_obj.query_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = amodal_segmentation
    query_obj.occ_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = new_occluder_mask
    query_obj.outpaint_mask_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = np.zeros(amodal_segmentation.shape)

    amodal_sides_touched = check_touch_boundary(amodal_segmentation)

    # If sides are touched,
    # expand the `new_occluder_mask` and `query_obj.occ_mask_canvas` on the corresponding sides
    if "top" in amodal_sides_touched:
        query_obj.occ_mask_canvas[:crop_x_min + 5, :] = 1  # Update the top side of the canvas mask
    if "bottom" in amodal_sides_touched:
        query_obj.occ_mask_canvas[crop_x_max - 5:, :] = 1  # Update the bottom side of the canvas mask
    if "left" in amodal_sides_touched:
        query_obj.occ_mask_canvas[:, :crop_y_min + 5] = 1  # Update the left side of the canvas mask
    if "right" in amodal_sides_touched:
        query_obj.occ_mask_canvas[:, crop_y_max - 5:] = 1  # Update the right side of the canvas mask

    if new_occluder_mask.sum() > 0 or len(amodal_sides_touched) > 0:
        query_obj.run_iter = True
    else:
        query_obj.run_iter = False

    return query_obj


def compute_offset(expanded_query_mask, init_outpainting_mask, amodal_segmentation):
    query_x_arr, query_y_arr = np.where(expanded_query_mask == 1)
    query_x_coord = min(query_x_arr); query_y_coord = min(query_y_arr)

    orig_x_arr, orig_y_arr = np.where(init_outpainting_mask == 0)
    orig_x_coord = min(orig_x_arr); orig_y_coord = min(orig_y_arr)

    amodal_seg_x_arr, amodal_seg_y_arr = np.where(amodal_segmentation == 1)
    amodal_x_coord = min(amodal_seg_x_arr); amodal_y_coord = min(amodal_seg_y_arr)

    x_offset = int(query_x_coord - amodal_x_coord - orig_x_coord)
    y_offset = int(query_y_coord - amodal_y_coord - orig_y_coord)
    return x_offset, y_offset


def run_iteration(
    query_obj,
    output_dir,
    masks,
    classes,
    class_names,
    pred_scores,
    gdino_model,
    sam_ckpt,
    instaorder_model,
    sd_inpaint_model,
    mc_clean_bkgd_img,
    sd_target_size=512,
    save_interm=True,  # Whether to save intermediate images
):
    """
    Returns whether to run an additional iteration
    """
    # Check whether query object is valid for amodal completion
    img = query_obj.img
    mask_id = query_obj.mask_id
    query_mask = query_obj.query_mask
    query_obj.skipflag = False

    check_query = check_valid_query(img, mask_id, query_mask, class_names, pred_scores, classes, save_interm=save_interm, output_img_dir=query_obj.output_img_dir)
    if check_query is None:
        print(f"Invalid query object for {query_obj.iter_id}")
        query_obj.run_iter = False
        query_obj.amodal_segmentation = None
        return query_obj
    query_mask, query_class = check_query
    query_obj.query_class = query_class
    print(f"Running iteration {query_obj.iter_id} for {query_class} {mask_id}") # Print query object
    
    # Analyze masks to determine occluders
    occluder_masks = analyze_masks(instaorder_model, img, masks, mask_id)
    occ_mask = aggregate_occluders(query_mask, occluder_masks, query_class, mask_id, query_obj.iter_id, save_interm=save_interm, output_img_dir=query_obj.output_img_dir)
    output_img_dir = query_obj.output_img_dir
    # Check occlusion by image boundary
    sides_touched = check_touch_boundary(query_mask)
    query_obj.run_iter = True if (occ_mask.sum() > 0 or len(sides_touched) > 0) else False
    if not query_obj.run_iter:
        query_obj.amodal_segmentation = None
        return query_obj

    if query_obj.iter_id == 0:
        # Preprocess the img, query mask, and occluder mask
        img_canvas = create_canvas(img, size_multiplier=6, canvas_val=255)
        query_mask_canvas = create_canvas(query_mask, size_multiplier=6, canvas_val=0)
        occ_mask_canvas = create_canvas(occ_mask, size_multiplier=6, canvas_val=0)
        outpaint_mask_canvas = create_canvas(np.zeros((query_mask.shape)), size_multiplier=6, canvas_val=1)

        # Save init image and mask canvas
        query_obj.img_canvas = img_canvas
        query_obj.query_mask_canvas = query_mask_canvas
        query_obj.occ_mask_canvas = occ_mask_canvas
        query_obj.outpaint_mask_canvas = outpaint_mask_canvas
        query_obj.init_img_canvas = img_canvas.copy()
        query_obj.init_query_mask_canvas = query_mask_canvas.copy()
        query_obj.init_occ_mask_canvas = occ_mask_canvas.copy()
        query_obj.init_outpaint_mask_canvas = outpaint_mask_canvas.copy()
        
    # Crop image and mask canvas
    crop_inputs = [query_obj.img_canvas, query_obj.query_mask_canvas, query_obj.occ_mask_canvas, query_obj.outpaint_mask_canvas]
    crop_outputs, crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop(crop_inputs, query_mask, query_obj.init_query_mask_canvas)
    crop_img, crop_query_mask, crop_occ_mask, crop_outpaint_mask = crop_outputs

    # Create input bundle for Stable Diffusion
    sd_img = crop_img
    sd_modal_mask = crop_query_mask
    sd_occ_mask = crop_occ_mask + crop_outpaint_mask
    sd_occ_mask[sd_occ_mask > 0] = 1
    sd_prompt = query_class
    sd_prompt = 'a ' + sd_prompt

    # Create a new erosion version of the sd_modal_mask
    kernel = np.ones((5, 5), np.uint8)
    sd_modal_mask_erode = cv2.erode(sd_modal_mask, kernel, iterations=1).astype(np.uint8)
    # Create a new sd_img with the sd_modal_mask cut out as RGB, background replaced with white
    sd_img_cut = sd_img.copy()
    sd_img_cut[sd_modal_mask_erode == 0] = [255, 255, 255]
    sd_img_cut_intermediate_np = np.array(sd_img_cut)
    sd_img_cut = sd_img.copy()
    sd_img_cut = cv2.cvtColor(sd_img_cut, cv2.COLOR_BGR2RGBA)
    # Set the alpha channel to 0 for the sd_modal_mask
    sd_img_cut[sd_modal_mask_erode == 0, 3] = 0
    sd_img_cut_intermediate_np = np.array(sd_img_cut)

    if query_obj.iter_id == 0 and output_img_dir is not None:
        # Save the last iteration's sd_img_cut
        output_path = f"{output_img_dir}/sd_img_cut.png"
        cv2.imwrite(output_path, sd_img_cut_intermediate_np)
        query_obj.first_sd_occ_mask = sd_occ_mask.copy()

    if query_obj.iter_id > 0 and query_obj.first_sd_occ_mask is not None:
        # Update sd_occ_mask to be the intersection of the current sd_occ_mask and the first iteration's sd_occ_mask
        sd_occ_mask = np.logical_and(sd_occ_mask, query_obj.first_sd_occ_mask).astype(np.uint8)

    # Check if sd_occ_mask is empty (all zeros)
    if not np.any(sd_occ_mask):
        query_obj.run_iter = False
        print("aggregated occluder mask is empty, stop iteration")
        return query_obj

    input_height, input_width = sd_img.shape[0], sd_img.shape[1]
   
   # Dilate the occlusion mask more for the first iteration
    if query_obj.iter_id == 0:
        kernel = np.ones((5, 5), np.uint8)
        sd_occ_mask = cv2.dilate(sd_occ_mask, kernel, iterations=3).astype(np.uint8)
    else:
        kernel = np.ones((3, 3), np.uint8)
        sd_occ_mask = cv2.dilate(sd_occ_mask, kernel, iterations=1).astype(np.uint8)

    sd_img_intermediate_np = np.array(sd_img)

    if sd_img_intermediate_np.shape[-1] == 3:
        sd_img_intermediate_np = cv2.cvtColor(sd_img_intermediate_np, cv2.COLOR_RGB2BGR)
    
    if query_obj.iter_id == 0:
        # Swap background
        clean_bkgd_img = np.array(Image.open(mc_clean_bkgd_img).convert("RGB"))
        clean_bkgd_mask = 1 - sd_modal_mask
        clean_bkgd_x_arr, clean_bkgd_y_arr = np.where(clean_bkgd_mask == 1)
        
        # Skip images if object too large
        if (clean_bkgd_x_arr.max() >= clean_bkgd_img.shape[0]) or (clean_bkgd_y_arr.max() >= clean_bkgd_img.shape[1]):
            print(f"Skipping image due to index out-of-bounds: {query_obj.img_path}")
            query_obj.run_iter = False
            query_obj.skipflag = True
            return query_obj
        
        sd_img_syn = sd_img.copy()        
        sd_img_syn[clean_bkgd_x_arr, clean_bkgd_y_arr] = clean_bkgd_img[clean_bkgd_x_arr, clean_bkgd_y_arr]

    else:
        sd_img_syn = sd_img.copy()

    sd_img = Image.fromarray(sd_img_syn).convert("RGB").resize((sd_target_size, sd_target_size), resample=Image.LANCZOS)
    sd_occ_mask = Image.fromarray(255 * sd_occ_mask).convert("L").resize((sd_target_size, sd_target_size), resample=Image.NEAREST)


    with torch.no_grad():
        amodal_completion = sd_inpaint_model(
            image=sd_img,
            mask_image=sd_occ_mask,
            prompt=sd_prompt,
        ).images[0]

    query_obj.iter_id += 1
    amodal_completion = amodal_completion.resize((input_width, input_height))
    
    query_obj.amodal_completion = amodal_completion

    # Occlusion check
    query_obj = check_occlusion(
        amodal_completion,
        sd_modal_mask,
        query_class,
        mask_id,
        gdino_model,
        sam_ckpt,
        instaorder_model,
        classes,
        query_obj,
        crop_x_min,
        crop_x_max,
        crop_y_min,
        crop_y_max,
        save_interm,
    )

    return query_obj


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def lisa_mask_pickle_path(lisa_mask_dir, img_filename):
    """Stable path for a cached LISA mask pickle given an image filename."""
    stem = img_filename.split(".")[0]
    # Preserve nested relative dirs under the cache root
    return os.path.join(lisa_mask_dir, f"{stem}.pkl")


def load_or_query_lisa_mask(args, img_path, img_filename, text_query, output_img_dir):
    """
    Prefer cached LISA mask pickle (after batch stage). Optionally fall back to live Gradio server.
    Returns numpy mask array.
    """
    pickle_path = lisa_mask_pickle_path(args.lisa_mask_dir, img_filename)
    os.makedirs(os.path.dirname(pickle_path) or ".", exist_ok=True)

    if os.path.isfile(pickle_path):
        print(f"[LISA] Using cached mask: {pickle_path}")
        with open(pickle_path, "rb") as f:
            mask = pickle.load(f)
        mask = np.array(mask)
        # Optional overlay copy for debugging
        visible_seg_path = os.path.join(output_img_dir, "visible_seg.png")
        if not os.path.exists(visible_seg_path):
            # Write a simple grayscale visualization
            Image.fromarray((mask.astype(np.uint8) * 255)).save(visible_seg_path)
        return mask

    if args.skip_lisa_server or args.require_lisa_cache:
        raise FileNotFoundError(
            f"LISA mask cache missing for {img_filename}: {pickle_path}. "
            "Run scripts/run_lisa_batch.py then scripts/stop_lisa.sh before main."
        )

    print(f"[LISA] Cache miss — calling server {args.lisa_server_url}")
    client = Client(args.lisa_server_url)
    resultclient = client.predict(text_query, img_path, api_name="/predict")
    with open(resultclient[1], "r") as file_from_json:
        pre_maskdata = json.load(file_from_json)
    mask = np.array(pre_maskdata["data"])

    visible_seg_path = os.path.join(output_img_dir, "visible_seg.png")
    shutil.copyfile(resultclient[0], visible_seg_path)

    with open(pickle_path, "wb") as f:
        pickle.dump(mask, f)
    print(f"[LISA] Saved mask cache: {pickle_path}")
    return mask


def run_pipeline(args, read_img_filenames, range_len, round_number):
    gdino_model, sd_inpaint_model, instaorder_model = load_models(
        args.gdino_config,
        args.gdino_ckpt,
        args.instaorder_ckpt,
        sd_model_id=args.sd_model_id,
        sd_cpu_offload=args.sd_cpu_offload,
    )

    img_filenames = read_img_filenames

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.lisa_mask_dir, exist_ok=True)

    with open(args.json_label_path, 'r') as f:
        jsonlabel_data = json.load(f)

    # Create a dictionary mapping image base names to labels
    imgname_prompt_map = {}
    for json_ann in jsonlabel_data['annotations']:
        # Extract the user input (image base name) for prompt calculation
        img_basename = json_ann['filename'].split("/")[-1].split(".")[0]
        imgname_prompt_map[img_basename] = json_ann['labels'][0] 

    # Initialize a variable to store total processing time excluding the specified part
    total_processing_time = 0.0
    image_count = 0  # Counter for images processed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_classes = get_transform(image_size=384)
    
    # Load ram_plus model
    remodel = ram_plus(pretrained=args.ram_ckpt, image_size=384, vit='swin_l')
    # the threshold for unseen categories is often lower
    remodel.class_threshold = torch.ones(remodel.num_class) * 0.6
    remodel.eval()
    remodel = remodel.to(device)

    # Load the ClIP model
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)

    # Iterate over each image filename
    for img_filename in tqdm(img_filenames, desc="Iterate images"):
        print('img_filename:', img_filename)
        start_time = time.time()  # Start timer before processing an image
        # Extract the base name from the image file path
        img_basename = img_filename.split("/")[-1].split(".")[0]

        #Search image label
        if img_basename not in imgname_prompt_map:
            print(f'{img_basename}: No image label found in JSON data (filename: {img_filename})')
            continue

        # Print the found label for the image
        print(f'Image: {img_filename}, Label: {imgname_prompt_map[img_basename]}')

        img_path = os.path.join(args.input_dir, img_filename)
        img_pil = Image.open(img_path).convert('RGB')
        output_img_dir = os.path.join(args.output_dir, img_basename)

        resimage = transform_classes(img_pil.resize((384,384))).unsqueeze(0).to(device)
        res = inference(resimage, remodel)

        classes_list = res[0].split('|')
        classes_list = [x.strip() for x in classes_list]
        classes_list.append('background') # add background class
        classes = classes_list

        if os.path.exists(output_img_dir):
            shutil.rmtree(output_img_dir)

        # Create output directories
        os.makedirs(output_img_dir, exist_ok=True)
        if args.save_interm:
            #subdirs = ["amodal_completions_raw", "amodal_segmentations"]
            subdirs = ["amodal_completions_raw"]
            for subdir in subdirs:
                os.makedirs(os.path.join(output_img_dir, subdir), exist_ok=True)

        # Perform instance segmentation
        img_tensor = transform_image(img_pil, save_interm=args.save_interm, output_img_dir=output_img_dir)
        img_offsets_dict = {}

        args.text_query = imgname_prompt_map[img_basename]

        # LISA: prefer disk cache (batch stage); optional live Gradio fallback
        try:
            lisa_mask = load_or_query_lisa_mask(
                args, img_path, img_filename, args.text_query, output_img_dir
            )
        except FileNotFoundError as exc:
            print(exc)
            continue
        except Exception as exc:
            print(f"[LISA] Failed for {img_filename}: {exc}")
            continue

        if lisa_mask is None or np.asarray(lisa_mask).size == 0:
            print("Visible mask empty, skipping image")
            continue

        pre_maskdata_arr = np.array(lisa_mask)

        #Use Clip to decide inpaint prompt 
        # Assuming mask contains 0s and 1s, where 1s indicate the mask regions
        pre_maskdata_npmask = np.array(pre_maskdata_arr, dtype=bool)
        target_image_np = np.array(img_pil)

        # Apply the mask to each color channel
        pre_maskdata_npmask_image = np.zeros_like(target_image_np)
        for i in range(3):  # Assuming image has three channels
            pre_maskdata_npmask_image[:, :, i] = target_image_np[:, :, i] * pre_maskdata_npmask

        args.inpaint_prompt = clip_similarity_w_query(Image.fromarray(np.uint8(pre_maskdata_npmask_image)) , classes, args.text_query, clip_model, clip_preprocess, device)
        print('inpaint_prompt:', args.inpaint_prompt)

        # Save the image with the mask applied
        pre_maskdata_npmask_image = Image.fromarray(np.uint8(pre_maskdata_npmask_image))
        pre_maskdata_npmask_image.save(os.path.join(output_img_dir, 'masked_image.png'))

        img, masks, class_names, pred_scores = segment(gdino_model, run_sam, args.sam_ckpt, img_pil, img_tensor, classes, target_class=args.inpaint_prompt)
    
        if masks is None:
            print("No object masks detected, skipping image") 
            continue  # If no masks are detected, then proceed to the next image

        query_obj = QueryObject(img_path, img, img_pil, len(masks), pre_maskdata_arr, output_img_dir)
        masks = np.append(masks, [pre_maskdata_arr], axis=0)
        select_class_name = args.inpaint_prompt
        class_names.append(select_class_name)   # class name of the query object
        pred_scores.append(float(1))

        separated_masks_bools = handlemask(img, masks, class_names)

        if separated_masks_bools==[]:
            pass
        else:
            # Append separated_masks to masks
            masks = np.concatenate((masks, np.array(separated_masks_bools)), axis=0)

            for background_num in range(len(separated_masks_bools)):
                class_names.append('background_'+str(background_num))
                pred_scores.append(float(1))

        classes.append(select_class_name)
        classes = remove_duplicates(classes)

        mask_id = len(masks) -1 
        query_mask = masks[-1]

        while query_obj.run_iter:
            query_obj = run_iteration(
                query_obj,
                args.output_dir,
                masks,
                classes,
                class_names,
                pred_scores,
                gdino_model,
                args.sam_ckpt,
                instaorder_model,
                sd_inpaint_model,
                args.mc_clean_bkgd_img,
                save_interm=args.save_interm,
            )
            
            if query_obj.iter_id > args.max_iter_id: break

        if query_obj.skipflag: continue

        # Post-processing
        if query_obj.amodal_segmentation is not None and query_obj.iter_id > 0:
            query_class = query_obj.query_class
            x_offset, y_offset = compute_offset(query_obj.query_mask_canvas, query_obj.init_outpaint_mask_canvas, query_obj.amodal_segmentation)
            img_offsets_dict[f'{query_class}_{query_obj.mask_id}'] = [x_offset, y_offset]
            img_offset_save_path = os.path.join(query_obj.output_img_dir, "offsets.json")
            with open(img_offset_save_path, 'w') as fp:
                json.dump(img_offsets_dict, fp, sort_keys=True, indent=4)

            amodal_completion_to_save = query_obj.amodal_completion
            amodal_completion_to_save.save(os.path.join(query_obj.output_img_dir, "amodal_completions_raw", f'_{mask_id}.jpg'), quality=90)
            amodal_segmentation_to_save = Image.fromarray(query_obj.amodal_segmentation * 255).convert("RGB")
            amodal_segmentation_to_save.save(os.path.join(query_obj.output_img_dir, f'amodal_segmentation.png'))

            with open(os.path.join(query_obj.output_img_dir,  'prompt.txt') , 'w') as file:
                file.write(select_class_name)
            
            #Apply amodal mask to the amodal completion image
            amodal_completion_np = np.array(amodal_completion_to_save)
            amodal_segmentation_np = np.array(amodal_segmentation_to_save.convert("L"))
            amodal_mask = amodal_segmentation_np > 0
            # Apply mask to the amodal completion image and add alpha channel
            amodal_completion_masked_np = np.zeros((amodal_completion_np.shape[0], amodal_completion_np.shape[1], 4), dtype=np.uint8)
            amodal_completion_masked_np[..., :3] = amodal_completion_np
            amodal_completion_masked_np[..., 3] = amodal_mask.astype(np.uint8) * 255

            amodal_completion_masked_to_save = Image.fromarray(amodal_completion_masked_np, 'RGBA')
            amodal_completions_masked_dir = os.path.join(query_obj.output_img_dir, "amodal_completions_processed")
            os.makedirs(amodal_completions_masked_dir, exist_ok=True)
            amodal_completion_masked_to_save.save(os.path.join(amodal_completions_masked_dir, f'_{mask_id}.png'), quality=100)
            img_blend(amodal_completions_masked_dir, mask_id)
            
            # Delete intermediate folders
            clean_up_intermediate_results(query_obj.output_img_dir)
        
        # Free GPU and CPU memory after each image
        del img, masks, class_names, pred_scores, query_obj
        torch.cuda.empty_cache()
        gc.collect()  # Manual garbage collection to free up memory
    

def handlemask(img, masks, class_names):
    """
    Identifies and extracts additional background regions
    """
    mask_all_zeros = np.zeros_like(masks[0])

    for mask in masks:
        mask_all_zeros += mask

    new_mask = 1 - mask_all_zeros
    # Define the structuring element for erosion
    selem = disk(2)  # You can adjust the radius of the disk as needed

    # Perform erosion
    eroded_mask = erosion(new_mask, selem)

    # Find contours in the eroded mask
    contours, _ = cv2.findContours(eroded_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the separated masks
    separated_masks = []

    # Iterate through the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
    
        # If the area is greater than 100 pixels
        if area > 100:
            # Create a new mask for the contour with the correct dtype
            mask = np.zeros_like(eroded_mask, dtype=np.uint8)  # Ensure mask is of type uint8
            cv2.drawContours(mask, [contour], -1, 1, -1)
            
            # Add the mask to the list of separated masks
            separated_masks.append(mask)

    # Convert separated_masks to boolean
    separated_masks_bool = [mask.astype(bool) for mask in separated_masks]

    return separated_masks_bool


def img_blend(amodal_completions_masked_dir, mask_id):
    # Parent of amodal_completions_processed is the per-image output dir
    amodal_completions_dir = os.path.dirname(amodal_completions_masked_dir)

    # Blend the amodal completion RGBA image with the original RGBA image
    visible_obj_img_path = os.path.join(amodal_completions_dir, f'sd_img_cut.png')
    inpainted_img_path = os.path.join(amodal_completions_masked_dir, f'_{mask_id}.png')

    # Load the images as RGBA
    src_im = cv2.imread(visible_obj_img_path, cv2.IMREAD_UNCHANGED)

    # Check if inpainted image file exists
    if not os.path.exists(inpainted_img_path):
        print(f"amodal completion image not found, returning original image.")
        return src_im
    
    dst_im = cv2.imread(inpainted_img_path, cv2.IMREAD_UNCHANGED)
 
    blended_img = alpha_blending(shrink_edges_to_transparent(src_im), dst_im)

    blended_img_path = os.path.join(amodal_completions_dir, f'amodal_completion.png')
    cv2.imwrite(blended_img_path, blended_img)

    # Load the images
    visible_obj_img = Image.open(visible_obj_img_path)
    inpainted_img = Image.open(inpainted_img_path)

    # Resize the inpainted image to match the size of the visible object image
    inpainted_img_resized = inpainted_img.resize(visible_obj_img.size, resample=Image.LANCZOS)

    # Blend the images
    blended_img = Image.blend(visible_obj_img, inpainted_img_resized, alpha=0.5)

    return blended_img


def alpha_blending(src, dst, transi_wid=5):
    """
    Performs alpha blending between two images with RGBA channels using smooth transitions.
    If dst is transparent in some transition_region, use the original src color for that transition_region.

    Args:
        src: The source image (RGBA).
        dst: The destination image (RGBA).
        transi_wid: The width of the transition region between src and dst.

    Returns:
        The blended image (RGBA).
    """

    # Ensure both images are RGBA and have the same dimensions
    if src.shape[2] != 4 or dst.shape[2] != 4 or src.shape[:2] != dst.shape[:2]:
        raise ValueError("Both src and dst images must be RGBA format and have the same dimensions.")

    # Create a binary mask from the alpha channel of the source image
    src_mask = (src[:, :, 3] > 0).astype(np.uint8)

    # Create an erosion of the source mask to define the interior region
    kernel = np.ones((transi_wid, transi_wid), np.uint8)
    src_interior = cv2.erode(src_mask, kernel, iterations=1)

    # Calculate the transition region as the difference between the original and the eroded mask
    transition_region = src_mask - src_interior

    # Calculate the distance transform on the transition region
    dist_transform = cv2.distanceTransform((1 - src_mask).astype(np.uint8), cv2.DIST_L2, 5)
    dist_transform = np.clip(dist_transform / transi_wid, 0, 1)  # Normalize distances within transition width

    # Create a weight mask based on the distance transform
    weight_src = np.where(transition_region > 0, dist_transform, 1.0)  # Apply distance transform in transition only
    weight_dst = 1.0 - weight_src

    # Initialize the blended image
    blended_image = dst.copy()

    # Directly use src RGB values in the source visible regions (interior of the source)
    blended_image[src_mask > 0] = src[src_mask > 0]

    # Apply smooth blending using the computed weights
    # Only blend if dst alpha channel in the transition region is not 0
    dst_alpha_transition = dst[transition_region > 0, 3]
    blend_indices = dst_alpha_transition > 0
    blended_image[transition_region > 0, :3][blend_indices] = (
        weight_dst[transition_region > 0, np.newaxis][blend_indices] * dst[transition_region > 0, :3][blend_indices] +
        weight_src[transition_region > 0, np.newaxis][blend_indices] * src[transition_region > 0, :3][blend_indices]
    )
    # Use src color if dst alpha is 0 in the transition region
    no_blend_indices = ~blend_indices
    blended_image[transition_region > 0, :3][no_blend_indices] = src[transition_region > 0, :3][no_blend_indices]

    # Set the alpha channel in the blended image
    blended_image[:, :, 3] = np.maximum(src[:, :, 3], dst[:, :, 3])

    return blended_image


def shrink_edges_to_transparent(image, shrink_amount=10):
    """
    Shrinks the edges of an RGBA image where the alpha channel is 0, making them transparent.

    Args:
        image: The input RGBA image (numpy array).
        shrink_amount: The number of pixels to shrink the edges by.

    Returns:
        The modified RGBA image with shrunk transparent edges.
    """

    if image.shape[2] != 4:
        raise ValueError("Input image must be in RGBA format.")

    alpha_channel = image[:, :, 3]
    mask = (alpha_channel == 0).astype(np.uint8)
    kernel = np.ones((shrink_amount * 2 + 1, shrink_amount * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    modified_image = image.copy()
    modified_image[dilated_mask > 0, 3] = 0  # Set alpha to 0 in the dilated area

    return modified_image


def clip_similarity_w_query(image, text, text_query, model, preprocess, device):
    # Preprocess image and text inputs
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text + [text_query]]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    query_index = len(text)  # Index of text_query in the text inputs
    similarity[0][query_index] *= 1 #adjust the weight of the query text if needed

    # Return the class label with the highest weighted score
    max_index = similarity[0].argmax().item()
    return text[max_index] if max_index < len(text) else text_query


def clean_up_intermediate_results(output_img_dir):
    """
    Deletes specified folders and files within the output image directory
    after processing one image.
    """
    # Define folders and files to delete
    folders_to_delete = ['amodal_completions_raw', 'amodal_completions_processed']
    #files_to_delete = ['masked_image.png','sd_img_cut.png']
    files_to_delete = ['masked_image.png']

    # Delete folders
    for folder in folders_to_delete:
        folder_path = os.path.join(output_img_dir, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    # Delete specific files
    for file_name in files_to_delete:
        file_path = os.path.join(output_img_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    args = parse_args()
    if not args.input_dir:
        raise SystemExit("--input_dir is required")
    read_img_filenames = read_txt(args.img_filenames_txt)
    round_number = args.round_number
    print('current img files:', read_img_filenames[args.line_num:args.line_num+round_number])
    run_pipeline(args, read_img_filenames[args.line_num:args.line_num+round_number], args.line_num, round_number)

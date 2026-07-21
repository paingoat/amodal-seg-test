import cv2
import numpy as np
import torch
import lpips
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from open_clip import create_model_and_transforms, tokenize
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
import torchvision.models as inception
import json
from tqdm import tqdm
import os

# ground truth visible image path
visible_img_path = 'your/visible/gt/path/here'

# amodal completion result image path
pre_img_path = 'your/results/path/here'

# ground truth json with labels file path
grouth_truth_json = 'your.json'

# saved json file path
json_file_name = 'results/yournew.json'


# Load LPIPS model (requires torch and lpips library)
lpips_alex = lpips.LPIPS(net='alex')  # Using AlexNet backbone

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load OpenCLIP model for prompt relevance evaluation
openclip_model, _, openclip_preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
openclip_model.eval()

# Load DINO model for feature extraction
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
dino_model.eval()

# Load VGG model for feature extraction
vgg = models.vgg16(pretrained=True).features.eval()

# Preprocessing transformations for CLIP and DINO
clip_preprocess_pil = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

# Preprocessing transformations for VGG and CLIP
vgg_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to convert RGBA to RGB with white background
def rgba_to_rgb(img):
    if img.shape[2] == 4:  # Check if the image has an alpha channel
        alpha_channel = img[:, :, 3] / 255.0
        img_rgb = img[:, :, :3]
        white_background = np.ones_like(img_rgb) * 255
        img_rgb = (img_rgb * alpha_channel[:, :, None] + white_background * (1 - alpha_channel[:, :, None])).astype(np.uint8)
        return img_rgb
    return img

# Function to compute LPIPS between two images
def compute_lpips(img1, img2):
    # Resize images to the same size if they are different
    if img1.shape != img2.shape:
        common_size = (max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))
        img1 = cv2.resize(img1, common_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, common_size, interpolation=cv2.INTER_AREA)
    # Convert images to torch tensors and normalize to [-1, 1]
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    # Compute LPIPS distance
    lpips_value = lpips_alex(img1_tensor, img2_tensor)
    return lpips_value.item()

# Function to compute CLIP score for prompt relevance
def compute_clip_score(image, prompt):
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.item()
    return similarity

# Function to compute OpenCLIP score for prompt relevance
def compute_openclip_score(image, prompt):
    image_input = openclip_preprocess(Image.fromarray(image)).unsqueeze(0)
    text_input = tokenize([prompt])
    with torch.no_grad():
        image_features = openclip_model.encode_image(image_input)
        text_features = openclip_model.encode_text(text_input)
        similarity = F.cosine_similarity(image_features, text_features).item()
    return similarity

# Function to extract DINO features from an image
def extract_dino_features(img):
    img_tensor = openclip_preprocess(Image.fromarray(img)).unsqueeze(0)
    with torch.no_grad():
        features = dino_model(img_tensor)
    return features

# Function to extract VGG features from an image
def extract_vgg_features(img):
    img_tensor = vgg_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = vgg(img_tensor)
    return features

# Function to compute feature-based similarity using VGG
def compute_feature_similarity(img1, img2):
    features1 = extract_vgg_features(img1)
    features2 = extract_vgg_features(img2)
    # Use cosine similarity to compare features
    similarity = F.cosine_similarity(features1, features2).mean().item()
    return similarity

# Function to compute PSNR between two images
def compute_psnr(img1, img2):
    # Resize images to the same size if they are different
    if img1.shape != img2.shape:
        common_size = (max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))
        img1 = cv2.resize(img1, common_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, common_size, interpolation=cv2.INTER_AREA)
    return psnr(img1, img2, data_range=img2.max() - img2.min())

# Function to compute SSIM between two images
def compute_ssim(img1, img2):
    # Resize images to the same size if they are different
    if img1.shape != img2.shape:
        common_size = (max(img1.shape[1], img2.shape[1]), max(img1.shape[0], img2.shape[0]))
        img1 = cv2.resize(img1, common_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, common_size, interpolation=cv2.INTER_AREA)
    # Set win_size to a smaller value if images are small
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # Ensure win_size is odd and <= min dimension
    if win_size < 3:
        win_size = 3  # Set a minimum value to avoid very small windows
    return ssim(img1, img2, channel_axis=-1, data_range=img2.max() - img2.min(), win_size=win_size)


# Compute CLIP Score for prompt relevance
def evaluate_prompt_relevance(completed_image, prompt):
    pil_image = Image.fromarray(completed_image)
    clip_score = compute_clip_score(pil_image, prompt)
    return clip_score


# Compute DINO Feature Similarity
def compute_dino_similarity(img1, img2):
    features1 = extract_dino_features(img1)
    features2 = extract_dino_features(img2)
    similarity = F.cosine_similarity(features1, features2).mean().item()
    return similarity


# Function to compute FID between two sets of features
def compute_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Function to extract features using Inception model
def extract_inception_features(img):
    img_tensor = inception_preprocess(Image.fromarray(img)).unsqueeze(0)
    with torch.no_grad():
        features = inception_model(img_tensor).view(1, -1).numpy()
    return features


# Function to calculate texture similarity using Local Patch Matching
def compute_texture_similarity(img1, img2, patch_size=16):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    texture_similarities = []
    for y in range(0, min(h1, h2) - patch_size + 1, patch_size):
        for x in range(0, min(w1, w2) - patch_size + 1, patch_size):
            patch1 = img1[y:y + patch_size, x:x + patch_size].reshape(-1, 3)
            patch2 = img2[y:y + patch_size, x:x + patch_size].reshape(-1, 3)
            dist = np.linalg.norm(patch1 - patch2)
            texture_similarities.append(dist)
    return np.mean(texture_similarities)


with open(grouth_truth_json, 'r', encoding='utf-8') as file:
    label_json = json.load(file)
label_map = {}
file_list=[]


result = {}
result['annotations']=[]
result['statistics']={}

total_score = 0
total_lpips_value=[]
total_feature_similarity=[]
total_psnr_value=[]
total_ssim_value=[]
total_fid_value=[]
total_clip_score=[]
total_openclip_score=[]
total_dino_similarity=[]

for sub_data in tqdm(label_json['annotations']) :

    filename = sub_data['filename'].split('/')[-1].split('.')[0]
    
    sub_ann = {}
    sub_ann['filename'] =sub_data['filename']
    sub_ann['labels'] = sub_data['labels']
    

    # Load visible part (ground truth) and completed image results
    visible_part = cv2.imread(visible_img_path+filename+'.png', cv2.IMREAD_UNCHANGED)
    completed_result1 = cv2.imread(pre_img_path+filename+'.png', cv2.IMREAD_UNCHANGED)


    # Convert images to RGB (from BGR, as OpenCV loads in BGR format)
    visible_part = cv2.cvtColor(rgba_to_rgb(visible_part), cv2.COLOR_BGR2RGB)
    completed_result1 = cv2.cvtColor(rgba_to_rgb(completed_result1), cv2.COLOR_BGR2RGB)

    # Create mask for occluded region (1 for occluded, 0 for visible)
    # Automatically create mask from the alpha channel of visible.png
    alpha_channel = cv2.imread(visible_img_path+filename+'.png', cv2.IMREAD_UNCHANGED)[:, :, 3]
    occluded_region = (alpha_channel < 255).astype(np.uint8)

    # Crop the occluded region from the completed results
    x, y, w, h = cv2.boundingRect(occluded_region)  # Get bounding box of occluded region
    visible_cropped = visible_part[y:y+h, x:x+w]
    result1_cropped = completed_result1[y:y+h, x:x+w]

    # Resize cropped images to a common size for consistency
    common_size = (max(visible_cropped.shape[1], result1_cropped.shape[1]),
                max(visible_cropped.shape[0], result1_cropped.shape[0]))

    visible_cropped_resized = cv2.resize(visible_cropped, common_size, interpolation=cv2.INTER_AREA)
    result1_cropped_resized = cv2.resize(result1_cropped, common_size, interpolation=cv2.INTER_AREA)


    prompt =  sub_data['labels'][0] 

    clip_score = evaluate_prompt_relevance(completed_result1, prompt)

    # Compute OpenCLIP Score for prompt relevance
    openclip_score = compute_openclip_score(completed_result1, prompt)


    dino_similarity = compute_dino_similarity(visible_part, completed_result1)

    # Load Inception model for FID calculation
    inception_model = inception.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    # Preprocessing transformations for  Inception
    inception_preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract Inception features for FID calculation
    visible_features = extract_inception_features(visible_part)
    result1_features = extract_inception_features(completed_result1)

    # Stack features to create feature sets with at least two samples to ensure valid covariance matrices
    features_visible = np.vstack([visible_features, visible_features])
    features_result1 = np.vstack([result1_features, result1_features])

    # Compute FID between visible part and completed results
    fid_value = compute_fid(features_visible, features_result1)

    # Compute Texture Similarity between visible part and completed results
    texture_similarity1 = compute_texture_similarity(visible_part, completed_result1)

    # Compute LPIPS for cropped regions
    lpips_value = compute_lpips(visible_cropped, result1_cropped)

    # print(f" ↓ LPIPS Result: 1: {lpips_result1}")
    # Compute Feature Similarity for cropped regions
    feature_similarity = compute_feature_similarity(visible_cropped, result1_cropped)

    # print(f" ↑ Feature Similarity Result: 1: {feature_similarity1}")
    # Compute PSNR for cropped regions
    psnr_value = compute_psnr(visible_cropped, result1_cropped)

    # print(f" ↑ PSNR Result: 1: {psnr_result1}")
    # Compute SSIM for cropped regions
    ssim_value = compute_ssim(visible_cropped, result1_cropped)


    # Modify perfect scores to opposite
    if lpips_value == 0:
        lpips_value = 1
    if feature_similarity == 1:
        feature_similarity = 0
    if psnr_value == float("inf"):
        psnr_value = 0
    if ssim_value == 1:
        ssim_value = 0
    if fid_value == 0:
        fid_value = float("inf")

    sub_ann['LPIPS'] = lpips_value
    sub_ann['Feature Similarity'] = feature_similarity
    sub_ann['PSNR'] = psnr_value
    sub_ann['SSIM'] = ssim_value
    sub_ann['FID'] = fid_value
    sub_ann['CLIP Score'] = clip_score
    sub_ann['OpenCLIP Score'] = openclip_score
    sub_ann['DINO Similarity'] = dino_similarity
    total_lpips_value.append(lpips_value)
    total_feature_similarity.append(feature_similarity)
    total_psnr_value.append(psnr_value)
    total_ssim_value.append(ssim_value)
    total_fid_value.append(fid_value)
    total_clip_score.append(clip_score)
    total_openclip_score.append(openclip_score)
    total_dino_similarity.append(dino_similarity)
    result['annotations'].append(sub_ann)

result['statistics'] = {
    "Average LPIPS": np.mean(total_lpips_value),
    "Average Feature Similarity": np.mean(total_feature_similarity),
    "Average PSNR": np.mean(total_psnr_value),
    "Average SSIM": np.mean(total_ssim_value),
    "Average FID": np.mean(total_fid_value),
    "Average CLIP Score": np.mean(total_openclip_score),
    "Average OpenCLIP Score": np.mean(total_openclip_score),
    "Average DINO Similarity": np.mean(total_dino_similarity),
}

with open(json_file_name, 'w', encoding='utf-8') as file:
    json.dump(result, file, indent=4)
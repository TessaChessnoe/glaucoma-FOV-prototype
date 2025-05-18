import os
import json
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def load_fixations(json_path):
    # Load fixation points from SALICON annotations JSON file
    with open(json_path) as f:
        data = json.load(f)
    # Unzip list of coordinates into 
    fixations = {ann["image_id"]: ann["fixations"]
            for ann in data["annotations"]}
    return fixations
def load_image_mapping(json_path):
    # Load mapping from filenames to image IDs
    with open(json_path) as f:
        data = json.load(f)
    return {img["file_name"]: img["id"] for img in data["images"]}

def calculate_metrics(sal_map, fixations, img_shape):
    # Calculate all metrics for a single image
    h, w = img_shape[:2]
    
    # Create binary fixation map
    fix_map = np.zeros((h, w))
    for x, y in fixations:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < w and 0 <= y < h:
            fix_map[y, x] = 1
    
    # Create density map with Gaussian blur
    density_map = cv2.GaussianBlur(fix_map, (0, 0), sigmaX=19, sigmaY=19)
    density_map /= density_map.sum() + 1e-12 # const to prevent 0 div
    
    # Normalize saliency map
    sal_map = cv2.normalize(sal_map, None, 0, 1, cv2.NORM_MINMAX)
    sal_map /= sal_map.sum() + 1e-12
    # Calculate metrics
    metrics = {
        "auc": calculate_auc(sal_map, fix_map),
        "nss": calculate_nss(sal_map, fixations, (h, w)),
        "sim": calculate_similarity(sal_map, density_map),
        "kl": calculate_kl_divergence(sal_map, density_map)
    }
    return metrics

def calculate_auc(sal_map, fix_map):
    # Calculate AUC using sampled negative locations
    sal_flat = sal_map.ravel()
    fix_flat = fix_map.ravel()
    
    pos_idx = np.where(fix_flat > 0)[0]
    if len(pos_idx) == 0:
        return 0.0
    
    neg_idx = np.where(fix_flat == 0)[0]
    np.random.seed(42)
    neg_sample = np.random.choice(neg_idx, min(len(pos_idx), len(neg_idx)), 
                               replace=False)
    
    labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_sample))])
    scores = np.concatenate([sal_flat[pos_idx], sal_flat[neg_sample]])
    
    return roc_auc_score(labels, scores)

def calculate_nss(sal_map, fixations, img_shape):
    # Calculate Normalized Scanpath Saliency.
    h, w = img_shape
    mean = sal_map.mean()
    std = sal_map.std()
    if std == 0:
        return 0.0
    
    nss_values = []
    for x, y in fixations:
        # Clamp x & y to image bounds
        x = max(0, min(int(x), w-1))
        y = max(0, min(int(y), h-1))
        nss_values.append((sal_map[y, x] - mean) / std) # y,x to match order h,w
    nss = np.mean(nss_values)
    return nss if nss_values else 0.0

def calculate_similarity(sal_map, density_map):
    # Calculate Similarity metric (histogram intersection)
    return np.minimum(sal_map, density_map).sum()

def calculate_kl_divergence(sal_map, density_map, eps=1e-12):
    # Calculate KL Divergence between distributions
    sal = sal_map + eps
    den = density_map + eps
    return np.sum(sal * np.log(sal / den))

# Configuration
DATASET_DIR = "app/salience/dataset/salicon"
FIXATIONS_JSON = "fixations_val2014.json"

# Load data mappings
im_map = load_image_mapping(os.path.join(DATASET_DIR, FIXATIONS_JSON))
fix_map = load_fixations(os.path.join(DATASET_DIR, FIXATIONS_JSON))

# Initialize saliency detector and results
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
metrics = []

all_filenames = [f for f in os.listdir(os.path.join(DATASET_DIR, "val"))]
N_IMAGES = 1000
sampled_filenames = random.sample(all_filenames, N_IMAGES)
# Process images
for filename in tqdm(sampled_filenames, desc='Processing Images'):
    if filename not in im_map:
        continue
    
    # Load image and get fixations
    img_path = os.path.join(DATASET_DIR, "val", filename)
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    image_id = im_map[filename]
    fixations = fix_map.get(image_id, [])
    
    # Compute saliency map
    success, sal_map = saliency.computeSaliency(img)
    if not success:
        continue
    
    # Calculate and store metrics
    img_metrics = calculate_metrics(sal_map, fixations, img.shape)
    metrics.append(img_metrics)

# Aggregate results
results = pd.DataFrame(metrics).mean()
print(f"\nFinal Metrics:")
print(f"AUC:  {results['auc']:.4f}")
print(f"NSS:  {results['nss']:.4f}")
print(f"SIM:  {results['sim']:.4f}")
print(f"KL:   {results['kl']:.4f}")
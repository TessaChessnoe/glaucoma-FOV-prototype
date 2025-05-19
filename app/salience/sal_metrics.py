import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def load_fixations(json_path):
    # Load fixation points from SALICON annotations JSON file
    with open(json_path) as f:
        data = json.load(f)
    return {ann["image_id"]: ann["fixations"] for ann in data["annotations"]}

def load_image_mapping(json_path):
    #Load mapping from filenames to image IDs.
    with open(json_path) as f:
        data = json.load(f)
    return {img["file_name"]: img["id"] for img in data["images"]}

def calculate_metrics(sal_map, fixations, img_shape):
    # Calculate all metrics for a single image
    h, w = img_shape[:2]
    
    # Create binary fixation map
    fix_map = np.zeros((h, w))
    for x,y in fixations:
        x = int(x)
        y = int(y)
        # Clamp x,y to image bounds
        if 0 <= x < w and 0 <= y < h:
            fix_map[y, x] = 1 # y first bc of h, w order
    
    # Create density map with Gaussian blur
    density_map = cv2.GaussianBlur(fix_map, (0, 0), sigmaX=5, sigmaY=5)
    density_map /= density_map.sum() + 1e-12
    
    # Normalize saliency map
    sal_map = cv2.normalize(sal_map, None, 0, 1, cv2.NORM_MINMAX)
    sal_sum = sal_map.sum()
    sal_map = sal_map/(sal_sum + 1e-12) if sal_sum > 0 else sal_map
    
    return {
        "auc": calculate_auc(sal_map, fix_map),
        "nss": calculate_nss(sal_map, fixations, (h, w)),
        "sim": calculate_similarity(sal_map, density_map),
        "kl": calculate_kl_divergence(sal_map, density_map)
    }

def calculate_auc(sal_map, fix_map):
    # Calculate AUC using shuffled negative sampling
    sal_flat = sal_map.ravel()
    fix_flat = fix_map.ravel()
    
    pos_indices = np.where(fix_flat > 0)[0]
    if len(pos_indices) == 0:
        return 0.5  # Random performance
    
    neg_indices = np.random.choice(
        np.where(fix_flat == 0)[0], 
        min(len(pos_indices), 10000),  # Cap negatives for efficiency
        replace=False
    )
    
    y_true = np.concatenate([np.ones_like(pos_indices), np.zeros_like(neg_indices)])
    y_score = np.concatenate([sal_flat[pos_indices], sal_flat[neg_indices]])
    
    return roc_auc_score(y_true, y_score)

def calculate_nss(sal_map, fixations, img_shape):
    # Calculate Normalized Scanpath Saliency
    h, w = img_shape
    if sal_map.std() == 0:
        return 0.0
    
    normalized_map = (sal_map - sal_map.mean()) / sal_map.std()
    nss_values = []
    
    for x,y in fixations:
        x = max(0, min(int(x), w-1))
        y = max(0, min(int(y), h-1))
        nss_values.append(normalized_map[y, x])
    
    return np.mean(nss_values) if nss_values else 0.0

def calculate_similarity(sal_map, density_map):
    # Histogram intersection similarity.
    return np.minimum(sal_map, density_map).sum()

def calculate_kl_divergence(sal_map, density_map, eps=1e-12):
    # KL Divergence between distributions
    return np.sum(density_map * np.log(eps + density_map/(sal_map + eps)))

# Configuration
SPLIT = "val"
DATASET_DIR = "app/salience/dataset/salicon"
FIXATIONS_JSON = f"fixations_{SPLIT}2014.json"

# Load data mappings
im_map = load_image_mapping(os.path.join(DATASET_DIR, FIXATIONS_JSON))
fix_map = load_fixations(os.path.join(DATASET_DIR, FIXATIONS_JSON))

# Initialize saliency detector and results
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
metrics = []

# Process images
image_dir = os.path.join(DATASET_DIR, SPLIT)
input_images = os.listdir(image_dir)

# # Ensure all images in dir are present in image map
# valid_inp_imgs = [f for f in input_images if f in im_map]

for filename in tqdm(input_images):
    # Load image and get fixations
    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    image_id = im_map[filename]
    fixations = fix_map.get(image_id, [])

    # Compute saliency map
    success, sal_map = saliency.computeSaliency(img)
    if not success or sal_map is None:
        continue
    
    # Calculate metrics
    try:
        img_metrics = calculate_metrics(sal_map, fixations, img.shape)
        metrics.append(img_metrics)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Aggregate and print results
if metrics:
    results = pd.DataFrame(metrics).mean()
    print("\nFinal Metrics:")
    print(f"AUC:  {results['auc']:.4f}")
    print(f"NSS:  {results['nss']:.4f}")
    print(f"SIM:  {results['sim']:.4f}")
    print(f"KL:   {results['kl']:.4f}")
else:
    print("No valid metrics calculated - check data paths and processing")
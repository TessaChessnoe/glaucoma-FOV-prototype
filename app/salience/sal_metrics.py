import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score

# Configuration
SPLIT = "val"
DATASET_DIR = "app/salience/dataset/salicon"
FIXATIONS_JSON = f"fixations_{SPLIT}2014.json"
GAUSSIAN_SIGMA = 19 # std for Salicon dataset
NEGATIVE_SAMPLES = 10000

def load_fixations(json_path):
    # Load fixation points from SALICON annotations JSON file
    with open(json_path) as f:
        data = json.load(f)
    fix_dict = {}
    for ann in tqdm(data['annotations'], desc="Loading fixations", unit = 'ann'):
        # Fixations dict uses img_id, fixation coords as key, val pairs
        fix_dict[ann['image_id']] = np.array(ann['fixations'], dtype=np.int32)
    return fix_dict


def load_image_mapping(json_path):
    # Load filenames & ids from salicon image JSON file
    with open(json_path) as f:
        data = json.load(f)
    img_dict = {}
    for img in tqdm(data['images'], desc="Loading filename maps", unit = 'img'):
        img_dict[img["file_name"]] = img['id']
    return img_dict

def calculate_metrics(sal_map, fixations, h, w):
    # Check for valid fixations before loading them into metric funcs
    valid = (fixations[:, 1] >= 0) & (fixations[:, 1] < w) & \
            (fixations[:, 0] >= 0) & (fixations[:, 0] < h)
    fix_coords = fixations[valid]
    
    # Create binary fixation map
    fix_map = np.zeros((h, w), dtype=np.float32)
    # Salicon fixation arrays are in row, col (y,x) order
    if fix_coords.size > 0:
        y = np.clip(fix_coords[:, 0], 0, h-1) # rows 1st
        x = np.clip(fix_coords[:, 1], 0, w-1) # cols 2nd
        np.add.at(fix_map, (y, x), 1)
    
    # Create density map
    density_map = cv2.GaussianBlur(fix_map, (0, 0), sigmaX=GAUSSIAN_SIGMA, sigmaY=GAUSSIAN_SIGMA)
    density_map /= density_map.sum() + 1e-12
    
    # Normalize saliency map
    sal_map = cv2.normalize(sal_map, None, 0, 1, cv2.NORM_MINMAX)
    sal_map /= sal_map.sum() + 1e-12
    
    return {
        "auc": calculate_auc(sal_map.ravel(), fix_map.ravel()),
        "nss": calculate_nss(sal_map, y, x),
        "sim": np.minimum(sal_map, density_map).sum(),
        "kl": calculate_kl_divergence(density_map, sal_map)
    }

def calculate_auc(sal_flat, fix_flat):
    pos_indices = np.flatnonzero(fix_flat)
    # If no pos indices, assume performance=random chance
    if len(pos_indices) == 0:
        return 0.5
    
    # Take negative sample BEFORE filtering using fixations
    neg_indices = np.random.choice(len(sal_flat), 
                   min(len(pos_indices), NEGATIVE_SAMPLES), 
                   replace=False)
    # Filter out indices that have fixations
    neg_indices = neg_indices[fix_flat[neg_indices] == 0]
    # Find positions of all true positives and negatives
    y_true =  np.concatenate([np.ones_like(pos_indices), np.zeros_like(neg_indices)])
    # Find saliency scores in computed map for true_pos and true_neg
    y_score = np.concatenate([sal_flat[pos_indices], sal_flat[neg_indices]])
    # Determine correlation between saliency score in computed map and labels in actual fixations
    return roc_auc_score(y_true, y_score)

def calculate_nss(sal_map, y, x):
    if sal_map.std() == 0 or x.size <= 0:
        return 0.0
    normalized = (sal_map - sal_map.mean()) / sal_map.std()
    return normalized[y, x].mean()

def calculate_similarity(sal_map, density_map):
    # Histogram intersection similarity.
    return np.minimum(sal_map, density_map).sum()

def calculate_kl_divergence(true_dist, model_dist, eps=1e-12):
    p = true_dist
    q = model_dist
    # Determines how well salience prob. dist mimics ground truth dist.
    return np.sum(p * np.log(eps + p/(q + eps)))

def main():
    # Pre-compute valid image paths & pre-load fixations
    print("Loading dataset metadata...")
    im_map = load_image_mapping(os.path.join(DATASET_DIR, FIXATIONS_JSON))
    fix_map = load_fixations(os.path.join(DATASET_DIR, FIXATIONS_JSON))
    image_dir = os.path.join(DATASET_DIR, SPLIT)
    files = os.listdir(image_dir)
    # Pre-filter valid files and create processing list
    valid_files = []
    for f in tqdm(files, desc="Filtering valid files", unit="img"):
        if f in im_map:
            img_id    = im_map[f]
            fixations = fix_map.get(img_id, np.empty((0,2),np.int32))
            valid_files.append((f, img_id, fixations))
    N_IMAGES = len(valid_files) // 10
    # Take random sample of images for faster metric calcs
    selected_files = random.sample(valid_files, N_IMAGES)

    # Initialize saliency detector and metrics array
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    metrics = []

    for filename, img_id, fixations in tqdm(selected_files, desc="Processing Images"):
        # Load image and get fixations
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Compute saliency map
        success, sal_map = saliency.computeSaliency(img)
        if not success or sal_map is None:
            continue

        # Verify dimensions
        h, w = img.shape[:2]
        if sal_map.shape != (h, w):
            print(f"Skipping {filename}: Saliency map {sal_map.shape} ≠ image {h}x{w}")
            continue
        
        # Verify fixations
        if fixations.size > 0:
            # Salicon fixations are (y,x)
            invalid_y = (fixations[:,0] > h) | (fixations[:,0] < 0)
            invalid_x = (fixations[:,1] > w) | (fixations[:,1] < 0)
            if invalid_y.any() or invalid_x.any():
                print(f"Invalid fixations in {filename}:")
                print(f"Y range: {fixations[:,0].min()}-{fixations[:,0].max()} (image height: {h})")
                print(f"X range: {fixations[:,1].min()}-{fixations[:,1].max()} (image width: {w})")
                continue
        
        # Calculate metrics
        try:
            h,w = img.shape[:2]
            img_metrics = calculate_metrics(sal_map, fixations, h, w)
            metrics.append(img_metrics)
        except Exception as e:
            print(f"Skipping {filename}: {str(e)}\n")
            continue

    # Aggregate and print results
    if metrics:
        results = pd.DataFrame(metrics)
        print(f"\nFinal Metrics ({len(results)} images):")
        print(f"AUC:  {results.auc.mean():.4f} ± {results.auc.std():.4f}")
        print(f"NSS:  {results.nss.mean():.4f} ± {results.nss.std():.4f}")
        print(f"SIM:  {results.sim.mean():.4f} ± {results.sim.std():.4f}")
        print(f"KL:   {results.kl.mean():.4f} ± {results.kl.std():.4f}")
    else:
        print("No valid metrics calculated - check data paths and processing")

if __name__ == '__main__':
    main()
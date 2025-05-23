import numpy as np
import cv2
from tqdm import tqdm
import os

def gather_dataset(img_dir):
    #Loads all images and corresponding binary masks from OSIE
    print(f"Loading OSIE dataset from {img_dir!r}")
    dataset = []
    img_fns = sorted(os.listdir(img_dir))
    for fn in tqdm(img_fns, desc="Loading Images", unit="file"):
        img_path = os.path.join(img_dir, fn)
        mask_path = os.path.join(img_dir, os.path.splitext(fn)[0] + '.png')
        if not os.path.exists(mask_path):
            continue
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        dataset.append((fn, img, mask))
    print(f" → Loaded {len(dataset)} image/mask pairs.\n")
    return dataset

def normalize_map(sal_map):
    sal = sal_map.astype(np.float32)
    sal -= sal.min()
    sal /= (sal.max() + 1e-12)
    return sal

def calc_seg_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Calculate segmentation metrics between binary prediction and ground truth.
    Returns dict with 'precision', 'recall', 'iou', 'dice'.
    """
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    # avoid division by zero
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    iou  = tp / (tp + fp + fn + 1e-12)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-12)
    return {'precision': prec, 'recall': rec, 'iou': iou, 'dice': dice}
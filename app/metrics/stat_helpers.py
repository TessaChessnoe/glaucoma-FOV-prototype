import numpy as np
import cv2
from tqdm import tqdm
import os
from pycocotools.coco import COCO

def gather_dataset(coco_annotation_file: str, img_dir: str):
    """
    Load validation images and binary masks from COCO.
    coco_annotation_file: path to instances_val2017.json
    img_dir: path to val2017/ folder containing .jpgs
    """
    print(f"Loading COCO annotations from {coco_annotation_file!r}")
    coco = COCO(coco_annotation_file)

    img_ids = coco.getImgIds()  # all validation images
    print(f"→ Found {len(img_ids)} images in COCO val set.")

    dataset = []
    for img_id in tqdm(img_ids, desc="Building COCO dataset", unit="img"):
        meta = coco.loadImgs(img_id)[0]
        fn = meta['file_name']
        path = os.path.join(img_dir, fn)
        # read image
        img = cv2.imread(path)
        if img is None:
            # skip missing files
            continue
        h, w = meta['height'], meta['width']

        # load all instance annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # build a single binary mask: union of all object masks
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            # annToMask returns a HxW uint8 array (1 inside segment)
            m = coco.annToMask(ann)
            mask = np.logical_or(mask, m)

        dataset.append((fn, img, mask))

    print(f"→ Loaded {len(dataset)} image/mask pairs.\n")
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
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    # avoid division by zero
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    iou  = tp / (tp + fp + fn + 1e-12)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return {'precision': prec, 'recall': rec, 'iou': iou, 'dice': dice, 'accuracy': acc}
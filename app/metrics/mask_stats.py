import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from app.models import BMS, IttiKoch
import pysaliency
from concurrent.futures import ThreadPoolExecutor, as_completed
import pysaliency as pys

from app.metrics.stat_helpers import (
    gather_dataset,
    normalize_map,
    calc_seg_stats)

HERE = os.path.dirname(__file__)

# Download built-in models into your model_loc folder
model_root = os.path.join(os.path.dirname(__file__), "pysal_models")

DETECTORS = {
    "AIM": pysaliency.AIM(location=model_root),
    "SUN": pysaliency.SUN(location=model_root),
    "Finegrain": cv2.saliency.StaticSaliencyFineGrained_create(),
    "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual_create(),
    "BMS": BMS,
    "IKN": IttiKoch,
}
EXPENSIVE = ["AIM", "SUN"]

def compute_stats(detector, img, gt_mask):
    """
    Compute segmentation stats for one image using the given detector.
    """
    # compute saliency map
    if hasattr(detector, 'compute_saliency'):
        success, sal_map = detector.compute_saliency(img)
        if not success:
            return None
    else:
        sal_map = detector.computeSaliency(img)
    sal = normalize_map(sal_map)
    pred_mask = sal >= 0.5
    return calc_seg_stats(pred_mask, gt_mask)

def evaluate(sample_size):
    print(">>> Gathering dataset…")
    # point these to your project layout
    coco_json = "data/COCO/annotations/instances_val2017.json"
    img_dir   = os.path.join("data/COCO/val2017")
    dataset = gather_dataset(coco_json, img_dir)

    print(f">>> Dataset size: {len(dataset)} samples\n")

    results = {}
    i = 0 # Tracks which model we're on
    for name, detector in DETECTORS.items():
        i += 1
        stats_list = []
        print(f"=== Instantiating & running model {i}/{len(DETECTORS)}: {name!r} ===")
        print(f"  raw detector object: {detector!r}")
        # Parallelize per-detector over images
        # Run expensive models with less samples
        if name in EXPENSIVE:
            max_workers = 3 # Excessive copies of SUN & AIM overallocate mem
            random.sample(dataset, sample_size)
        else:
            max_workers = max(1, os.cpu_count() - 2)
            sample_data = dataset
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_stats, detector, img, gt_mask): fn
                for fn, img, gt_mask in sample_data
            }
            # progress bar for saliency computation on this model
            for future in tqdm(as_completed(futures),
                                total=len(futures),
                                desc=f"[{i}/{len(DETECTORS)}] {name}",
                                unit="img"):
                stats = future.result()
                if stats is not None:
                    stats_list.append(stats)

        if stats_list:
            summary = {m: np.mean([s[m] for s in stats_list])
                       for m in stats_list[0].keys()}
            results[name] = summary

    # Print aggregated results
    for name, stats in results.items():
        print(f"{name}:")
        for metric, val in stats.items():
            print(f"  {metric}: {val:.4f}")

def main():
    # Set output dir for result
    here = os.path.dirname(__file__)
    output_dir = os.path.join(here, "results")
    os.makedirs(output_dir, exist_ok=True)

    expensive_sample = 10

    # Calculate aggregate metrics for each detector, write result to csv
    evaluate(expensive_sample)

if __name__ == '__main__':
    main()
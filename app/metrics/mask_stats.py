import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from app.models import BMS, IttiKoch
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
    "AIM": pys.AIM(location=model_root),
    "SUN": pys.SUN(location=model_root),
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
    # Compute saliency map
    # Use computeSaliency for OpenCV & custom
    if hasattr(detector, "computeSaliency"):
        success, sal_map = detector.computeSaliency(img)
    # Use saliency_map for pysal models
    else:
        sal_map = detector.saliency_map(img)
        success = sal_map is not None
    if not success:
        return None
    sal = normalize_map(sal_map)
    pred_mask = sal >= 0.5
    return calc_seg_stats(pred_mask, gt_mask)

def print_results(results: dict):
    for name, stats in results.items():
        print(f"{name}:")
    for metric, val in stats.items():
        print(f"  {metric}: {val:.4f}")

def results_to_csv(results: dict, output_dir: str, fname: str = "model_stats.csv"):
    # Flattens nested results dict into df & writes CSV.
    rows = []
    for name, stats in results.items():
        row = {'model': name}
        row.update(stats)
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname)
    df.to_csv(out_path, index=False)
    print(f"Wrote CSV to {out_path}")


def evaluate(n=2000, n_expensive=100, output_dir = None, filename = None, csv_out = False,):
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
            sample_data = random.sample(dataset, n_expensive)
        else:
            max_workers = max(1, os.cpu_count() - 2)
            sample_data = random.sample(dataset, n)
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

        # Aggregate stats across validation set
        if stats_list:
            summary = {m: np.mean([s[m] for s in stats_list])
                       for m in stats_list[0].keys()}
            results[name] = summary

    # Print aggregated results
    print_results(results)
    # Output to csv if enabled
    if (csv_out):
        results_to_csv(results, output_dir, filename)

def main():
    # Set output dir for result
    here = os.path.dirname(__file__)
    output_dir = os.path.join(here, "results")
    output_file = "model_stats.csv"
    os.makedirs(output_dir, exist_ok=True)

    slow_model_n = 1
    fast_model_n = 5

    # Calculate aggregate metrics for each detector, write result to csv
    evaluate(slow_model_n, fast_model_n,
             output_dir, output_file, csv_out = True)

if __name__ == '__main__':
    main()
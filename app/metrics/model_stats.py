import os
import random
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from app.models import BMS, BMSFast, IttiKoch
import pysaliency

from app.metrics.stat_helpers import (
    load_image_mapping,
    load_fixations,
    calculate_stats
)

# Download built-in models into your model_loc folder
model_root = os.path.join(os.path.dirname(__file__), "pysal_models")

DETECTORS = {
    "BMS": BMSFast,
    "IKN": IttiKoch,
    "AIM": pysaliency.AIM(location=model_root),
    "SUN": pysaliency.SUN(location=model_root),
    "Finegrain": cv2.saliency.StaticSaliencyFineGrained_create(),
    "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual_create(),
    # "GBVS-IKN": pysaliency.GBVSIttiKoch(location=model_root),
    # "COVSAL": pysaliency.CovSal(location=model_root),
    # "Judd": pysaliency.Judd(location=model_root),
    # "RARE2012": pysaliency.RARE2012(location=model_root),
    # "CAS": pysaliency.ContextAwareSaliency(location=model_root),
    # "GBVS": pysaliency.GBVS(location=model_root),
}
    
def gather_dataset(input_dir, fixation_json):
    """
    Loads mapping and fixations once, then returns
    a list of (filename, fix_array) for all valid images.
    """
    # Load mappings & fixations ONCE for entire eval
    im_map = load_image_mapping(fixation_json)
    fix_map = load_fixations     (fixation_json)

    # Return filenames and fixations for valid images
    files = sorted(os.listdir(input_dir))
    valid = []
    for fn in files:
        # If filename is not in map, DO NOT attempt id lookup
        if fn not in im_map:
            continue
        img_id = im_map[fn]
        fix_arr = fix_map.get(img_id, np.empty((0,2),np.int32))
        valid.append((fn, fix_arr))
    return valid

def process_one_image(detector, img_path, fixations):
    """
    Load, compute saliency, check dims, then calculate metrics.
    Returns a dict of metrics or None (if failed/skip).
    """
    #opencv_API = ["BMS", "Finegrain", "SpectralRes"]
    # Read input image
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Compute saliency map
    # Use computeSaliency for OpenCV & custom
    if hasattr(detector, "computeSaliency"):
        success, sal_map = detector.computeSaliency(img)
    # Use saliency_map for pysal
    else:
        sal_map = detector.saliency_map(img)
        success = sal_map is not None
    if not success:
        return None
    
    # ensure 2D float32 map
    sal_map = np.array(sal_map, dtype=np.float32)
    if sal_map.ndim == 3 and sal_map.shape[2] == 1:
        sal_map = sal_map[:, :, 0]

    # Verify that salience map has same shape as input
    h, w = img.shape[:2]
    if sal_map.shape != (h, w):
        print(f" Skipped {os.path.basename(img_path)}: shape mismatch")
        return None

    # Clipping invalid fixations happens in calc_stats
    yx = fixations

    # Skip images when metrics cannot be calculated
    try:
        return calculate_stats(sal_map, yx, h, w)
    except Exception as e:
        print(f"  -> skip {os.path.basename(img_path)}: {e}")
        return None

def aggregate_metrics(metrics_list):
    """
    Given a list of {metric: value} dicts for one model,
    return a single dict of average values.
    """
    agg = {}
    if not metrics_list:
        return agg
    keys = metrics_list[0].keys()
    for k in keys:
        agg[k] = float(np.mean([m[k] for m in metrics_list]))
    return agg

def evaluate_all(input_dir, output_dir, fixation_json, sample_size=None):
    # 1) Get input images and ground truth fixations for eval
    dataset = gather_dataset(input_dir, fixation_json)
    # Should add cond to bypass sample if using entire set
    if sample_size:
        dataset = random.sample(dataset, sample_size)

    # Init vars used in processing loop
    stats = []
    i = 0 # used for progress bar text

    # 2) Compute saliency maps & compare against ground truth
    stat_row = {}
    for name, detector in DETECTORS.items():
        i += 1
        all_runs_stats = []
        for fn, fix in tqdm(dataset, desc=f"Running model {i}/{len(DETECTORS)} {name} "):
            img_path = os.path.join(input_dir, fn)
            m = process_one_image(detector, img_path, fix)
            if m:
                all_runs_stats.append(m)
        # stat_row is a dict with stat names as keys
        stat_row["model"] = name
        stat_row["n_images"] = len(all_runs_stats)
        stat_row = aggregate_metrics(all_runs_stats)
        stats.append(stat_row)

    # 3) Write stats to csv
    if stats:
        metrics_df = pd.DataFrame(stats)
        metrics_df.to_csv(os.path.join(output_dir,"model_stats.csv"), index=False)
        print(f"Done! Outputted metrics for {len(stats)} models to model_metrics.csv")
    else:
        print("Error: No metrics added to output list.")

def main():
    # Set output dir for result
    here = os.path.dirname(__file__)
    output_dir = os.path.join(here, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Set other args for eval func
    input_dir = "data/Salicon/val"
    fixation_json = "data/Salicon/fixations_val2014.json"
    sample_size = 5000

    # Calculate aggregate metrics for each detector, write result to csv
    evaluate_all(input_dir, output_dir, fixation_json, sample_size)

if __name__ == '__main__':
    main()
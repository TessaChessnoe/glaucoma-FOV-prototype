import os
import random
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from app.models import BMS, BMSFast, IttiKoch
import pysaliency
from concurrent.futures import ThreadPoolExecutor

from app.metrics.stat_helpers import (
    load_image_mapping,
    load_fixations,
    calculate_stats
)

# Download built-in models into your model_loc folder
model_root = os.path.join(os.path.dirname(__file__), "pysal_models")

DETECTORS = {
    # "AIM": pysaliency.AIM(location=model_root),
    # "SUN": pysaliency.SUN(location=model_root),
    # "Finegrain": cv2.saliency.StaticSaliencyFineGrained_create(),
    # "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual_create(),
    "BMS": BMS,
    # "IKN": IttiKoch,
    # "GBVS-IKN": pysaliency.GBVSIttiKoch(location=model_root),
    # "COVSAL": pysaliency.CovSal(location=model_root),
    # "Judd": pysaliency.Judd(location=model_root),
    # "RARE2012": pysaliency.RARE2012(location=model_root),
    # "CAS": pysaliency.ContextAwareSaliency(location=model_root),
    # "GBVS": pysaliency.GBVS(location=model_root),
}
EXPENSIVE = ["AIM", "SUN"]

def _process(args):
    detector, img_path, fix = args
    return process_one_image(detector, img_path, fix)
    
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

    # Init vars used in processing loop
    stats = []
    i = 0 # used for progress bar text

    # 2) Compute saliency maps for each detector process images in parallel
    for name, detector in DETECTORS.items():
        # Display model progress bar message
        i += 1
        desc = f"Running model {i}/{len(DETECTORS)} {name}"

        # Use subset of data & smaller thread count for expensive models
        if name in EXPENSIVE:
            use_set = random.sample(dataset, k=min(sample_size, len(dataset)))
            # Prevent crashes from memory overallocation
            max_workers = 3
        else:
            use_set = dataset
            # Leave 2 cores free: prevents CPU saturation
            max_workers = max(1, os.cpu_count() - 2)  
            
        # Build process args over chosen set
        args = [(detector, os.path.join(input_dir, fn), fix)
                for fn, fix in use_set]
        
        detector_stats = []

        # parallel map over args
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for result in tqdm(exe.map(_process, args),
                               total=len(args),
                               desc=desc):
                 if result:
                     detector_stats.append(result)
        # aggregate and record
        stat_row = {
            "model": name,
            "n_images": len(detector_stats)} # record whether this was sampled or full
        stat_row.update(aggregate_metrics(detector_stats))
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
    sample_size = 1000

    # Calculate aggregate metrics for each detector, write result to csv
    evaluate_all(input_dir, output_dir, fixation_json, sample_size)

if __name__ == '__main__':
    main()
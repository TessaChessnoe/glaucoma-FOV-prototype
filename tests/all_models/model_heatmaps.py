import pysaliency
import cv2
import os
from app.models import BMS, IttiKoch
import numpy as np

# Download built-in models into your model_loc folder
model_root = "app\metrics\pysal_models"

DETECTORS = {
    "IKN": IttiKoch,
    "AIM": pysaliency.AIM(location=model_root),
    "SUN": pysaliency.SUN(location=model_root),
    "BMS": BMS,
    "Finegrain": cv2.saliency.StaticSaliencyFineGrained_create(),
    "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual_create(),
}

here = os.path.dirname(__file__)
input_dir = os.path.join(here, "..", "data")
output_dir = os.path.join(here, "results")

def compute_sal_map(detector, filename):
    img = cv2.imread(filename)
    if hasattr(detector, "computeSaliency"):
            success, sal_map = detector.computeSaliency(img)
    # Use saliency_map for pysal
    else:
        sal_map = detector.saliency_map(img)
        success = sal_map is not None
    if not success:
        print(f"{detector} failed to compute saliency map for {filename}. ")
    return sal_map

def build_heatmap(sal_map):
    sal_map_u8 = (sal_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_map_u8, cv2.COLORMAP_JET)
    return heatmap

for filename in input_dir:
    for name, detector in DETECTORS.items():
        sal_map = compute_sal_map(detector, filename)
        heatmap = build_heatmap(sal_map)
        cv2.imwrite(os.path.join(output_dir, f"{detector[name]}_{filename}"))
        



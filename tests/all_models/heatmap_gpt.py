import os
import cv2
import numpy as np
import pysaliency
from app.models import BMS, IttiKoch

# Where pysaliency will extract/download its .m models
model_root = os.path.join("app", "metrics", "pysal_models")

DETECTORS = {
    "IKN": IttiKoch,
    "AIM": pysaliency.AIM(location=model_root),
    "SUN": pysaliency.SUN(location=model_root),
    "BMS": BMS,
    "Finegrain": cv2.saliency.StaticSaliencyFineGrained_create(),
    "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual_create(),
}

# Input directory
here = os.path.dirname(__file__)
input_dir = os.path.abspath(os.path.join(here, "..", "data"))

# Output directory for heatmaps
output_dir = os.path.join(here, "results")
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def compute_sal_map(detector, img):
    if hasattr(detector, "computeSaliency"):
        success, sal_map = detector.computeSaliency(img)
    # Use saliency_map for pysal
    else:
        sal_map = detector.saliency_map(img)
        success = sal_map is not None

    if not success:
        return False, None

    # ensure float32 2D
    sal = np.array(sal_map, dtype=np.float32)
    if sal.ndim == 3 and sal.shape[2] == 1:
        sal = sal[:, :, 0]
    return True, sal


def build_heatmap(sal_map):
    # Ensure sal map is normalized
    sal_map = cv2.normalize(sal_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
    sal_u8 = (sal_map * 255).astype(np.uint8)
    return cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)

# Gather all image filenames
image_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
images = [
    fn for fn in os.listdir(input_dir)
    if fn.lower().endswith(image_ext)
]

if not images:
    raise RuntimeError(f"No images found in {input_dir!r}")

for img_name in images:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path!r}, skipping.")
        continue

    for name, detector in DETECTORS.items():
        success, sal_map = compute_sal_map(detector, img)
        if not success or sal_map is None:
            print(f"  {name:12s} failed on {img_name}")
            continue

        heatmap = build_heatmap(sal_map)

        out_name = f"{name}_{os.path.splitext(img_name)[0]}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, heatmap)
        print(f"Sucessfully outputted {out_name}")

print("Done generating heatmaps for all detectors and images.")
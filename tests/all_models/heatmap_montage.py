import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import pysaliency
from app.models import BMS, IttiKoch

# ── Paths ──────────────────────────────────────────────────────────────────────── 
here           = os.path.dirname(__file__)
input_dir      = os.path.abspath(os.path.join(here, "..", "data"))
salicon_json   = os.path.abspath(os.path.join(here, "..", "..", "data/Salicon/fixations_val2014.json"))
output_dir     = os.path.join(here, "results")
os.makedirs(output_dir, exist_ok=True)

# ── Detectors ────────────────────────────────────────────────────────────────────
DETECTORS = {
    "IKN": IttiKoch,
    "AIM": pysaliency.AIM(location=os.path.join("app", "metrics", "pysal_models")),
    "SUN": pysaliency.SUN(location=os.path.join("app", "metrics", "pysal_models")),
    "BMS": BMS,
    "FineGrained": cv2.saliency.StaticSaliencyFineGrained_create(),
    "SpectralRes": cv2.saliency.StaticSaliencySpectralResidual_create(),
}

# ── Helpers ──────────────────────────────────────────────────────────────────────
def compute_sal_map(detector, img):
    if hasattr(detector, "computeSaliency"):
        ok, sal = detector.computeSaliency(img)
        if not ok or sal is None:
            return None
    else:
        sal = detector.saliency_map(img)
        if sal is None:
            return None
    sal = np.asarray(sal, dtype=np.float32)
    if sal.ndim == 3 and sal.shape[2] == 1:
        sal = sal[:, :, 0]
    return cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX)

def build_heatmap(sal_map):
    u8 = (sal_map * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)

# ── Load Salicon JSON ────────────────────────────────────────────────────────────
with open(salicon_json, "r") as f:
    sal_data = json.load(f)

# id → file_name
image_map = {img["id"]: img["file_name"] for img in sal_data["images"]}

# file_name → fixations list
fix_dict = {
    image_map[ann["image_id"]]: ann["fixations"]
    for ann in sal_data["annotations"]
}

# ── Process Each Image ───────────────────────────────────────────────────────────
for img_name in sorted(os.listdir(input_dir)):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        continue

    img_path = os.path.join(input_dir, img_name)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        print(f"⚠️ Cannot read {img_name}, skipping.")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # —— 1) Heatmaps montage 2×3 —— 
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (name, det) in zip(axes, DETECTORS.items()):
        sal = compute_sal_map(det, img_bgr)
        if sal is None:
            ax.set_title(f"{name} failed")
            ax.axis("off")
            continue
        heat = build_heatmap(sal)
        ax.imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_heatmaps.png"))
    plt.close(fig)

    # —— 2) Original + Fixations overlay 1×2 ——
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Overlay fixations
    axes[1].imshow(img_rgb)
    fixes = fix_dict.get(img_name, [])
    if fixes:
        xs, ys = zip(*fixes)
        axes[1].scatter(xs, ys, c="red", s=5, alpha=0.6)
    axes[1].set_title("Fixations Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_fixations.png"))
    plt.close(fig)

    print(f"✓ {img_name} → montages saved.")

print("All done!") 

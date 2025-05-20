import numpy as np
import cv2
from skimage.color import rgb2gray

class BMS:
    @staticmethod
    def binarize_img(gray, threshold):
        # Use faster numpy vector comparison to threshold image
        return (gray > threshold).astype(np.uint8)

    @staticmethod
    def activate_bool_map(bool_map):
        """
        Label connected components, then keep only
        those that do NOT touch the image border.
        Gestalt principle: areas surrounded with contrasting info are salient. 
        """
        # connectedComponentsWithStats gives:
        #   num_labels, label_map, stats, centroids
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(bool_map, connectivity=8)
        h, w = bool_map.shape
        attn = np.zeros_like(bool_map, dtype=np.float32)

        # stats[i] = [x, y, width, height, area]
        for label in range(1, num_labels):  # skip background=0
            x, y, width, height, area = stats[label]
            # If region touches any border, 
            # its bbox will start at 0 
            # or extend up to w/h
            if x == 0 or y == 0 or x + width >= w or y + height >= h:
                continue
            # Otherwise activate those pixels
            attn[label_map == label] = 1.0

        return attn

    @staticmethod
    def computeSaliency(img, n_thresholds=16, lb=25, ub=230):
        """
        img: BGR or RGB image array (HxWx3)
        returns: success flag, saliency map normalized to [0,1]
        """
        # 0) Validate input
        if img is None:
            print("Image not found.")
            return False, None
        if img.ndim not in [2,3]:
            print(f"Invalid image dim: {img.ndim} Must be 2 or 3")
            return False, None
        # 1) Convert to gray float in [0,255]
        gray = rgb2gray(img) * 255.0
        # 2) Generate thresholds (equally spaced)
        thresholds = np.linspace(lb, ub, n_thresholds, endpoint=False)

        # 3) For each threshold, build and activate the boolean map
        attn_map = np.zeros(gray.shape, dtype=np.float32)
        for thr in thresholds:
            bool_map = BMS.binarize_img(gray, thr)
            attn_map += BMS.activate_bool_map(bool_map)

        # 4) Smooth the attention map
        attn_map = cv2.GaussianBlur(attn_map, ksize=(0, 0), sigmaX=3)

        # 5) Normalize to [0,1]
        if attn_map.max() > attn_map.min(): # Conditoinal prevents 0 div when max=min
            sal_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        # If max=min, bool maps likely were all 0
        # or dropped (all vals touched borders)
        else:
            return False, None
        return True, sal_map

import numpy as np
import cv2
from skimage.color import gray2rgb, rgb2gray, rgb2lab
#from scipy.ndimage import label

class BMS:
    def binarize_img(image, threshold):
        h,w = image.shape[:2]
        for i in range(h):
            for j in range(w):
                if image[i,j] <= threshold:
                    image[i,j] = 0
                else:
                    image[i,j] = 1
        return image
    
    def touches_border(region_coords, bounding_box):
        y_bound, x_bound = bounding_box
        for x,y in region_coords:
            if x >= x_bound or y >= y_bound:
                return True
        return False
    
    def activate_bool_map(bool_map):
        # Label connected components in map
        label_map = cv2.connectedComponents(bool_map)
        attn_map = np.zeros_like(bool_map)
        # num_features uses 1-based indexing
        num_features = max(label_map)
        for label in range(1, num_features+1):
            # Find surrounded regions of px, activate its coord in attention
            region_coords = np.where(label_map == label)
            if not BMS.touches_border(region_coords, bool_map.shape):
                attn_map[region_coords] = 1
        return attn_map

    def compute_saliency(img_path, n_thresholds = 10, lb = 25, ub = 230):
        # Read input image & convert to grayscale
        input_img = cv2.imread(img_path)
        h,w = input_img.shape[:2]
        gray_img = rgb2gray(input_img)

        # Init binarizaton thresholds & maps list
        thresholds = np.arange(lb, ub, ub / n_thresholds)
        ls_bool_maps = []

        # Create boolean maps based on random thresholds
        for threshold in thresholds:
            bin_img = BMS.binarize_img(gray_img, threshold)
            ls_bool_maps.append(bin_img)
        # Stack lists horiztonally onto numpy arr for faster processing
        bool_maps = np.stack(ls_bool_maps, axis=0)
        attn_map = np.zeros_like(input_img.shape[:2])
        for map in bool_maps:
            # Increment regions marked for attention
            attn_map += BMS.activate_bool_map(map)
        # Ask for Gaussian blur with σₓ=3, let OpenCV pick kernel size:
        attn_map = cv2.GaussianBlur(attn_map, ksize=(0,0), sigmaX=3)


    

        


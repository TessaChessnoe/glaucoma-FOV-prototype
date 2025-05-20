import numpy as np
import cv2
from skimage.color import gray2rgb, rgb2gray, rgb2lab

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

    def compute_saliency(img_path, n_thresholds = 10, lb = 25, ub = 230):
        input_img = cv2.imread(img_path)
        h,w = input_img.shape[:2]
        gray_img = rgb2gray(input_img)

        thresholds = np.arange(lb, ub, ub / n_thresholds)
        ls_bool_maps = []

        # Create boolean maps based on random thresholds
        for threshold in thresholds:
            bin_img = BMS.binarize_img(gray_img, threshold)
            ls_bool_maps.append(bin_img)
        # Stack lists horiztonally onto numpy arr for faster processing
        bool_maps = np.stack(ls_bool_maps, axis=0)

        
    

        


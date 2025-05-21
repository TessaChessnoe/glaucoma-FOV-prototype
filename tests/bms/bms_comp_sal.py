from app.models import BMS
import cv2
import numpy as np
import os

def main():
    here = os.path.dirname(__file__)
    img_path = os.path.join(here,'..','data/bluejay.jpg')

    input_img = cv2.imread(img_path)

    success, sal_map = BMS.computeSaliency(input_img)
    if not success:
        raise RuntimeError("ERROR: Saliency computation failed")
    
    sal_u8 = (sal_map * 255).astype(np.uint8)
    sal_color = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(here,'sal_map.png'), sal_u8)
    cv2.imwrite(os.path.join(here,'sal_colormap.png'), sal_color)
    print("Successfully wrote saliency_gray.png and saliency_color.png")

if __name__ == '__main__':
    main()


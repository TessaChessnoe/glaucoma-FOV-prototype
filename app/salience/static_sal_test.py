import os
import json
import cv2
from tqdm import tqdm

def load_fixations(jsonn):
    with open(jsonn) as file:
        data_train = json.load(file)
    fix_train = {ann["image_id"]: ann["fixations"] for ann in data_train["annotations"]}

def sal_im_montage(sal_obj, input_dir="dataset/Salicon/test", output_dir='dataset/Salicon/results'):
    # Ensure output dir is created
    os.makedirs(output_dir)
    # Config for accepted ext
    accepted_ext = ('.jpg', 'jpeg', '.png', '.tif')
    # Scan directory for images
    input_images = [file for file in sorted(os.listdir(input_dir))
                if file.lower().endswith(accepted_ext)]
    
    print(f"Detected {len(input_images)} in {input_dir}.")
    # Iterate over input images
    for filename in tqdm(input_images, desc="Processing images"):
        img_path = os.path.join(input_dir, filename)
        # Read images
        img = cv2.imread(img_path)
        # Log any read failures in console w/o breaking out
        if img is None: 
            tqdm.write(f"ERROR: {img} could not be read.")
            continue
        # Compute saliency map
        success, sal_map = sal_obj.computeSaliency(img)
        # Convert saliency map to 8-bit image
        map_8u = (sal_map * 255).astype('uint8')
        # Display image and saliency map side-by-side
        h, w = img.shape[:2]
        target = (w // 2, h // 2)
        comparison = cv2.hconcat([
            cv2.resize(img,   target),
            cv2.cvtColor(cv2.resize(map_8u, target), cv2.COLOR_GRAY2BGR),
        ])
        # Output images to results dir
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, comparison)
    print(f"Done! Computed static saliency maps for {len(input_images)} images!")
    
    





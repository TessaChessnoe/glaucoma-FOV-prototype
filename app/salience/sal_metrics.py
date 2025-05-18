import os
import json
import cv2
import pandas as pd
from tqdm import tqdm

# def load_img(img_path):
#     # Config for accepted ext
#     accepted_ext = ('.jpg', 'jpeg', '.png', '.tif')
#     # Scan directory for images
#     input_images = [file for file in sorted(os.listdir(input_dir))
#                 if file.lower().endswith(accepted_ext)]
#     for filename in input_images:

# def sal_im_montage(sal_obj, input_dir="dataset/Salicon/test", output_dir='dataset/Salicon/results'):
#     # Ensure output dir is created
#     os.makedirs(output_dir)
#     # Config for accepted ext
#     accepted_ext = ('.jpg', 'jpeg', '.png', '.tif')
#     # Scan directory for images
#     input_images = [file for file in sorted(os.listdir(input_dir))
#                 if file.lower().endswith(accepted_ext)]
    
#     print(f"Detected {len(input_images)} in {input_dir}.")
#     # Iterate over input images
#     for filename in tqdm(input_images, desc="Processing images"):
#         img_path = os.path.join(input_dir, filename)
#         # Read images
#         img = cv2.imread(img_path)
#         # Log any read failures in console w/o breaking out
#         if img is None: 
#             tqdm.write(f"ERROR: {img} could not be read.")
#             continue
#         # Compute saliency map
#         success, sal_map = sal_obj.computeSaliency(img)
#         # Convert saliency map to 8-bit image
#         map_8u = (sal_map * 255).astype('uint8')
#         # Display image and saliency map side-by-side
#         h, w = img.shape[:2]
#         target = (w // 2, h // 2)
#         comparison = cv2.hconcat([
#             cv2.resize(img,   target),
#             cv2.cvtColor(cv2.resize(map_8u, target), cv2.COLOR_GRAY2BGR),
#         ])
#         # Output images to results dir
#         out_path = os.path.join(output_dir, filename)
#         cv2.imwrite(out_path, comparison)
#     print(f"Done! Computed static saliency maps for {len(input_images)} images!")

def load_fix_map(jsonn, input_dir='app/dataset/salience/salicon'):
    json_path = os.path.join(input_dir, jsonn)
    with open(json_path) as file:
        data = json.load(file)
    fix_map = {ann["image_id"]: ann["fixations"] for ann in data["annotations"]}
    return fix_map

def load_im_map(jsonn, input_dir='app/dataset/salience/salicon'):
    json_path = os.path.join(input_dir, jsonn)
    with open(json_path) as file:
        data = json.load(file)
    im_map = {img["file_name"]: img["id"] for img in data["img"]}
    return im_map

def build_im2fix_lookup(im_map, fix_map):
    im_map_df = pd.DataFrame(im_map)
    im_map_df.rename(columns={'id':'image_id'})
    fix_map_df = pd.DataFrame(fix_map)
    fix_lookup = im_map_df.merge(fix_map_df, on='image_id')
    return fix_lookup

# Build fixations lookup table for validation images
im_map_val = load_im_map('fixations_train2014')
fix_map_val = load_fix_map('fixations_train2014')
val_images = build_im2fix_lookup(im_map_val, im_map_val)

metrics = []
input_dir = 'app/dataset/salience/Salicon'
input_images = [filename for filename in sorted(os.listdir(input_dir))
                if filename in val_images['file_name']]
fine = cv2.saliency.StaticSaliencyFineGrained_create()
for img, img_id in val_images:
    sal_map = fine.computeSaliency(img)
    normal_map = (sal_map - sal_map.mean() / sal_map.std())
    vals = [normal_map[int(x), int(y)] for x,y in val_images[im]]







import os
from tqdm import tqdm
import cv2

# Test OpenCV's static salience algos
# Create saliency objs
spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
fine = cv2.saliency.StaticSaliencyFineGrained_create()

# Set input & output dirs
input_dir = "dataset/Salicon/test"
output_dir = "dataset/Salicon/results/static"
os.makedirs(output_dir, exist_ok=True)

accepted_ext = ('.jpg', 'jpeg', '.png', '.tif')
input_images = [file for file in sorted(os.listdir(input_dir))
                if file.lower().endswith(accepted_ext)]

print(f"Detected {len(input_images)} in {input_dir}.")
# Iterate over test images
for filename in tqdm(input_images, desc="Processing images"):
    img_path = os.path.join(input_dir, filename)
    # Read images
    img = cv2.imread(img_path)
    # Log any read failures in console w/o breaking out
    if img is None: 
        tqdm.write(f"ERROR: {img} could not be read.")
        continue
    # Compute Spectral Residual saliency
    success, map_sr = spectral.computeSaliency(img)
    # Convert saliency map to 8-bit image
    sr_8u = (map_sr * 255).astype('uint8')

    # Compute Fine-Grained saliency
    success, map_fg = fine.computeSaliency(img)
    fg_8u = (map_fg * 255).astype('uint8')

    # Display image and saliency map side-by-side
    h, w = img.shape[:2]
    target = (w // 2, h // 2)
    comparison = cv2.hconcat([
        cv2.resize(img,   target),
        cv2.cvtColor(cv2.resize(sr_8u, target), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.resize(fg_8u, target), cv2.COLOR_GRAY2BGR),
    ])
    # Output images to results dir
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, comparison)
print(f"Done! Computed static saliency maps for {len(input_images)} images!")

    





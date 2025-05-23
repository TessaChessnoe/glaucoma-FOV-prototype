import json
import os
import pandas as pd

def pretty_print_annotation(ann, k=3):
    coords = ann["fixations"]
    head = coords[:k]
    tail = coords[-k:]
    truncated = head + ["…"] + tail
    snippet = {
        "image_id": ann["image_id"],
        "fixations": truncated
    }
    print(json.dumps(snippet, indent=2))

input_dir = 'data/Salicon'

# Adjust the path to your actual JSON file
with open(os.path.join(input_dir, "fixations_train2014.json")) as f:
    data = json.load(f)

pretty_print_annotation(data["annotations"][0], k=5)

image_map = {img["id"]: img["file_name"] for img in data["images"]}
rows = []
for ann in data["annotations"][:10]:  # just first 10 for quick check
    coords = ann["fixations"]
    sample = coords[:5] + ["…"] + coords[-5:]
    rows.append({
        "file_name": image_map[ann["image_id"]],
        "image_id":   ann["image_id"],
        "n_fixations": len(coords),
        "sample_fixations": sample
    })

df = pd.DataFrame(rows)
print(df)
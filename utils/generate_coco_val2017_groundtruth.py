import json
import os

# Path to your existing MS-COCO captions file
annotations_path = "./datasets/coco_subset/annotations/captions_val2017.json"

# Output ground truth json
output_path = "./datasets/nocaps/ground_truth_coco_val2017.json"

# Load captions
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

ground_truth = {}

for ann in coco_data['annotations']:
    image_id = ann['image_id']
    caption = ann['caption']

    if image_id not in ground_truth:
        ground_truth[image_id] = []
    ground_truth[image_id].append(caption)

# Save ground truth in proper format
with open(output_path, 'w') as f:
    json.dump(ground_truth, f, indent=4)

print(f"✅ Ground truth file created at: {output_path}")

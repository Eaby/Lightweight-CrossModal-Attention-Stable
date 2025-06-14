import json
import os
from datasets import load_dataset

# Load nocaps validation set from HuggingFace datasets
dataset = load_dataset("nocaps", split="validation")

# Build the ground truth dictionary
ground_truth = {}

for item in dataset:
    image_id = str(item['image_id'])
    captions = item['captions']
    ground_truth[image_id] = captions

# Create directory if not exists
os.makedirs("./datasets/nocaps", exist_ok=True)

# Save as ground_truth.json
with open("./datasets/nocaps/ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)

print("✅ ground_truth.json created successfully!")

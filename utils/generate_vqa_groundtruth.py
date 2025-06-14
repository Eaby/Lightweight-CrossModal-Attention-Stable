import json
import os

# Paths to your existing annotation files
vqa_annotations_path = "./datasets/vqa2/v2_mscoco_val2014_annotations.json"

# Output file
output_groundtruth_path = "./datasets/vqa2/ground_truth.json"

# Load annotations
with open(vqa_annotations_path, 'r') as f:
    annotations = json.load(f)["annotations"]

# Build dictionary: question_id -> answer
ground_truth = {}
for ann in annotations:
    question_id = ann["question_id"]
    answers = [a["answer"] for a in ann["answers"]]
    ground_truth[str(question_id)] = answers

# Save output
os.makedirs("./datasets/vqa2", exist_ok=True)
with open(output_groundtruth_path, "w") as f:
    json.dump(ground_truth, f, indent=4)

print(f"✅ VQA Ground Truth generated: {output_groundtruth_path}")

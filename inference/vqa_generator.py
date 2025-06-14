import json
import random
import os

# Simulate dummy model predictions for testing
# (later you will replace this with your actual VQA model output)

# Load ground truth question IDs
ground_truth_path = "./datasets/vqa2/ground_truth.json"
with open(ground_truth_path, 'r') as f:
    ground_truth = json.load(f)

# Simulate predictions
predictions = {}
for qid in ground_truth.keys():
    predictions[qid] = random.choice(ground_truth[qid])

# Save predictions
os.makedirs("./results/vqa", exist_ok=True)
predictions_path = "./results/vqa_predictions.json"
with open(predictions_path, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"✅ VQA Predictions generated: {predictions_path}")

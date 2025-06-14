import json
import os
from sklearn.metrics import accuracy_score

class VQAEvaluator:
    def __init__(self, ground_truth_path, predictions_path, save_dir):
        self.ground_truth_path = ground_truth_path
        self.predictions_path = predictions_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def load_data(self):
        with open(self.ground_truth_path) as f:
            ground_truth = json.load(f)
        with open(self.predictions_path) as f:
            predictions = json.load(f)
        return ground_truth, predictions

    def evaluate(self):
        ground_truth, predictions = self.load_data()

        y_true = [ground_truth[qid] for qid in predictions]
        y_pred = [predictions[qid] for qid in predictions]

        acc = accuracy_score(y_true, y_pred)
        with open(os.path.join(self.save_dir, "vqa_results.json"), "w") as f:
            json.dump({"VQA Accuracy": acc}, f, indent=4)

        print({"VQA Accuracy": acc})


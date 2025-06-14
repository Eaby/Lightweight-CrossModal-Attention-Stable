import json
import os
from sklearn.metrics import accuracy_score
from collections import Counter
from tqdm import tqdm

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

    def majority_vote(self, answers):
        counter = Counter(answers)
        return counter.most_common(1)[0][0]

    def evaluate(self):
        ground_truth, predictions = self.load_data()

        # Normalize keys
        gt_keys = set(ground_truth.keys())
        pred_keys = set(predictions.keys())
        common_keys = gt_keys & pred_keys

        y_true = []
        y_pred = []

        for key in tqdm(common_keys, desc="Evaluating VQA Samples"):
            gt_answers = ground_truth[key]  # list of multiple ground-truth answers
            majority_gt = self.majority_vote(gt_answers)
            pred_answer = predictions[key]

            y_true.append(majority_gt.strip().lower())
            y_pred.append(pred_answer.strip().lower())

        acc = accuracy_score(y_true, y_pred)

        results = {"VQA Accuracy": acc}
        with open(os.path.join(self.save_dir, "vqa_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(results)


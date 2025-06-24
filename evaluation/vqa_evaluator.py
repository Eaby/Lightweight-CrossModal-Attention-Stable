import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from types import SimpleNamespace # Import SimpleNamespace for config type hinting

# Add the path to your VQA API tools to sys.path
sys.path.append(os.path.abspath('./utils/vqa_eval_tools'))

try:
    from vqa import VQA 
    from vqaEval import VQAEval 
except ImportError as e:
    print(f"Error importing VQA API tools: {e}")
    print("Please ensure 'utils/vqa_eval_tools' is correctly set up in your sys.path "
          "and contains vqa.py and vqaEval.py from the official VQA tools.")
    sys.exit(1) 

class VQAEvaluator:
    def __init__(self, config):
        self.config = config 
        self.save_dir = os.path.join(self.config.results_dir, 'vqa') 
        os.makedirs(self.save_dir, exist_ok=True)

        self.ground_truth_questions_path = os.path.join(self.config.datasets_path, self.config.vqa_val_questions) 
        self.ground_truth_annotations_path = os.path.join(self.config.datasets_path, self.config.vqa_val_annotations) 
        
        self.predictions_filename = self.config.vqa_predictions_filename 
        self.predictions_path = os.path.join(self.save_dir, self.predictions_filename)

    def evaluate(self):
        print(f"Loading VQA ground truth questions from: {self.ground_truth_questions_path}")
        print(f"Loading VQA ground truth annotations from: {self.ground_truth_annotations_path}")
        print(f"Loading VQA predictions from: {self.predictions_path}")

        try:
            vqa_gt = VQA(self.ground_truth_annotations_path, self.ground_truth_questions_path)
        except Exception as e:
            print(f"Error loading VQA ground truth: {e}")
            print("Please ensure your VQA annotation and question files are correct and in the expected format.")
            return {"VQA Accuracy": 0.0}

        if not os.path.exists(self.predictions_path):
            print(f"Error: VQA predictions file not found at {self.predictions_path}")
            print("Please ensure your inference script generates predictions in the correct format "
                  f"and saves them to {self.predictions_path}.")
            return {"VQA Accuracy": 0.0}

        with open(self.predictions_path, 'r') as f:
            vqa_predictions_raw = json.load(f)
        
        try:
            vqa_res = vqa_gt.loadRes(vqa_predictions_raw)
        except Exception as e:
            print(f"Error loading VQA results: {e}")
            print("Please ensure your prediction file is in the correct VQA results format: "
                  f"[{{\"question_id\": int, \"answer\": \"str\"}}, ...]")
            return {"VQA Accuracy": 0.0}

        vqa_eval = VQAEval(vqa_gt, vqa_res, n=10) # VQAv2 typically has 10 annotations per question

        vqa_eval.evaluate()

        results = vqa_eval.accuracy # This is the dictionary {overall, perQuestionType, perAnswerType}
        
        print("\n--- VQA Results ---")
        if 'overall' in results:
            print(f"  Overall Accuracy: {results['overall']:.4f}") # 'overall' is a float
        
        # FIX: Iterate through nested dictionaries for perQuestionType and perAnswerType
        if 'perQuestionType' in results:
            print("  Per Question Type Accuracy:")
            for q_type, score in results['perQuestionType'].items():
                print(f"    - {q_type}: {score:.4f}")
        
        if 'perAnswerType' in results:
            print("  Per Answer Type Accuracy:")
            for a_type, score in results['perAnswerType'].items():
                print(f"    - {a_type}: {score:.4f}")
        
        print("---------------------\n")

        with open(os.path.join(self.save_dir, "vqa_results.json"), "w") as f:
            json.dump(results, f, indent=4) # Results can be dumped as is, JSON handles nested dicts
        
        return results

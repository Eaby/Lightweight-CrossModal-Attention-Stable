import torch
import os
import json

class EfficiencyEvaluator:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def evaluate(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        model_size = num_params * 4 / 1e6  # size in MB (assuming float32)

        results = {
            "Num Parameters": num_params,
            "Model Size (MB)": model_size,
            "Rank k": self.model.rank_attention.rank_k
        }

        with open(os.path.join(self.save_dir, "efficiency_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(results)


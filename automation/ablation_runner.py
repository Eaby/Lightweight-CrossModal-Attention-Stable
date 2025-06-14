import os
import copy
import yaml
import torch

from data_loader import CrossModalDatasetLoader
from models.multimodal_model import CrossModalModel
from trainer.train import Trainer

from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator

# Disable huggingface tokenizer parallelism warnings:
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AblationRunner:
    def __init__(self, config_path="./configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_ablation(self, rank_list=[16, 32, 64, 128, 256]):
        for rank_k in rank_list:
            print(f"\n🚀 Running experiment with rank_k = {rank_k}")

            # Clone config for this experiment
            config = copy.deepcopy(self.base_config)
            config['rank_k'] = rank_k

            result_dir = f"./results/ablation/rank_{rank_k}"
            os.makedirs(result_dir, exist_ok=True)

            # Load dataset
            loader = CrossModalDatasetLoader(config)
            dataset = loader.load_coco(split="train")

            # Build model
            model = CrossModalModel(device=self.device, rank_k=rank_k).to(self.device)

            # Train
            trainer = Trainer(model, dataset, config, self.device)
            trainer.train()

            # Save model after training
            model_path = os.path.join(result_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model saved at: {model_path}")

            # Evaluation (Retrieval)
            val_dataset = loader.load_coco(split="val")
            retrieval_eval = RetrievalEvaluator(model, val_dataset, self.device, save_dir=os.path.join(result_dir, "retrieval"))
            retrieval_eval.evaluate()

            # Efficiency metrics
            eff_eval = EfficiencyEvaluator(model, save_dir=os.path.join(result_dir, "efficiency"))
            eff_eval.evaluate()

            print(f"✅ Completed experiment rank_k={rank_k}")

if __name__ == "__main__":
    runner = AblationRunner()
    runner.run_ablation()


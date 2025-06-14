import os
from evaluation.captioning_evaluator import CaptioningEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.vqa_evaluator import VQAEvaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator
from models.multimodal_model import CrossModalModel
import torch
import yaml
from utils.gpu_manager import GPUMemoryManager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load config
with open('./configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load device
gpu_manager = GPUMemoryManager(preferred_device=None)
device = gpu_manager.get_device()

# Load trained model
model = CrossModalModel(device=device, rank_k=config['rank_k']).to(device)
model.load_state_dict(torch.load("./results/model.pth"))
model.eval()

# Captioning
caption_eval = CaptioningEvaluator(
    ground_truth_path="./datasets/nocaps/ground_truth_coco_val2017.json",
    generated_captions_path="./results/generated_captions_coco_val2017.json",
    save_dir="./results/captioning/"
)
caption_eval.evaluate()

# VQA
vqa_eval = VQAEvaluator(
    ground_truth_path="./datasets/vqa2/ground_truth.json",
    predictions_path="./results/vqa_predictions.json",
    save_dir="./results/vqa/"
)
vqa_eval.evaluate()

# Efficiency
eff_eval = EfficiencyEvaluator(model, save_dir="./results/efficiency/")
eff_eval.evaluate()

print("✅ Full pipeline completed successfully.")


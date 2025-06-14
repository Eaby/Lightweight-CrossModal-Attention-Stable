import torch
import yaml
from models.multimodal_model import CrossModalModel
from evaluation.efficiency_evaluator import EfficiencyEvaluator
from utils.gpu_manager import GPUMemoryManager

# Load config
with open('./configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load device
gpu_manager = GPUMemoryManager(preferred_device=None)
device = gpu_manager.get_device()

# Load model checkpoint path
model = CrossModalModel(device=device, rank_k=config['rank_k']).to(device)
model.load_state_dict(torch.load("./results/model.pth"))
model.eval()

# Run efficiency evaluator
eff_eval = EfficiencyEvaluator(model, save_dir="./results/efficiency/")
eff_eval.evaluate()

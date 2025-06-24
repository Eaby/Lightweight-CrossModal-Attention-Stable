import os
import torch
import numpy as np
import json
from types import SimpleNamespace

# ✅ Correct model import
from models.rank_attention import RankAttentionModel as MultiModalModel
from data_loader import CrossModalDatasetLoader
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.captioning_evaluator import CaptioningEvaluator
from evaluation.vqa_evaluator import VQAEvaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator
from evaluation.ablation_runner import AblationRunner

from utils.config_loader import load_config  # if you have this

def main():
    print("Using device:", device := ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load config
    with open("configs/config.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    dataset_loader = CrossModalDatasetLoader(config)

    print("Loading datasets for evaluation...")
    datasets = {
        'coco_val': dataset_loader.load_coco(split='val'),
        'flickr': dataset_loader.load_flickr30k(),
        'nocaps': dataset_loader.load_nocaps(),
        'vqa_val': dataset_loader.load_vqa(split='val'),
        'okvqa_val': dataset_loader.load_okvqa(split='val')
    }

    # ✅ Load model
    model = MultiModalModel(config).to(device)
    checkpoint_path = "./results/final_model/model_final.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run evaluations
    print("\nRunning Retrieval Evaluation on COCO val dataset...")
    retrieval_eval = RetrievalEvaluator(model=model, dataset=datasets['coco_val'], device=device,
                                        save_dir="./results/retrieval/coco", config=config)
    retrieval_results = retrieval_eval.evaluate()

    print("\nRunning Captioning Evaluation on COCO val dataset...")
    caption_eval = CaptioningEvaluator(config=config)
    caption_results = caption_eval.evaluate()

    print("\nRunning VQA Evaluation...")
    vqa_eval = VQAEvaluator(config=config)
    vqa_results = vqa_eval.evaluate()

    print("\nRunning Efficiency Evaluation...")
    efficiency_eval = EfficiencyEvaluator(config=config)
    efficiency_results = efficiency_eval.evaluate()

    print("\nRunning Ablation Study Evaluation...")
    ablation_eval = AblationRunner(config=config)
    ablation_results = ablation_eval.run()

    all_results = {
        "retrieval": retrieval_results,
        "captioning": caption_results,
        "vqa": vqa_results,
        "efficiency": efficiency_results,
        "ablation": ablation_results
    }

    # Save all combined results
    os.makedirs("./results/evaluations_during_training/final_eval", exist_ok=True)
    with open("./results/evaluations_during_training/final_eval/all_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n✅ All evaluations completed successfully!\n")

if __name__ == "__main__":
    import yaml
    main()


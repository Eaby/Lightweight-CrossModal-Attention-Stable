import os
import json
import torch
from types import SimpleNamespace 
import copy 
import random
import numpy as np
import sys

# Add project root to sys.path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all necessary components
from utils.config_loader import load_config
from data_loader import CrossModalDatasetLoader # Assuming BaseDataset for evaluation
from models.multimodal_model import LightweightMultimodalModel
from trainer.train import MultiTaskTrainer, prepare_vqa_targets 
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.captioning_evaluator import CaptioningEvaluator
from evaluation.vqa_evaluator import VQAEvaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator
from inference.caption_generator import CaptionGenerator
from inference.vqa_generator import VQAGenerator

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AblationRunner:
    def __init__(self, base_config_path="configs/config.yaml"):
        self.base_config_dict = load_config(base_config_path) # Keep dict for passing to data_loader
        self.base_config = SimpleNamespace(**self.base_config_dict) # Convert to SimpleNamespace for easy access
        
        self.device = torch.device(self.base_config.preferred_device if self.base_config.preferred_device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")

        self.dataset_loader = CrossModalDatasetLoader(self.base_config_dict) # Pass dict to data_loader
        self.val_datasets = self._load_validation_datasets()

        self.ablation_results = {}
        self.ablation_output_dir = os.path.join(self.base_config.results_dir, "ablation_studies")
        os.makedirs(self.ablation_output_dir, exist_ok=True)

    def _load_validation_datasets(self):
        datasets = {}
        print("Loading validation datasets for evaluation...")
        try:
            datasets['coco_val'] = self.dataset_loader.load_coco(split="val")
        except Exception as e:
            print(f"Could not load COCO val dataset: {e}")
        try:
            datasets['flickr30k'] = self.dataset_loader.load_flickr30k()
        except Exception as e:
            print(f"Could not load Flickr30K dataset: {e}")
        try:
            datasets['vqa_val'] = self.dataset_loader.load_vqa(split="val")
        except Exception as e:
            print(f"Could not load VQA val dataset: {e}")
        try:
            datasets['nocaps'] = self.dataset_loader.load_nocaps()
        except Exception as e:
            print(f"Could not load NoCaps dataset: {e}")
        try:
            datasets['okvqa_val'] = self.dataset_loader.load_okvqa(split="val")
        except Exception as e:
            print(f"Could not load OK-VQA val dataset: {e}")
        
        # Check if necessary datasets are loaded
        if any(ds is None for ds in datasets.values()):
            print("Warning: Some validation datasets could not be loaded. Evaluation for these tasks might be skipped.")

        return datasets
    
    # Modified to accept optional model_path
    def _run_single_experiment(self, experiment_name, config_override, model_path=None):
        print(f"\n--- Running Experiment: {experiment_name} ---")
        
        current_config_dict = copy.deepcopy(self.base_config_dict)
        for key, value in config_override.items():
            if isinstance(value, dict): # Handle nested dictionaries
                # Convert SimpleNamespace back to dict for modification, then back to SimpleNamespace
                nested_dict = current_config_dict.get(key, {})
                nested_dict.update(value)
                current_config_dict[key] = nested_dict
            else:
                current_config_dict[key] = value
        
        current_config = SimpleNamespace(**current_config_dict) # Convert to SimpleNamespace for use
        
        set_seed(current_config.seed)

        exp_save_dir = os.path.join(self.ablation_output_dir, experiment_name)
        os.makedirs(exp_save_dir, exist_ok=True)
        
        with open(os.path.join(exp_save_dir, "config.json"), "w") as f:
            json.dump(current_config.__dict__, f, indent=4) 

        # --- Model Initialization & (Optional) Loading ---
        model = LightweightMultimodalModel(current_config).to(self.device)
        
        if model_path:
            print(f"Loading pre-trained model from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Pre-trained model loaded successfully.")
            except Exception as e:
                print(f"Error loading pre-trained model from {model_path}: {e}. Initializing with random weights.")
        else:
            print("No pre-trained model path provided. Training model from scratch.")
            # --- Training ---
            train_datasets = {
                'coco_train': self.dataset_loader.load_coco(split="train"),
                'vqa_train': self.dataset_loader.load_vqa(split="train")
            }
            if any(ds is None for ds in train_datasets.values()):
                print("Error: Not all required training datasets could be loaded. Skipping training.")
                return # Abort experiment if training data is missing

            trainer = MultiTaskTrainer(
                model=model,
                config=current_config_dict, # Pass dict config to trainer
                train_datasets=train_datasets,
                val_datasets=self.val_datasets, # Pass val datasets to trainer for its internal eval
                tokenizer=self.dataset_loader.tokenizer
            )
            trainer.train()

        # --- Evaluation ---
        print("\n--- Running Evaluations ---")
        exp_results = {}
        
        model.eval() # Set model to evaluation mode

        # a. Retrieval Evaluation
        if 'coco_val' in self.val_datasets and self.val_datasets['coco_val'] is not None:
            retrieval_eval_dir = os.path.join(exp_save_dir, "retrieval_metrics")
            os.makedirs(retrieval_eval_dir, exist_ok=True)
            retrieval_evaluator = RetrievalEvaluator(model, self.val_datasets['coco_val'], self.device, retrieval_eval_dir, current_config)
            retrieval_results = retrieval_evaluator.evaluate()
            exp_results['retrieval_coco'] = retrieval_results['Average']
        else:
            print("Skipping Retrieval Evaluation: COCO val dataset not loaded.")

        # b. Captioning Evaluation (Generate then Evaluate)
        if 'coco_val' in self.val_datasets and self.val_datasets['coco_val'] is not None:
            print("Generating Captions...")
            captioning_gen_output_dir = os.path.join(exp_save_dir, "captioning_generated")
            os.makedirs(captioning_gen_output_dir, exist_ok=True)
            
            caption_output_filename = current_config.caption_predictions_filename
            caption_generator = CaptionGenerator(model, self.val_datasets['coco_val'], self.device, current_config)
            generated_captions_path = caption_generator.generate_captions(
                output_filename=caption_output_filename,
                base_output_dir=captioning_gen_output_dir
            )
            
            print("Evaluating Captioning...")
            captioning_results_dir = os.path.join(exp_save_dir, "captioning_metrics")
            os.makedirs(captioning_results_dir, exist_ok=True)
            
            captioning_evaluator = CaptioningEvaluator(
                ground_truth_json_path=os.path.join(current_config.datasets_path, current_config.coco_val_annotations),
                generated_captions_json_path=generated_captions_path,
                save_dir=captioning_results_dir
            )
            captioning_results = captioning_evaluator.evaluate()
            exp_results['captioning_coco'] = captioning_results # Store all results
        else:
            print("Skipping Captioning Evaluation: COCO val dataset not loaded.")

        # c. VQA Evaluation (Generate then Evaluate)
        if 'vqa_val' in self.val_datasets and self.val_datasets['vqa_val'] is not None:
            print("Generating VQA Answers...")
            vqa_gen_output_dir = os.path.join(exp_save_dir, "vqa_generated")
            os.makedirs(vqa_gen_output_dir, exist_ok=True)

            vqa_output_filename = current_config.vqa_predictions_filename
            vqa_generator = VQAGenerator(model, self.val_datasets['vqa_val'], self.device, current_config)
            generated_vqa_path = vqa_generator.generate_answers(
                output_filename=vqa_output_filename,
                base_output_dir=vqa_gen_output_dir
            )

            print("Evaluating VQA...")
            vqa_results_dir = os.path.join(exp_save_dir, "vqa_metrics")
            os.makedirs(vqa_results_dir, exist_ok=True)
            
            vqa_evaluator = VQAEvaluator(config=current_config)
            vqa_evaluator.predictions_path = generated_vqa_path # Ensure evaluator points to current generated file
            vqa_evaluator.save_dir = vqa_results_dir # Ensure evaluator saves to current dir
            vqa_results = vqa_evaluator.evaluate()
            exp_results['vqa_v2'] = vqa_results # Store all results
        else:
            print("Skipping VQA Evaluation: VQA val dataset not loaded.")

        # d. Efficiency Evaluation
        if 'coco_val' in self.val_datasets and self.val_datasets['coco_val'] is not None:
            print("Evaluating Efficiency...")
            efficiency_results_dir = os.path.join(exp_save_dir, "efficiency_metrics")
            os.makedirs(efficiency_results_dir, exist_ok=True)
            
            efficiency_evaluator = EfficiencyEvaluator(model, self.val_datasets['coco_val'], current_config, self.device, efficiency_results_dir)
            efficiency_results = efficiency_evaluator.evaluate()
            exp_results['efficiency'] = efficiency_results # Store all results
        else:
            print("Skipping Efficiency Evaluation: COCO val dataset not loaded.")

        exp_results['rank_k'] = current_config.rank_k # Store rank for easy access in summary

        self.ablation_results[experiment_name] = exp_results
        
        with open(os.path.join(self.ablation_output_dir, "ablation_summary.json"), "w") as f:
            json.dump(self.ablation_results, f, indent=4)

        print(f"--- Experiment '{experiment_name}' finished. Results logged. ---")

    # NEW: Method to run full evaluation on a specific pre-trained model
    def run_full_evaluation_on_model(self, model_path):
        experiment_name = f"final_model_evaluation_{os.path.basename(model_path).replace('.pth', '')}"
        # No config overrides needed for final eval, just use base config
        self._run_single_experiment(experiment_name, {}, model_path=model_path)
        print(f"\nFull evaluation for {model_path} completed.")

    def run_ablation_studies(self):
        print("Starting ablation studies...")

        # --- 1. Rank Analysis ---
        ranks_to_test = [16, 32, 64, 128, 256]
        for rank in ranks_to_test:
            experiment_name = f"rank_analysis_k_{rank}"
            config_override = {"rank_k": rank}
            self._run_single_experiment(experiment_name, config_override)
        
        # --- 2. Training Strategies ---
        experiment_name = "training_strategy_retrieval_only"
        config_override = {"task_weights": {"retrieval": 1.0, "captioning": 0.0, "vqa": 0.0}}
        self._run_single_experiment(experiment_name, config_override)

        experiment_name = "training_strategy_captioning_only"
        config_override = {"task_weights": {"retrieval": 0.0, "captioning": 1.0, "vqa": 0.0}}
        self._run_single_experiment(experiment_name, config_override)
        
        experiment_name = "training_strategy_vqa_only"
        config_override = {"task_weights": {"retrieval": 0.0, "captioning": 0.0, "vqa": 1.0}}
        self._run_single_experiment(experiment_name, config_override)

        print("\nAll ablation studies completed. Summary saved to ablation_summary.json")

# Main execution block for AblationRunner (can be run directly or called)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies or full evaluation.")
    parser.add_argument('--mode', type=str, default='ablation', choices=['ablation', 'final_eval'],
                        help="Mode to run: 'ablation' for ablation studies, 'final_eval' for single full evaluation on trained model.")
    parser.add_argument('--model_path', type=str, 
                        default='./results/final_model/model_final.pth',
                        help="Path to the trained model checkpoint for 'final_eval' mode.")
    parser.add_argument('--config_path', type=str, default='configs/config.yaml',
                        help="Path to the base configuration file.")
    
    args = parser.parse_args()

    runner = AblationRunner(base_config_path=args.config_path)

    if args.mode == 'ablation':
        runner.run_ablation_studies()
    elif args.mode == 'final_eval':
        runner.run_full_evaluation_on_model(args.model_path)

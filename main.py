import torch
import os
import sys
from types import SimpleNamespace # For dot notation config access

# Add project root to sys.path to allow absolute imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all necessary components
from utils.config_loader import load_config
from data_loader import CrossModalDatasetLoader # This also uses BaseDataset internally
from models.multimodal_model import LightweightMultimodalModel
from trainer.train import MultiTaskTrainer
# Removed AblationRunner import from main.py as it's typically run separately or controlled by args
# from evaluation.ablation_runner import AblationRunner # For running full ablation studies

# It's good practice to ensure reproducibility
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # CORRECTED: Use manual_seed_all for setting CUDA seed
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 1. Load Configuration
    config_dict = load_config("configs/config.yaml")
    config = SimpleNamespace(**config_dict) # Access config parameters via dot notation

    print(f"Project: {config.project_name}")
    print(f"Configuration Loaded: {config.__dict__}")

    # 2. Set Device
    device = torch.device(config.preferred_device if config.preferred_device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 3. Set Random Seed for reproducibility
    set_seed(config.seed)

    # 4. Initialize Data Loaders and Datasets
    print("\nLoading datasets...")
    dataset_loader = CrossModalDatasetLoader(config_dict) # Pass dict config to data_loader

    # Load training datasets
    train_datasets = {
        'coco_train': dataset_loader.load_coco(split="train"),
        'vqa_train': dataset_loader.load_vqa(split="train")
        # Add other training datasets if you expand, e.g., Flickr30K train
    }
    # Load validation datasets for evaluation
    val_datasets = {
        'coco_val': dataset_loader.load_coco(split="val"),
        'vqa_val': dataset_loader.load_vqa(split="val")
        # Add other validation datasets if you expand, e.g., Flickr30K for retrieval
        # NoCaps and OK-VQA should be loaded here if used for evaluation during training
    }
    
    # Check if necessary datasets are loaded
    if any(ds is None for ds in train_datasets.values()):
        print("Error: Not all required training datasets could be loaded. Please check paths in config.yaml.")
        return
    if any(ds is None for ds in val_datasets.values()):
        print("Error: Not all required validation datasets could be loaded. Please check paths in config.yaml.")
        # Continue with warning, or exit if validation is critical
        pass # Allow to continue with warning if val datasets are critical

    print("Datasets loaded successfully.")

    # 5. Initialize Model
    print("\nInitializing model...")
    model = LightweightMultimodalModel(config).to(device)
    print(f"Model initialized: {model.__class__.__name__}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- NEW: Compile model for speedup if PyTorch 2.0+ ---
    try:
        if hasattr(torch, 'compile'): # Check for PyTorch 2.0+ API
            model = torch.compile(model)
            print("Model compiled with torch.compile() for potential speedup.")
        else:
            print("torch.compile() not available (requires PyTorch 2.0+). Skipping compilation.")
    except Exception as e:
        print(f"Warning: Could not compile model with torch.compile(): {e}. Proceeding without compilation.")
    # --- END NEW ---

    # 6. Initialize and Run Trainer
    print("\nInitializing trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        config=config_dict, # Pass dict config to trainer
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        tokenizer=dataset_loader.tokenizer # Pass the tokenizer for captioning loss
    )
    
    print("\nStarting training...")
    trainer.train()
    print("Training complete.")

    # 7. Post-training Evaluation (Optional, can be integrated into trainer or ablation runner)
    print("\nRunning final evaluations (if not covered by in-training eval)...")
    final_model_path = os.path.join(config.final_model_dir, "model_final.pth") # Use config.final_model_dir
    
    # Check if the final model checkpoint exists
    if os.path.exists(final_model_path):
        # Dynamically import EfficiencyEvaluator here to avoid circular dependencies if it wasn't used earlier
        from evaluation.efficiency_evaluator import EfficiencyEvaluator 
        
        # Load the entire state to correctly restore model (and potentially optimizer, etc. if needed later)
        # For evaluation, only model_state_dict is typically strictly necessary
        final_checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
        model.load_state_dict(final_checkpoint['model_state_dict']) # Load only model_state_dict
        print(f"Loaded final model for post-training evaluation from {final_model_path}")
        
        # Example: Run final efficiency check on the final model
        final_efficiency_evaluator = EfficiencyEvaluator(
            model=model,
            dataset=val_datasets['coco_val'], # Use a val dataset for sample input
            config=config, # Pass SimpleNamespace config
            device=device,
            save_dir=os.path.join(config.results_dir, "final_evaluation", "efficiency")
        )
        final_efficiency_evaluator.evaluate()
    else:
        print("Final model checkpoint not found for post-training evaluation.")

    # 8. Ablation Studies
    # The `AblationRunner` is typically run as a separate command to orchestrate multiple training runs.
    # You would run: `python evaluation/ablation_runner.py` directly.
    
    print("\nProject pipeline execution finished.")


if __name__ == "__main__":
    main()

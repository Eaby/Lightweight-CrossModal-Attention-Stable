import os
import torch
import yaml
from types import SimpleNamespace
from data_loader import CrossModalDatasetLoader
from models.multimodal_model import LightweightMultimodalModel

def strip_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """Strip a prefix from all keys in state_dict."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # Convert dict to SimpleNamespace for dot access
    return SimpleNamespace(**config_dict)

def main():
    config_path = "./configs/config.yaml"
    config = load_config(config_path)

    device = torch.device(config.preferred_device if config.preferred_device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load validation dataset only (evaluation)
    dataset_loader = CrossModalDatasetLoader(config)
    val_dataset = dataset_loader.load_coco(split="val")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Instantiate model
    model = LightweightMultimodalModel(config).to(device)

    # Load checkpoint with prefix fix
    checkpoint_path = os.path.join(config.final_model_dir, "model_final.pth")
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Fix keys by stripping _orig_mod. prefix
    fixed_state_dict = strip_prefix_from_state_dict(checkpoint['model_state_dict'], prefix="_orig_mod.")
    model.load_state_dict(fixed_state_dict)
    print("Model loaded successfully.")

    model.eval()

    # TODO: Implement your evaluation/inference code here using val_dataset and model
    # For example, generate captions or VQA answers on val_dataset samples and calculate metrics.

    # Example (very basic inference loop on a few samples):
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Forward pass - example for captioning with greedy decoding
            captions = model(batch, task='captioning', generate_caption=True)
            print(f"Sample {i} caption: {captions[0]}")
            if i >= 4:  # Just demo first 5 samples
                break

if __name__ == "__main__":
    main()


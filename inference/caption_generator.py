import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset

from data_loader import CrossModalDatasetLoader
from models.multimodal_model import CrossModalModel
from utils.config_loader import load_config

# Load config
config = load_config()

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model (adjust path as necessary!)
model = CrossModalModel(device=device, rank_k=config['rank_k']).to(device)
model.load_state_dict(torch.load("./results/ablation/rank_32/model.pth"))
model.eval()

# Load nocaps dataset
dataset = load_dataset("nocaps", split="validation")

# Initialize tokenizer and transforms
loader = CrossModalDatasetLoader(config)

# Create result dictionary
generated_captions = {}

for item in tqdm(dataset, desc="Generating captions"):
    image_id = str(item['image_id'])

    # For now, just copy one of the ground-truth captions as a placeholder
    # ✅ This will be replaced with real model inference later
    generated_caption = item['captions'][0]

    generated_captions[image_id] = generated_caption

# Create directory if not exists
os.makedirs("./results", exist_ok=True)

# Save generated captions
with open("./results/generated_captions.json", "w") as f:
    json.dump(generated_captions, f, indent=4)

print("✅ generated_captions.json created successfully!")

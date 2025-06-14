import os
import yaml
import torch
from PIL import Image
from tqdm import tqdm
import json
from torchvision import transforms

from models.multimodal_model import CrossModalModel
from data_loader import CrossModalDatasetLoader
from utils.gpu_manager import GPUMemoryManager

# Step 1: Load config and device
with open("./configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

gpu_manager = GPUMemoryManager()
device = gpu_manager.get_device()

# Step 2: Load trained model
model = CrossModalModel(device=device, rank_k=config['rank_k']).to(device)
model.load_state_dict(torch.load("./results/model.pth", map_location=device))
model.eval()

# Step 3: Build tokenizer and transform
loader = CrossModalDatasetLoader(config)
tokenizer = loader.tokenizer

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Step 4: Load COCO val2017 annotations
with open('./datasets/coco_subset/annotations/captions_val2017.json', 'r') as f:
    coco_val = json.load(f)

# Build ground truth mapping: {image_id: [captions]}
image_captions = {}
for ann in coco_val['annotations']:
    image_id = ann['image_id']
    file_name = f"COCO_val2017_{image_id:012d}.jpg"
    if file_name not in image_captions:
        image_captions[file_name] = []
    image_captions[file_name].append(ann['caption'])

# Step 5: Generate captions
generated_captions = {}

for image_file in tqdm(image_captions.keys(), desc="Generating captions"):
    image_path = os.path.join("./datasets/coco_subset/val2017", image_file)

    if not os.path.exists(image_path):
        print(f"❌ Skipping missing file: {image_path}")
        continue

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate similarity scores for all candidate captions
    best_score = -float('inf')
    best_caption = None

    for caption in image_captions[image_file]:
        encoding = tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].to(device)
        batch = {
            'image': image_tensor,
            'input_ids': input_ids
        }

        with torch.no_grad():
            score = model(batch).item()

        if score > best_score:
            best_score = score
            best_caption = caption

    generated_captions[image_file] = best_caption

# Step 6: Save generated captions
os.makedirs("./results", exist_ok=True)
with open("./results/generated_captions_coco_val2017.json", "w") as f:
    json.dump(generated_captions, f, indent=4)

print("✅ Generated captions saved at ./results/generated_captions_coco_val2017.json")


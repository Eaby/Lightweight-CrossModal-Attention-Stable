import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer 
from types import SimpleNamespace 

from data_loader import CrossModalDatasetLoader
from models.multimodal_model import LightweightMultimodalModel

class CaptionGenerator:
    def __init__(self, model, dataset, device, config):
        self.model = model.to(device)
        self.dataset = dataset 
        self.device = device
        self.config = config 
        
        # We need a DataLoader that provides UNIQUE images for generation
        # Your `self.dataset` is likely configured to yield one caption-image pair per annotation.
        # To get unique images, we'll iterate differently or modify the dataset for this generator.
        
        # A simple way for generation: get unique image_ids from the dataset
        # and create a temporary "unique image" dataset or DataLoader.
        
        # Let's create a temporary DataLoader that only yields unique images.
        # This requires `data_loader.py` to have a way to filter for unique images.
        # For simplicity, we can extract unique image IDs from the dataset annotations
        # and create a new temporary dataset or loader.
        
        # For evaluation, we only need one prediction per image.
        # The COCO val dataset has 25014 unique images.
        # Your `self.dataset` (coco_val) is `BaseDataset` which yields one `annotation`.
        # So, if image 123 has 5 captions, it appears 5 times in self.dataset.
        # We need to process each UNIQUE image only once.

        # Retrieve all annotations from the dataset.
        # This is not ideal as it loads all annotations into memory, but simpler for demo.
        # For large datasets, you might define a specific `UniqueImageDataset` in `data_loader.py`.
        
        # For now, let's filter the input `dataset` to have unique images.
        # Create a mapping from image_id to a single representative sample from the dataset.
        unique_image_samples = {}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx] # Get sample from BaseDataset
            img_id = sample['image_id']
            if img_id not in unique_image_samples:
                # Store the first occurrence of each unique image
                unique_image_samples[img_id] = sample
        
        # Create a list of these unique samples, and then a DataLoader for them.
        self.unique_image_list = list(unique_image_samples.values())

        self.dataloader = DataLoader(
            self.unique_image_list, # Iterate over unique images only
            batch_size=self.config.eval_batch_size, 
            shuffle=False, 
            num_workers=self.config.eval_num_workers,
            pin_memory=True
        )
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_caption_length = self.config.max_text_length 

    # Modified generate_captions to accept an optional `base_output_dir`
    def generate_captions(self, output_filename="generated_captions.json", base_output_dir=None):
        self.model.eval() 
        generated_captions_list = []

        with torch.no_grad():
            # Iterate through the DataLoader which now yields unique image samples
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Generating captions")):
                # The batch now contains unique images and their corresponding metadata (image_id).
                # This ensures we generate one caption per unique image.
                
                # `batch['image_id']` will have the image IDs for the current batch
                decoded_captions = self.model(
                    batch, # Pass the entire batch of unique images
                    task='captioning', 
                    generate_caption=True, 
                    max_caption_length=self.max_caption_length
                )
                
                image_ids = batch['image_id'].tolist() 

                processed_captions = []
                for cap in decoded_captions:
                    if not cap.strip(): 
                        processed_captions.append("a photo of an object") 
                    else:
                        processed_captions.append(cap)

                for i, caption in enumerate(processed_captions):
                    generated_captions_list.append({
                        "image_id": image_ids[i], # Each image_id should now be unique in this list
                        "caption": caption
                    })

        # Determine the final save path
        if base_output_dir: 
            save_path = os.path.join(base_output_dir, output_filename)
        else: 
            save_path = os.path.join(self.config.results_dir, "captioning", output_filename)
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 

        with open(save_path, "w") as f:
            json.dump(generated_captions_list, f, indent=4)

        print(f"✅ Generated captions saved to {save_path}!")
        return save_path 

# Main block for testing `CaptionGenerator` in isolation.
if __name__ == '__main__':
    from utils.config_loader import load_config
    
    config_dict = load_config()
    config = SimpleNamespace(**config_dict) # Convert dict to SimpleNamespace for this main block testing

    device = torch.device(config.preferred_device if config.preferred_device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_loader_instance = CrossModalDatasetLoader(config_dict) # Pass dict to data_loader
    generation_dataset = data_loader_instance.load_coco(split="val") 
    
    model_path = "./results/final_model/model_final.pth" 
    model = LightweightMultimodalModel(config).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Model will use random weights.")

    generator = CaptionGenerator(model, generation_dataset, device, config) # Pass SimpleNamespace config
    generator.generate_captions(output_filename="test_generated_captions.json")

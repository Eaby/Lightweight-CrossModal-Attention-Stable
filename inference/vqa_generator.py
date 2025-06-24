# In inference/vqa_generator.py

import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np 
from types import SimpleNamespace 

# --- IMPORTANT: Placeholder for VQA Answer Vocabulary Mapping ---
DUMMY_IDX_TO_ANSWER = {str(i): f"dummy_ans_{i}" for i in range(10)}

class VQAGenerator:
    def __init__(self, model, dataset, device, config, idx_to_answer_map=None):
        self.model = model.to(device)
        self.dataset = dataset 
        self.device = device
        self.config = config 

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.eval_batch_size, 
            shuffle=False, 
            num_workers=self.config.eval_num_workers,
            pin_memory=True
        )
        
        if idx_to_answer_map:
            self.idx_to_answer = idx_to_answer_map
            print(f"VQAGenerator loaded vocabulary of size: {len(self.idx_to_answer)}")
        else:
            print("WARNING: VQAGenerator initialized without a pre-built answer vocabulary. "
                  "Using a small internal dummy vocabulary for isolated testing. Predictions will be inaccurate.")
            self.idx_to_answer = {str(i): f"dummy_ans_{i}" for i in range(self.config.num_answers)} 

        if len(self.idx_to_answer) < self.config.num_answers:
             print(f"WARNING: VQA `idx_to_answer` vocab size ({len(self.idx_to_answer)}) "
                   f"is smaller than config `num_answers` ({self.config.num_answers}). "
                   f"Predictions for indices >= {len(self.idx_to_answer)} will be 'unknown'. "
                   f"Please build a proper VQA vocabulary matching `num_answers`.")


    # Modified generate_answers to accept an optional `base_output_dir`
    def generate_answers(self, output_filename="vqa_predictions.json", base_output_dir=None):
        self.model.eval() 
        generated_vqa_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Generating VQA answers")):
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                question_ids = batch['question_id'].tolist() 

                vqa_logits = self.model(
                    {'image': images, 'input_ids': input_ids, 'attention_mask': attention_mask},
                    task='vqa'
                ) 

                predicted_answer_indices = torch.argmax(vqa_logits, dim=-1).cpu().numpy()

                for i, qid in enumerate(question_ids):
                    pred_idx = predicted_answer_indices[i]
                    predicted_answer = self.idx_to_answer.get(str(pred_idx), "unknown") 
                    
                    generated_vqa_predictions.append({
                        "question_id": qid,
                        "answer": predicted_answer
                    })

        # Determine the final save path
        if base_output_dir: # If a specific output directory is provided (e.g., from trainer)
            save_path = os.path.join(base_output_dir, output_filename)
        else: # Fallback to default (config.results_dir) for standalone runs
            save_path = os.path.join(self.config.results_dir, "vqa", output_filename)
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(generated_vqa_predictions, f, indent=4)

        print(f"✅ VQA Predictions saved to {save_path}!")
        return save_path 

# Main block needs to pass the config object correctly as SimpleNamespace
if __name__ == '__main__':
    from utils.config_loader import load_config
    from types import SimpleNamespace 
    
    config_dict = load_config()
    config = SimpleNamespace(**config_dict) # Convert dict to SimpleNamespace for this main block

    device = torch.device(config.preferred_device if config.preferred_device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_loader_instance = CrossModalDatasetLoader(config_dict) # Pass dict to data_loader
    generation_dataset = data_loader_instance.load_vqa(split="val") 
    
    model_path = "./results/final_model/model_final.pth" 
    model = LightweightMultimodalModel(config).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Model will use random weights.")

    generator = VQAGenerator(model, generation_dataset, device, config) # Pass SimpleNamespace config
    generator.generate_answers(output_filename="test_vqa_predictions.json") # Example call

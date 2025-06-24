import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import json # For handling COCO/Flickr structure if needed

class RetrievalEvaluator:
    def __init__(self, model, dataset, device, save_dir, config):
        self.model = model.to(device) # Ensure model is on the correct device
        self.dataset = dataset # This is your BaseDataset instance
        self.device = device
        self.save_dir = save_dir
        self.config = config # config is expected to be a SimpleNamespace here

        os.makedirs(save_dir, exist_ok=True)

        self.dataloader = DataLoader(
            # Access config parameters directly from SimpleNamespace
            self.dataset,
            batch_size=self.config.eval_batch_size, # Access directly from SimpleNamespace
            shuffle=False, # Do not shuffle for evaluation
            num_workers=self.config.eval_num_workers, # Access directly from SimpleNamespace
            pin_memory=True
        )

    def encode_features(self):
        self.model.eval() # Set model to evaluation mode
        image_embeddings = []
        text_embeddings = []
        
        unique_image_features_map = {} # {image_id: feature_tensor}
        all_caption_features_list = []
        all_caption_image_ids_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Extracting features")):
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # image_ids will be in batch if you updated data_loader.py
                # Assuming 'image_id' is directly present and is an int or list of ints.
                # If BaseDataset returns it, DataLoader will batch it into a tensor.
                # Use .tolist() to convert tensor to Python list
                batch_image_ids = batch['image_id'].tolist() 
                
                # Get the aligned features from your model for retrieval
                img_aligned_feats, text_aligned_feats, _ = self.model(
                    {'image': images, 'input_ids': input_ids, 'attention_mask': attention_mask}, 
                    task='retrieval'
                ) # img_aligned_feats: [B, D], text_aligned_feats: [B, D]

                # Store unique image features
                for i, img_id in enumerate(batch_image_ids):
                    if img_id not in unique_image_features_map:
                        unique_image_features_map[img_id] = img_aligned_feats[i].cpu().numpy()
                
                # Store all caption features and their corresponding image IDs
                all_caption_features_list.extend(text_aligned_feats.cpu().numpy())
                all_caption_image_ids_list.extend(batch_image_ids) # image_id for each caption

        # Convert maps/lists to numpy arrays
        sorted_image_ids = sorted(unique_image_features_map.keys())
        unique_image_embeddings = np.array([unique_image_features_map[img_id] for img_id in sorted_image_ids])
        
        all_caption_embeddings = np.array(all_caption_features_list)
        all_caption_image_ids = np.array(all_caption_image_ids_list)

        # Map image IDs to their index in `unique_image_embeddings`
        self.image_id_to_idx = {img_id: idx for idx, img_id in enumerate(sorted_image_ids)}

        # Save embeddings for later analysis
        np.save(os.path.join(self.save_dir, "unique_image_embeddings.npy"), unique_image_embeddings)
        np.save(os.path.join(self.save_dir, "all_caption_embeddings.npy"), all_caption_embeddings)
        np.save(os.path.join(self.save_dir, "all_caption_image_ids.npy"), all_caption_image_ids)

        return unique_image_embeddings, all_caption_embeddings, all_caption_image_ids


    def compute_metrics(self, unique_image_embeddings, all_caption_embeddings, all_caption_image_ids, k_values=[1, 5, 10]):
        # Calculate similarity matrix: (Num_unique_images, Num_captions)
        similarity_matrix = np.matmul(unique_image_embeddings, all_caption_embeddings.T) # [N_img_unique, N_cap_all]

        num_unique_images = unique_image_embeddings.shape[0]
        num_captions = all_caption_embeddings.shape[0]

        # --- Image-to-Text (I2T) Retrieval ---
        i2t_ranks = [] 
        i2t_recalls = {k: 0 for k in k_values}
        
        for img_idx, img_id in enumerate(tqdm(self.image_id_to_idx.keys(), desc="I2T Retrieval")):
            ground_truth_caption_indices = np.where(all_caption_image_ids == img_id)[0]
            sorted_caption_indices = np.argsort(similarity_matrix[img_idx])[::-1] # Descending order

            rank = num_captions 
            for true_cap_idx in ground_truth_caption_indices:
                current_rank = np.where(sorted_caption_indices == true_cap_idx)[0][0] + 1
                if current_rank < rank:
                    rank = current_rank 
            i2t_ranks.append(rank)

            for k in k_values:
                if any(cap_idx in sorted_caption_indices[:k] for cap_idx in ground_truth_caption_indices):
                    i2t_recalls[k] += 1
        
        i2t_medr = np.median(i2t_ranks)
        i2t_meanr = np.mean(i2t_ranks)
        i2t_results = {f"R@{k}": i2t_recalls[k] / num_unique_images for k in k_values}
        i2t_results["MedR"] = i2t_medr
        i2t_results["MeanR"] = i2t_meanr


        # --- Text-to-Image (T2I) Retrieval ---
        reverse_similarity_matrix = similarity_matrix.T # [N_cap_all, N_img_unique]

        t2i_ranks = [] 
        t2i_recalls = {k: 0 for k in k_values}
        
        for cap_idx, true_img_id in enumerate(tqdm(all_caption_image_ids, desc="T2I Retrieval")):
            true_img_idx = self.image_id_to_idx[true_img_id]

            sorted_image_indices = np.argsort(reverse_similarity_matrix[cap_idx])[::-1] # Descending order

            rank = np.where(sorted_image_indices == true_img_idx)[0][0] + 1
            t2i_ranks.append(rank)

            for k in k_values:
                if true_img_idx in sorted_image_indices[:k]:
                    t2i_recalls[k] += 1
        
        t2i_medr = np.median(t2i_ranks)
        t2i_meanr = np.mean(t2i_ranks)
        t2i_results = {f"R@{k}": t2i_recalls[k] / num_captions for k in k_values}
        t2i_results["MedR"] = t2i_medr
        t2i_results["MeanR"] = t2i_meanr

        # Calculate average metrics
        average_results = {}
        for k in k_values:
            average_results[f"R@{k}"] = (i2t_results[f"R@{k}"] + t2i_results[f"R@{k}"]) / 2
        average_results["MedR"] = (i2t_medr + t2i_medr) / 2
        average_results["MeanR"] = (i2t_meanr + t2i_meanr) / 2

        all_results = {
            "Image-to-Text": i2t_results,
            "Text-to-Image": t2i_results,
            "Average": average_results
        }
        
        # Save results to CSV (or JSON)
        df = pd.DataFrame([
            {"Metric": "I2T " + k, "Value": v} for k, v in i2t_results.items()
        ] + [
            {"Metric": "T2I " + k, "Value": v} for k, v in t2i_results.items()
        ] + [
            {"Metric": "Average " + k, "Value": v} for k, v in average_results.items()
        ])
        df.to_csv(os.path.join(self.save_dir, "retrieval_results.csv"), index=False)
        
        return all_results

    def evaluate(self):
        print(f"Starting retrieval evaluation for {len(self.dataset)} samples...")
        unique_image_embeddings, all_caption_embeddings, all_caption_image_ids = self.encode_features()
        results = self.compute_metrics(unique_image_embeddings, all_caption_embeddings, all_caption_image_ids)
        
        print("\n--- Retrieval Results ---")
        for direction, metrics in results.items():
            print(f"  {direction}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        print("-------------------------\n")
        
        return results

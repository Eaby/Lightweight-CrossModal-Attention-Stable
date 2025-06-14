import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score

class RetrievalEvaluator:
    def __init__(self, model, dataset, device, save_dir):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

    def encode_dataset(self):
        image_embeddings = []
        text_embeddings = []

        for sample in tqdm(self.dataset, desc="Encoding dataset"):
            batch = {
                'image': sample['image'].unsqueeze(0).to(self.device),
                'input_ids': sample['input_ids'].unsqueeze(0).to(self.device)
            }
            with torch.no_grad():
                image_feat = self.model.clip_encoder.encode_image(batch['image'])
                text_feat = self.model.clip_encoder.encode_text(batch['input_ids'])
            image_embeddings.append(image_feat.cpu().numpy())
            text_embeddings.append(text_feat.cpu().numpy())

        image_embeddings = np.vstack(image_embeddings)
        text_embeddings = np.vstack(text_embeddings)
        np.save(os.path.join(self.save_dir, "image_embeddings.npy"), image_embeddings)
        np.save(os.path.join(self.save_dir, "text_embeddings.npy"), text_embeddings)
        return image_embeddings, text_embeddings

    def compute_recall_at_k(self, image_embeddings, text_embeddings, k_values=[1, 5, 10]):
        sims = np.matmul(image_embeddings, text_embeddings.T)
        recalls = {k: 0 for k in k_values}
        for i in range(len(image_embeddings)):
            sorted_idx = np.argsort(-sims[i])
            for k in k_values:
                if i in sorted_idx[:k]:
                    recalls[k] += 1
        results = {f"Recall@{k}": recalls[k] / len(image_embeddings) for k in k_values}
        pd.DataFrame([results]).to_csv(os.path.join(self.save_dir, "retrieval_results.csv"), index=False)
        return results

    def evaluate(self):
        image_embeddings, text_embeddings = self.encode_dataset()
        results = self.compute_recall_at_k(image_embeddings, text_embeddings)
        print(results)


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import os  # ✅ added for saving

class Trainer:
    def __init__(self, model, dataset, config, device):
        self.model = model
        self.config = config
        self.device = device

        self.dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'])
        self.scaler = GradScaler()

    def train(self):
        self.model.train()

        for epoch in range(self.config['epochs']):
            epoch_loss = 0.0

            progress_bar = tqdm(self.dataloader, desc=f"Epoch [{epoch+1}/{self.config['epochs']}]", leave=False)

            for batch in progress_bar:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)

                batch_size = images.size(0)

                # Build positive pairs
                batch_data_pos = {
                    'image': images,
                    'input_ids': input_ids
                }
                labels_pos = torch.ones(batch_size, device=self.device)

                # Build negative pairs
                shuffled_indices = torch.randperm(batch_size)
                batch_data_neg = {
                    'image': images,
                    'input_ids': input_ids[shuffled_indices]
                }
                labels_neg = torch.zeros(batch_size, device=self.device)

                self.optimizer.zero_grad()

                with autocast():
                    scores_pos = self.model(batch_data_pos)
                    scores_neg = self.model(batch_data_neg)

                    all_scores = torch.cat([scores_pos, scores_neg], dim=0)
                    all_labels = torch.cat([labels_pos, labels_neg], dim=0)

                    loss = self.criterion(all_scores, all_labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            print(f"Epoch [{epoch+1}] finished. Average Loss: {epoch_loss / len(self.dataloader):.6f}")

        # ✅ SAVE MODEL AFTER FINAL EPOCH:
        save_path = "./results/ablation/rank_32/"
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))
        print("✅ Model successfully saved after training.")


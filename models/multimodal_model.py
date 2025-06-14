import torch
import torch.nn as nn
from models.baseline_clip import CLIPEncoder
from models.rank_attention import RankBasedAttention

class CrossModalModel(nn.Module):
    def __init__(self, device, rank_k=32):
        super().__init__()
        self.device = device
        self.clip_encoder = CLIPEncoder(device=device)
        self.rank_attention = RankBasedAttention(input_dim=512, rank_k=rank_k, device=self.device)

    def forward(self, batch):
        images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)

        image_feats = self.clip_encoder.encode_image(images)
        text_feats = self.clip_encoder.encode_text(input_ids)

        scores = self.rank_attention(image_feats, text_feats)
        return scores


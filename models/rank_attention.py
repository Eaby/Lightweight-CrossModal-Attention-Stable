import torch
import torch.nn as nn

class RankBasedAttention(nn.Module):
    def __init__(self, input_dim, rank_k, device="cuda"):
        super().__init__()
        self.rank_k = rank_k

        # We always initialize weights in float32
        self.U = nn.Linear(input_dim, rank_k, bias=False, dtype=torch.float32, device=device)
        self.V = nn.Linear(input_dim, rank_k, bias=False, dtype=torch.float32, device=device)
        self.output = nn.Linear(rank_k, 1, dtype=torch.float32, device=device)

    def forward(self, image_feats, text_feats):
        # Cast input features to match layer dtype (float32)
        image_feats = image_feats.to(self.U.weight.dtype)
        text_feats = text_feats.to(self.V.weight.dtype)

        U_proj = self.U(image_feats)
        V_proj = self.V(text_feats)
        interaction = U_proj * V_proj
        logits = self.output(interaction).squeeze(1)
        return logits


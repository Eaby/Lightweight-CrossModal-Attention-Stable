import torch
import torch.nn as nn
import torch.nn.functional as F

class RankAttentionModel(nn.Module): # Renamed to match plan
    def __init__(self, vision_dim, text_dim, rank, temperature=0.07, device="cuda"): # Aligned with plan's __init__
        super().__init__()
        self.rank = rank
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.device = device # Storing device

        # Low-rank factorization matrices for Vision to Text attention
        # Q_v = V_U * V_V^T
        self.vision_proj_q = nn.Linear(vision_dim, rank, bias=False, dtype=torch.float32).to(device) # Represents U for Queries
        self.vision_proj_k = nn.Linear(vision_dim, rank, bias=False, dtype=torch.float32).to(device) # Represents U for Keys
        self.vision_proj_v = nn.Linear(vision_dim, rank, bias=False, dtype=torch.float32).to(device) # Represents U for Values

        # Low-rank factorization matrices for Text to Vision attention (or just for general cross-attention logic)
        self.text_proj_q = nn.Linear(text_dim, rank, bias=False, dtype=torch.float32).to(device)
        self.text_proj_k = nn.Linear(text_dim, rank, bias=False, dtype=torch.float32).to(device)
        self.text_proj_v = nn.Linear(text_dim, rank, bias=False, dtype=torch.float32).to(device)

        # Output projection for aligned features (mapping from rank_k back to original dimension or combined)
        # This part of the plan was a bit ambiguous ("vision_proj_v = nn.Linear(rank, text_dim)").
        # A common way is to project the low-rank attention output back to the original feature dimension
        # or a shared embedding space. Let's create output layers for aligned features.
        self.v2t_output_proj = nn.Linear(rank, text_dim, bias=False, dtype=torch.float32).to(device) # Project rank_k to text_dim for v2t aligned features
        self.t2v_output_proj = nn.Linear(rank, vision_dim, bias=False, dtype=torch.float32).to(device) # Project rank_k to vision_dim for t2v aligned features

        # Learnable temperature parameter as per your plan
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32, device=device))


    def forward(self, vision_features, text_features):
        # vision_features: [B, N_v, vision_dim] (e.g., [B, 1, 768] if using CLS token)
        # text_features:   [B, N_t, text_dim] (e.g., [B, 1, 768] if using CLS token)

        # Ensure inputs are float32
        vision_features = vision_features.to(torch.float32)
        text_features = text_features.to(torch.float32)

        # === Vision to Text Attention ===
        # Here we approximate the Query for Vision and Key/Value for Text
        # The plan's notation was A ≈ UV^T for attention matrix.
        # This implies:
        # Q_v = vision_features @ W_Q_v (W_Q_v is (vision_dim, rank))
        # K_t = text_features @ W_K_t (W_K_t is (text_dim, rank))
        # Attention_Scores = Q_v @ K_t.T
        # The plan mentioned vision_proj_u and vision_proj_v for *vision* features (V_U, V_V^T).
        # Let's interpret it as: Vision features (Queries) are projected to 'rank', and Text features (Keys) are also projected to 'rank'.

        # Project vision features to Queries for cross-attention (V -> T)
        # v_query = self.vision_proj_u(vision_features) # [B, N_v, rank]
        # Project text features to Keys for cross-attention (V -> T)
        # t_key = self.text_proj_u(text_features) # [B, N_t, rank]

        # Let's refine based on typical attention structure with low-rank projections:
        # Instead of full attention (Q @ K^T), we aim for a low-rank product of Q and K matrices.
        # Original plan: A ≈ UV^T (U in R^(n x k), V in R^(m x k))
        # This implies: A_ij = sum_p U_ip * V_jp
        # If A is Q_i K_j^T, then Q_i = U_i, K_j = V_j.
        # So we project vision_features to U, and text_features to V.

        # Query from Vision, Key from Text
        v_query_proj = self.vision_proj_q(vision_features) # [B, N_v, rank]
        t_key_proj = self.text_proj_k(text_features)       # [B, N_t, rank]

        # Low-rank attention scores (similar to bilinear product in rank space)
        # We need N_v x N_t attention map.
        # This requires broadcasting or explicit multiplication.
        # [B, N_v, rank] x [B, N_t, rank] -> (incorrect matmul)
        # [B, N_v, rank] @ [B, rank, N_t] (t_key_proj.transpose(-2, -1)) is the correct dot product
        attention_logits = torch.matmul(v_query_proj, t_key_proj.transpose(-2, -1)) # [B, N_v, N_t]
        
        # Scale by temperature (crucial for InfoNCE)
        attention_logits = attention_logits / self.temperature.clamp(min=1e-8) # Clamp to avoid division by zero

        # Apply softmax to get attention weights if performing weighted sum for values
        # For cross-attention, it's often softmax over the target sequence (text tokens for V->T)
        attention_weights = F.softmax(attention_logits, dim=-1) # Softmax over N_t (text tokens)


        # Compute aligned features using text values
        t_value_proj = self.text_proj_v(text_features) # [B, N_t, rank]

        # v2t_aligned_features are vision features attended to text values
        # If we take `vision_features` as the base and transform it based on text features
        # Or more commonly, compute a context vector based on `text_features` weighted by attention_weights
        # and then transform it to be "vision-like" or fused.
        
        # Let's align features: Vision attending to Text Values
        # (Attention_Weights @ Text_Values_Projected)
        # [B, N_v, N_t] @ [B, N_t, rank] -> [B, N_v, rank]
        v2t_context = torch.matmul(attention_weights, t_value_proj) # This is the context from text for each vision query
        v2t_aligned_features = self.v2t_output_proj(v2t_context) # [B, N_v, text_dim] -> project back to text_dim for vision-aligned-to-text

        # === Text to Vision Attention === (Reverse direction, if needed by downstream tasks)
        # Query from Text, Key from Vision
        t_query_proj = self.text_proj_q(text_features)   # [B, N_t, rank]
        v_key_proj = self.vision_proj_k(vision_features) # [B, N_v, rank]

        # [B, N_t, rank] @ [B, rank, N_v] -> [B, N_t, N_v]
        reverse_attention_logits = torch.matmul(t_query_proj, v_key_proj.transpose(-2, -1))
        reverse_attention_logits = reverse_attention_logits / self.temperature.clamp(min=1e-8)
        
        reverse_attention_weights = F.softmax(reverse_attention_logits, dim=-1) # Softmax over N_v (vision tokens)

        # Compute aligned features using vision values
        v_value_proj = self.vision_proj_v(vision_features) # [B, N_v, rank]
        
        # t2v_aligned_features are text features attended to vision values
        # [B, N_t, N_v] @ [B, N_v, rank] -> [B, N_t, rank]
        t2v_context = torch.matmul(reverse_attention_weights, v_value_proj)
        t2v_aligned_features = self.t2v_output_proj(t2v_context) # [B, N_t, vision_dim] -> project back to vision_dim for text-aligned-to-vision

        return attention_logits, v2t_aligned_features, t2v_aligned_features

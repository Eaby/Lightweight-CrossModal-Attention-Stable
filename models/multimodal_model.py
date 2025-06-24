import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel 
from models.rank_attention import RankAttentionModel  # Fixed import here
from transformers import BertTokenizer
import torch.nn.functional as F

class LightweightMultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(
            self.config.preferred_device if getattr(self.config, 'preferred_device', None) and torch.cuda.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Vision encoder
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224').to(self.device)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_dim = self.vision_encoder.config.hidden_size  # typically 768

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_dim = self.text_encoder.config.hidden_size  # typically 768

        if self.vision_dim != self.text_dim:
            print(f"Warning: vision/text dim mismatch: {self.vision_dim} vs {self.text_dim}")

        # Rank-based cross-modal attention (your RankAttentionModel)
        self.cross_modal_attention = RankAttentionModel(
            vision_dim=self.vision_dim,
            text_dim=self.text_dim,
            rank=self.config.rank_k,
            temperature=self.config.attention_temperature,
            device=self.device
        ).to(self.device)

        # Retrieval projection head
        self.retrieval_projection = nn.Linear(self.vision_dim, self.config.input_dim).to(self.device)

        # Load tokenizer to get vocabulary size & token ids
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.config.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id

        # Transformer decoder for captioning
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.vision_dim,
            nhead=self.config.decoder_nheads,
            dim_feedforward=self.config.decoder_dim_feedforward,
            dropout=self.config.decoder_dropout,
            batch_first=True
        )
        self.captioning_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.decoder_num_layers
        ).to(self.device)
        self.captioning_lm_head = nn.Linear(self.vision_dim, self.vocab_size).to(self.device)

        # VQA head
        self.vqa_head = nn.Linear(self.vision_dim, self.config.num_answers).to(self.device)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, batch, task='retrieval', generate_caption=False, max_caption_length=50): 
        images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        vision_features = self.vision_encoder(images).last_hidden_state[:, 0, :]
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]

        attention_logits, v2t_aligned, t2v_aligned = self.cross_modal_attention(
            vision_features.unsqueeze(1), text_features.unsqueeze(1)
        )
        v2t_aligned = v2t_aligned.squeeze(1)
        t2v_aligned = t2v_aligned.squeeze(1)

        if task == 'retrieval':
            return v2t_aligned, t2v_aligned, attention_logits

        elif task == 'captioning':
            if generate_caption:
                return self._generate_caption_greedy(v2t_aligned, max_caption_length)
            else:
                decoder_input_tokens = input_ids[:, :-1]
                decoder_embeddings = self.text_encoder.embeddings.word_embeddings(decoder_input_tokens)

                decoder_output = self.captioning_decoder(
                    tgt=decoder_embeddings,
                    memory=v2t_aligned.unsqueeze(1),
                    tgt_mask=self._generate_square_subsequent_mask(decoder_input_tokens.size(1))
                )

                caption_logits = self.captioning_lm_head(decoder_output)
                return caption_logits

        elif task == 'vqa':
            fused = v2t_aligned + t2v_aligned
            vqa_output = self.vqa_head(fused)
            return vqa_output

        else:
            raise ValueError(f"Unknown task: {task}")

    def _generate_caption_greedy(self, image_features, max_caption_length):
        batch_size = image_features.size(0)
        input_ids = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=self.device)
        generated_sequences = input_ids

        for _ in range(max_caption_length - 1):
            decoder_embeddings = self.text_encoder.embeddings.word_embeddings(input_ids)
            causal_mask = self._generate_square_subsequent_mask(input_ids.size(1))

            decoder_output = self.captioning_decoder(
                tgt=decoder_embeddings,
                memory=image_features.unsqueeze(1),
                tgt_mask=causal_mask
            )

            logits = self.captioning_lm_head(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
            generated_sequences = torch.cat([generated_sequences, next_token], dim=-1)
            input_ids = generated_sequences

            if (next_token == self.eos_token_id).all():
                break

        decoded_captions = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_sequences]
        return decoded_captions


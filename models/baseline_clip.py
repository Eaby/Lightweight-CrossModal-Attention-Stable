import torch
import clip

class CLIPEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_image(self, image):
        """
        image: torch tensor [B, 3, 224, 224] already batched!
        """
        image = image.to(self.device)
        image_features = self.model.encode_image(image)
        return image_features

    def encode_text(self, input_ids):
        """
        input_ids: tensor already tokenized, we need to properly handle text encoding.
        """
        # Temporary dummy mapping: CLIP requires its own tokenizer.
        text_list = ["dummy text"] * input_ids.shape[0]
        text_tokens = clip.tokenize(text_list).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        return text_features


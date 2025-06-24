import torch
import clip
import torch.nn as nn # Import nn for potential normalization of features

class CLIPEncoder(nn.Module): # Make it an nn.Module for consistency with other models
    def __init__(self, device='cuda'):
        super().__init__() # Call super init
        self.device = device
        # Ensure 'ViT-B/32' is available. If not, consider 'RN50' or checking available_models().
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Freeze CLIP parameters as it's a baseline encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image=None, raw_text=None): # Use forward method for nn.Module
        image_features = None
        text_features = None

        if image is not None:
            image = image.to(self.device)
            image_features = self.model.encode_image(image).float() # Ensure float type
            # CLIP features are often L2 normalized
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)


        if raw_text is not None:
            # CLIP's own tokenizer needs raw text strings.
            # `context_length` is usually 77 for ViT-B/32
            text_tokens = clip.tokenize(raw_text, context_length=77).to(self.device) 
            text_features = self.model.encode_text(text_tokens).float() # Ensure float type
            # CLIP features are often L2 normalized
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return image_features, text_features # Return both, or only what's requested by caller


# Example of how you might use it (for reference, not part of the file itself)
# from data_loader import CrossModalDatasetLoader
# from torch.utils.data import DataLoader
# from utils.config_loader import ConfigLoader # Assuming you have this now

# if __name__ == '__main__':
#     # Dummy config for testing
#     class DummyConfig:
#         def __init__(self):
#             self.datasets_path = "./datasets"
#             self.coco_annotations = "coco_subset/annotations/captions_train2017.json"
#             self.train_image_dir = "coco_subset/train2017"
#             self.max_text_length = 77
#             self.vqa_train_image_dir = "coco_subset/train2014" # Needed for VQA loader
#             self.vqa_questions = "vqa2/v2_OpenEnded_mscoco_train2014_questions.json"
#             self.vqa_annotations = "vqa2/v2_mscoco_train2014_annotations.json"
#             self.flickr_csv = "flickr30k_images_andCaptions/flickr30k_captions.csv"
#             # Add any other config parameters that load_coco needs
#             self.coco_val_annotations = "coco_subset/annotations/captions_val2017.json"
#             self.val_image_dir = "coco_subset/val2017"


#     config_loader = DummyConfig() # Replace with actual config loader
#     dataset_loader = CrossModalDatasetLoader(config_loader)
#     coco_dataset = dataset_loader.load_coco(split="val") # Use val split for testing image prefix
#     dataloader = DataLoader(coco_dataset, batch_size=4)

#     clip_encoder = CLIPEncoder(device='cuda' if torch.cuda.is_available() else 'cpu')
#     clip_encoder.eval() # Set to eval mode

#     for i, batch in enumerate(dataloader):
#         images = batch['image']
#         raw_texts = batch['raw_text'] # Now raw_text is available!

#         image_features, text_features = clip_encoder(image=images, raw_text=raw_texts)
#         print(f"Batch {i}: Image features shape: {image_features.shape}, Text features shape: {text_features.shape}")
#         if i == 0:
#             break

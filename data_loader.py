import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from transformers import AutoTokenizer

# ✅ Base Dataset Class
class BaseDataset(Dataset):
    def __init__(self, image_dir, annotations, transform, tokenizer, task="captioning"):
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        image_path = os.path.join(self.image_dir, sample['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.task == "captioning":
            text = sample['caption']
        elif self.task == "vqa":
            text = sample['question']
            answer = sample.get('answer', "unknown")
        else:
            text = sample['caption']

        encoding = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=50, return_tensors="pt"
        )

        output = {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

        if self.task == "vqa":
            output['answer'] = answer

        return output


# ✅ Dataset Loader Class
class CrossModalDatasetLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    # ✅ COCO Loader
    def load_coco(self, split="train"):
        if split == "train":
            json_path = os.path.join(self.config['datasets_path'], self.config['coco_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], "coco_subset/train2014")
            image_prefix = "COCO_train2014_"
        else:
            json_path = os.path.join(self.config['datasets_path'], self.config['coco_val_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], "coco_subset/val2014")
            image_prefix = "COCO_val2014_"

        with open(json_path, 'r') as f:
            data = json.load(f)

        annotations = []
        missing = 0

        for item in data['annotations']:
            filename = f"{image_prefix}{item['image_id']:012d}.jpg"
            full_path = os.path.join(image_dir, filename)

            if os.path.exists(full_path):
                annotations.append({
                    'file_name': filename,
                    'caption': item['caption']
                })
            else:
                missing += 1

        print(f"Loaded {len(annotations)} valid COCO samples, skipped {missing} missing images.")
        dataset = BaseDataset(image_dir, annotations, self.transform, self.tokenizer, task="captioning")
        return dataset

    # ✅ FULLY FIXED FLICKR30K LOADER
    def load_flickr30k(self):
        csv_path = os.path.join(self.config['datasets_path'], self.config['flickr_csv'])
        df = pd.read_csv(csv_path)

        print("✅ Flickr30K Columns:", df.columns)  # Debugging print

        annotations = []
        for _, row in df.iterrows():
            annotations.append({
                'file_name': row['image_file'],
                'caption': row['caption']
            })

        image_dir = os.path.join(self.config['datasets_path'], "flickr30k_images_andCaptions")
        dataset = BaseDataset(image_dir, annotations, self.transform, self.tokenizer, task="captioning")
        return dataset

    # ✅ VQAv2 Loader
    def load_vqa(self):
        ques_path = os.path.join(self.config['datasets_path'], self.config['vqa_questions'])
        ans_path = os.path.join(self.config['datasets_path'], self.config['vqa_annotations'])

        with open(ques_path, 'r') as f:
            questions = json.load(f)['questions']

        with open(ans_path, 'r') as f:
            answers = json.load(f)['annotations']

        answers_map = {item['question_id']: item['answers'][0]['answer'] for item in answers}

        annotations = []
        for item in questions:
            annotations.append({
                'file_name': f"COCO_train2014_{item['image_id']:012d}.jpg",
                'question': item['question'],
                'answer': answers_map.get(item['question_id'], "unknown")
            })

        image_dir = os.path.join(self.config['datasets_path'], "coco_subset/train2014")
        dataset = BaseDataset(image_dir, annotations, self.transform, self.tokenizer, task="vqa")
        return dataset

    # ✅ NoCaps Loader (not fully implemented, for later)
    def load_nocaps(self):
        print("✅ NoCaps loader placeholder — will fully implement later")
        return None

    # ✅ OK-VQA Loader (not fully implemented, for later)
    def load_okvqa(self):
        print("✅ OK-VQA loader placeholder — will fully implement later")
        return None


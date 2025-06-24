import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm 
from torch.utils.data.dataloader import default_collate 

# ADDED: Custom collate function for VQA data (ensure this is at top-level)
def vqa_collate_fn(batch):
    """
    Custom collate function for VQA data to ensure consistent batching of answer_texts.
    It separates out 'answer_texts' (list of lists of strings) before default_collate,
    then re-attaches them, as default_collate struggles with nested lists of varying lengths.
    """
    # Extract answer_texts separately as they are lists of strings and don't stack directly into a tensor
    answer_texts = [item['answer_texts'] for item in batch]
    
    # Temporarily remove 'answer_texts' from individual batch dictionaries for default_collate
    for item in batch:
        del item['answer_texts']

    # Use default_collate for the rest of the items (tensors, numericals, simple lists)
    collated_batch = default_collate(batch)
    
    # Add the extracted raw lists back to the collated batch
    collated_batch['answer_texts'] = answer_texts
    
    return collated_batch


# ✅ Base Dataset Class
class BaseDataset(Dataset):
    def __init__(self, image_dir, annotations, transform, tokenizer, task="captioning", max_text_length=77):
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform
        self.tokenizer = tokenizer
        self.task = task
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        image_path = os.path.join(self.image_dir, sample['file_name'])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Error: {e}. Returning a black image placeholder.")
            image = torch.zeros(3, 224, 224) 

        # Determine text and potential answers based on task
        text = "" 
        answer_texts = [] 
        
        if self.task == "captioning":
            text = sample['caption']
        elif self.task == "vqa":
            text = sample['question']
            answer_texts = sample.get('answers', []) 
        else: 
            text = sample['caption']

        encoding = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_text_length, return_tensors="pt"
        )

        output = {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'raw_text': text, 
            'image_id': sample['image_id'] if 'image_id' in sample else -1, 
        }

        # Add task-specific outputs
        if self.task == "vqa":
            output['answer_texts'] = answer_texts 
            output['question_id'] = sample['question_id'] 
        elif self.task == "captioning":
            output['caption'] = text 

        return output


# ✅ Dataset Loader Class
class CrossModalDatasetLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        self.max_text_length = config.get('max_text_length', 77) 


    # ✅ COCO Loader with filename fix for train vs val
    def load_coco(self, split="train"):
        if split == "train":
            json_path = os.path.join(self.config['datasets_path'], self.config['coco_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], self.config['train_image_dir'])
            expected_prefix = ""  
        else: 
            json_path = os.path.join(self.config['datasets_path'], self.config['coco_val_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], self.config['val_image_dir'])
            expected_prefix = "COCO_val2017_"  

        if not os.path.exists(json_path):
            print(f"Error: COCO annotations file not found at {json_path}. Skipping {split} dataset loading.")
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_id_to_canonical_filename = {img['id']: img['file_name'] for img in data['images']}

        annotations = []
        missing_images_count = 0
        
        for item in tqdm(data['annotations'], desc=f"Loading COCO {split} annotations"):
            image_id = item['image_id']
            canonical_filename_from_json = image_id_to_canonical_filename.get(image_id)
            
            if canonical_filename_from_json is None:
                print(f"Warning: Image ID {image_id} from annotation not found in 'images' section of {json_path}. Skipping annotation.")
                continue

            file_name_to_check_on_disk = canonical_filename_from_json
            if split == "val" and not canonical_filename_from_json.startswith(expected_prefix):
                file_name_to_check_on_disk = expected_prefix + canonical_filename_from_json

            full_image_path = os.path.join(image_dir, file_name_to_check_on_disk)

            if os.path.exists(full_image_path):
                annotations.append({
                    'file_name': file_name_to_check_on_disk, 
                    'caption': item['caption'],
                    'image_id': image_id, 
                    'annotation_id': item['id'] 
                })
            else:
                missing_images_count += 1

        print(f"Loaded {len(annotations)} valid COCO {split} samples, skipped {missing_images_count} missing images.")
        dataset = BaseDataset(image_dir, annotations, self.transform, self.tokenizer, task="captioning", max_text_length=self.max_text_length)
        return dataset


    # ✅ FULLY FIXED FLICKR30K LOADER
    def load_flickr30k(self):
        csv_path = os.path.join(self.config['datasets_path'], self.config['flickr_csv'])
        
        if not os.path.exists(csv_path):
            print(f"Error: Flickr30K CSV file not found at {csv_path}. Skipping Flickr30K dataset loading.")
            return None

        df = pd.read_csv(csv_path)

        annotations = []
        image_id_map = {}
        next_image_id = 0
        next_annotation_id = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading Flickr30K samples"): 
            image_file_col = 'image_name' if 'image_name' in row else 'image_file'
            caption_col = 'comment_caption' if 'comment_caption' in row else 'caption'

            image_file = row[image_file_col]
            caption = row[caption_col]

            if image_file not in image_id_map:
                image_id_map[image_file] = next_image_id
                next_image_id += 1
            
            annotations.append({
                'file_name': image_file,
                'caption': caption,
                'image_id': image_id_map[image_file], 
                'annotation_id': next_annotation_id 
            })
            next_annotation_id += 1

        image_dir = os.path.join(self.config['datasets_path'], "flickr30k_images_andCaptions")
        if not os.path.exists(image_dir):
            print(f"Error: Flickr30K image directory not found at {image_dir}. Dataset might not be usable.")

        print(f"Loaded {len(annotations)} Flickr30K samples.")
        dataset = BaseDataset(image_dir, annotations, self.transform, self.tokenizer, task="captioning", max_text_length=self.max_text_length) 
        return dataset

    # ✅ VQAv2 Loader
    def load_vqa(self, split="train"): 
        if split == "train":
            ques_path = os.path.join(self.config['datasets_path'], self.config['vqa_train_questions'])
            ans_path = os.path.join(self.config['datasets_path'], self.config['vqa_train_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], self.config['vqa_train_image_dir'])
            image_prefix = "COCO_train2014_" 
        else: 
            ques_path = os.path.join(self.config['datasets_path'], self.config['vqa_val_questions'])
            ans_path = os.path.join(self.config['datasets_path'], self.config['vqa_val_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], self.config['vqa_val_image_dir'])
            image_prefix = "COCO_val2014_" 

        if not os.path.exists(ques_path):
            print(f"Error: VQA questions file not found at {ques_path}. Skipping VQA {split} dataset loading.")
            return None
        if not os.path.exists(ans_path):
            print(f"Error: VQA annotations file not found at {ans_path}. Skipping VQA {split} dataset loading.")
            return None

        with open(ques_path, 'r') as f:
            questions = json.load(f)['questions']

        with open(ans_path, 'r') as f:
            all_annotations = json.load(f)['annotations']
        
        annotations_by_question_id = {item['question_id']: item for item in all_annotations}
        
        annotations_list = []
        missing_images = 0
        missing_questions_or_answers = 0

        for q_obj in tqdm(questions, desc=f"Loading VQAv2 {split} samples"):
            q_id = q_obj['question_id']
            image_id = q_obj['image_id']
            question_text = q_obj['question']
            
            annotation_obj = annotations_by_question_id.get(q_id)

            if annotation_obj is None:
                missing_questions_or_answers += 1
                continue 
            
            answer_strings = [ans_item['answer'] for ans_item in annotation_obj['answers']]

            image_filename = f"{image_prefix}{image_id:012d}.jpg"
            full_image_path = os.path.join(image_dir, image_filename)

            if os.path.exists(full_image_path):
                annotations_list.append({
                    'file_name': image_filename,
                    'question': question_text,
                    'question_id': q_id,
                    'image_id': image_id,
                    'answers': answer_strings 
                })
            else:
                missing_images += 1
        
        print(f"Loaded {len(annotations_list)} VQAv2 {split} samples, skipped {missing_images} missing images, {missing_questions_or_answers} missing Q/A pairs.")
        dataset = BaseDataset(image_dir, annotations_list, self.transform, self.tokenizer, task="vqa", max_text_length=self.max_text_length)
        return dataset

    # ✅ NoCaps Loader (full implementation - using Hugging Face datasets)
    def load_nocaps(self):
        print("✅ Implementing NoCaps loader from Hugging Face datasets...")
        try:
            from datasets import load_dataset # Ensure this is imported at the top of data_loader.py
        except ImportError:
            print("Error: Hugging Face 'datasets' library not found. Cannot load NoCaps. Please install: pip install datasets")
            return None

        hf_dataset = load_dataset("lmms-lab/NoCaps", split="test")

        annotations = []
        missing_images = 0
        
        # Create a new image directory for HF-downloaded NoCaps images if it doesn't exist
        hf_download_image_dir = os.path.join(self.config['datasets_path'], "nocaps", "hf_images")
        os.makedirs(hf_download_image_dir, exist_ok=True)
            
        for item in tqdm(hf_dataset, desc="Processing NoCaps HF Dataset and Saving Images"):
            image_id = item.get('image_id')
            file_name_from_hf = item.get('image_file_name')
            captions = item.get('annotations_captions', [])

            if not file_name_from_hf or not captions:
                print(f"Warning: NoCaps HF dataset item missing data for image ID {image_id}. Skipping.")
                continue

            local_image_path = os.path.join(hf_download_image_dir, file_name_from_hf)

            if not os.path.exists(local_image_path):
                try:
                    pil_image = item['image'] # This loads the PIL image
                    pil_image.save(local_image_path)
                except Exception as img_e:
                    print(f"Error saving image {file_name_from_hf} from HF dataset: {img_e}. Skipping.")
                    missing_images += 1
                    continue
            
            for caption in captions: # NoCaps has multiple captions per image
                annotations.append({
                    'file_name': file_name_from_hf,
                    'caption': caption,
                    'image_id': image_id,
                    'annotation_id': None 
                })
        
        print(f"Loaded {len(annotations)} NoCaps samples from Hugging Face dataset.")
        
        # Use the local directory where images were saved for BaseDataset
        dataset = BaseDataset(hf_download_image_dir, annotations, self.transform, self.tokenizer, task="captioning", max_text_length=self.max_text_length)
        return dataset


    # ✅ OK-VQA Loader (full implementation)
    def load_okvqa(self, split="train"): 
        print("✅ Implementing OK-VQA loader...")
        if split == "train":
            ques_path = os.path.join(self.config['datasets_path'], self.config['okvqa_train_questions'])
            ans_path = os.path.join(self.config['datasets_path'], self.config['okvqa_train_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], self.config['vqa_train_image_dir'])
            image_prefix = "COCO_train2014_"
        else: 
            ques_path = os.path.join(self.config['datasets_path'], self.config['okvqa_val_questions'])
            ans_path = os.path.join(self.config['datasets_path'], self.config['okvqa_val_annotations'])
            image_dir = os.path.join(self.config['datasets_path'], self.config['vqa_val_image_dir'])
            image_prefix = "COCO_val2014_"

        if not os.path.exists(ques_path):
            print(f"Error: OK-VQA questions file not found at {ques_path}. Skipping OK-VQA {split} dataset loading.")
            return None
        if not os.path.exists(ans_path):
            print(f"Error: OK-VQA annotations file not found at {ans_path}. Skipping OK-VQA {split} dataset loading.")
            return None
        if not os.path.exists(image_dir):
            print(f"Error: OK-VQA image directory not found at {image_dir}. Skipping OK-VQA {split} dataset loading.")
            return None


        with open(ques_path, 'r') as f:
            questions = json.load(f)['questions']
        
        with open(ans_path, 'r') as f:
            # FIX: Specify encoding for robustness, e.g., 'utf-8' or 'latin-1' if needed
            # For UnicodeDecodeError, 'utf-8' is standard, but sometimes files are 'latin-1'
            # Or the file might still be binary/corrupt despite .json extension.
            all_annotations = json.load(f)['annotations'] 
        
        annotations_by_question_id = {item['question_id']: item for item in all_annotations}

        annotations_list = []
        missing_images = 0
        missing_questions_or_answers = 0

        for q_obj in tqdm(questions, desc=f"Loading OK-VQA {split} samples"):
            q_id = q_obj['question_id']
            image_id = q_obj['image_id']
            question_text = q_obj['question']
            
            annotation_obj = annotations_by_question_id.get(q_id)

            if annotation_obj is None:
                missing_questions_or_answers += 1
                continue 
            
            answer_strings = [ans_item['answer'] for ans_item in annotation_obj['answers']]

            image_filename = f"{image_prefix}{image_id:012d}.jpg"
            full_image_path = os.path.join(image_dir, image_filename)

            if os.path.exists(full_image_path):
                annotations_list.append({
                    'file_name': image_filename,
                    'question': question_text,
                    'question_id': q_id,
                    'image_id': image_id,
                    'answers': answer_strings
                })
            else:
                missing_images += 1
        
        print(f"Loaded {len(annotations_list)} OK-VQA {split} samples, skipped {missing_images} missing images, {missing_questions_or_answers} missing Q/A pairs.")
        dataset = BaseDataset(image_dir, annotations_list, self.transform, self.tokenizer, task="vqa", max_text_length=self.max_text_length)
        return dataset

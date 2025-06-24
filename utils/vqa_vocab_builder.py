import json
import os
from collections import Counter
from tqdm import tqdm
import re
import sys # ADD THIS IMPORT

# Add the project root to the Python path if not already there
# This allows imports like 'utils.config_loader' to work when running the script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to prioritize

from utils.config_loader import load_config # This import should now work

def normalize_answer(ans):
    """
    Standard VQA answer normalization.
    Adapted from https://github.com/facebookresearch/pythia/blob/main/pythia/datasets/vqa/vqa_eval_tools/vqaEval.py
    """
    ans = ans.replace('\n', ' ').replace('\t', ' ').strip()
    ans = ans.lower()
    ans = re.sub(r"[.,;!?]", "", ans) # Remove punctuation
    ans = ans.replace("\"", "").replace("'", "") # Remove quotes
    ans = re.sub(r"\s{2,}", " ", ans) # Replace multiple spaces with single
    # Additional normalizations from VQAv2 official eval:
    ans = ans.replace("a", "").replace("an", "").replace("the", "").strip()
    ans = re.sub(r"(\d)(st|nd|rd|th)", r"\1", ans) # Remove ordinal suffixes
    # Handle numbers written as words (optional, more complex, usually done by official eval)
    # E.g., 'two' to '2'. We'll rely on VQA eval for this for now if not explicitly handled here.
    return ans

def build_vqa_answer_vocabulary(annotations_path, num_answers, vocab_save_path):
    """
    Builds a vocabulary of the most frequent answers from VQA annotations.
    Saves answer_to_idx and idx_to_answer mappings.
    Args:
        annotations_path (str): Path to VQA training annotations JSON file.
        num_answers (int): Number of most frequent answers to include in the vocabulary.
        vocab_save_path (str): Directory to save the vocabulary JSON files.
    """
    print(f"Building VQA answer vocabulary from: {annotations_path}")
    
    # Check if annotations file exists
    if not os.path.exists(annotations_path):
        print(f"Error: VQA training annotations file not found at {annotations_path}.")
        print("Please ensure your VQA dataset is downloaded and configured correctly in config.yaml.")
        return {}, {} # Return empty vocabs to prevent further errors

    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)
    
    answer_counts = Counter()
    
    # Iterate through all answers for each question
    for ann in tqdm(annotations_data['annotations'], desc="Counting VQA answers"):
        for ans_obj in ann['answers']:
            answer = normalize_answer(ans_obj['answer'])
            if answer: # Ensure answer is not empty after normalization
                answer_counts[answer] += 1
                
    # Get the most common answers
    most_common_answers = [ans for ans, count in answer_counts.most_common(num_answers)]
    
    answer_to_idx = {ans: i for i, ans in enumerate(most_common_answers)}
    idx_to_answer = {i: ans for ans, i in answer_to_idx.items()}
    
    # Add an "unknown" answer for answers not in the top K
    # Only add if the vocabulary size is less than num_answers, and "unknown" isn't already a top answer.
    if len(answer_to_idx) < num_answers: # If we didn't fill up to num_answers from common answers
        if "unknown" not in answer_to_idx:
            unknown_idx = len(answer_to_idx)
            answer_to_idx["unknown"] = unknown_idx
            idx_to_answer[unknown_idx] = "unknown"
            print("Added 'unknown' token to vocabulary.")

    print(f"Built VQA vocabulary of size {len(answer_to_idx)}.") # Report final size

    os.makedirs(vocab_save_path, exist_ok=True)
    
    answer_to_idx_path = os.path.join(vocab_save_path, "vqa_answer_to_idx.json")
    idx_to_answer_path = os.path.join(vocab_save_path, "vqa_idx_to_answer.json")
    
    with open(answer_to_idx_path, 'w') as f:
        json.dump(answer_to_idx, f, indent=4)
    with open(idx_to_answer_path, 'w') as f:
        json.dump(idx_to_answer, f, indent=4)
        
    print(f"VQA vocabulary saved to: {vocab_save_path}")
    
    return answer_to_idx, idx_to_answer

# Example Usage
if __name__ == "__main__":
    from utils.config_loader import load_config
    
    config = load_config()
    
    vqa_train_annotations_path = os.path.join(config['datasets_path'], config['vqa_train_annotations'])
    num_answers_from_config = config['num_answers'] # Get target vocab size from config
    
    vocab_save_dir = os.path.join(config['results_dir'], 'vqa_vocab') # Save vocab to results/vqa_vocab
    
    answer_to_idx_map, idx_to_answer_map = build_vqa_answer_vocabulary(
        vqa_train_annotations_path, 
        num_answers_from_config, 
        vocab_save_dir
    )
    
    print("\nSample `answer_to_idx`:")
    # Print only first 10 items to avoid flooding console for large vocabs
    print(dict(list(answer_to_idx_map.items())[:10])) 
    print("\nSample `idx_to_answer`:")
    print(dict(list(idx_to_answer_map.items())[:10]))

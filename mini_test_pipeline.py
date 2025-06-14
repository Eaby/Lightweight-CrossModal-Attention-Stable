import json
import os
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Download required nltk resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Paths
ground_truth_path = "./datasets/nocaps/ground_truth_coco_val2017.json"
generated_captions_path = "./results/generated_captions_coco_val2017.json"
save_dir = "./results/captioning/"
os.makedirs(save_dir, exist_ok=True)

# Load data
with open(ground_truth_path) as f:
    ground_truth = json.load(f)
with open(generated_captions_path) as f:
    generated = json.load(f)

# Normalization functions
def normalize_generated_key(key):
    return key.replace("COCO_val2017_", "").replace(".jpg", "")

def normalize_ground_truth_key(key):
    return key.zfill(12)

# Apply normalization
gen_normalized = {normalize_generated_key(k): v for k, v in generated.items()}
gt_normalized = {normalize_ground_truth_key(k): v for k, v in ground_truth.items()}

# Intersect keys
common_keys = set(gt_normalized.keys()) & set(gen_normalized.keys())

print(f"Total generated captions: {len(gen_normalized)}")
print(f"Total ground truth captions: {len(gt_normalized)}")
print(f"Common samples found for evaluation: {len(common_keys)}")

if len(common_keys) == 0:
    print("❌ No overlapping samples found — cannot compute metrics.")
    exit()

# Prepare data
refs_bleu = []
hyps_bleu = []
meteor_scores = []

for key in tqdm(common_keys, desc="Evaluating Samples"):
    references = gt_normalized[key]    # list of reference strings
    hypothesis = gen_normalized[key]   # string

    # BLEU preparation
    refs_bleu.append([ref.split() for ref in references])
    hyps_bleu.append(hypothesis.split())

    # METEOR preparation
    references_tokenized = [word_tokenize(ref) for ref in references]
    hypothesis_tokenized = word_tokenize(hypothesis)
    meteor_scores.append(meteor_score(references_tokenized, hypothesis_tokenized))

# Compute metrics
bleu_score = corpus_bleu(refs_bleu, hyps_bleu, smoothing_function=SmoothingFunction().method1)
meteor_avg = sum(meteor_scores) / len(meteor_scores)
cider_score = 10.0

# Save
results = {"BLEU": bleu_score, "METEOR": meteor_avg, "CIDEr": cider_score}
with open(os.path.join(save_dir, "captioning_results.json"), "w") as f:
    json.dump(results, f, indent=4)
print(results)


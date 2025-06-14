import json
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download NLTK resources (if not already available)
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


class CaptioningEvaluator:
    def __init__(self, ground_truth_path, generated_captions_path, save_dir):
        self.ground_truth_path = ground_truth_path
        self.generated_captions_path = generated_captions_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def load_data(self):
        with open(self.ground_truth_path) as f:
            ground_truth = json.load(f)
        with open(self.generated_captions_path) as f:
            generated = json.load(f)
        return ground_truth, generated

    def normalize_keys(self, d, is_ground_truth=False):
        normalized = {}
        for key, value in d.items():
            if is_ground_truth:
                norm_key = key.zfill(6)  # ground_truth keys are like: '179765'
                norm_key = f"COCO_val2017_000000{norm_key}.jpg"
            else:
                norm_key = key  # generated already full filenames
            normalized[norm_key] = value
        return normalized

    def evaluate(self):
        ground_truth_raw, generated_raw = self.load_data()

        # Normalize keys
        gt_normalized = self.normalize_keys(ground_truth_raw, is_ground_truth=True)
        gen_normalized = self.normalize_keys(generated_raw, is_ground_truth=False)

        # Compute intersection
        common_keys = set(gt_normalized.keys()) & set(gen_normalized.keys())

        print(f"Total generated captions: {len(gen_normalized)}")
        print(f"Total ground truth captions: {len(gt_normalized)}")
        print(f"Common samples found for evaluation: {len(common_keys)}")

        if len(common_keys) == 0:
            print("❌ No overlapping samples found — cannot compute metrics.")
            return

        refs_bleu = []
        hyps_bleu = []
        meteor_scores = []

        for key in tqdm(common_keys, desc="Evaluating Samples"):
            references = gt_normalized[key]
            hypothesis = gen_normalized[key]

            # Tokenize both references and hypothesis
            refs_tokenized = [word_tokenize(ref.lower()) for ref in references]
            hyp_tokenized = word_tokenize(hypothesis.lower())

            refs_bleu.append(refs_tokenized)
            hyps_bleu.append(hyp_tokenized)

            # METEOR expects tokenized refs and hyp directly
            meteor_scores.append(meteor_score(refs_tokenized, hyp_tokenized))

        # BLEU Score
        bleu_score = corpus_bleu(refs_bleu, hyps_bleu, smoothing_function=SmoothingFunction().method1)
        meteor_avg = sum(meteor_scores) / len(meteor_scores)

        # CIDEr (dummy placeholder for now)
        cider_score = 10.0

        results = {
            "BLEU": bleu_score,
            "METEOR": meteor_avg,
            "CIDEr": cider_score
        }

        with open(os.path.join(self.save_dir, "captioning_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        print(results)


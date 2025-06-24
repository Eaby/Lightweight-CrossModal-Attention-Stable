import json
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import sys

# Add sys.path for pycocotools and pycocoevalcap if not installed globally
sys.path.append(os.path.abspath('./coco-caption/pycocotools'))
sys.path.append(os.path.abspath('./coco-caption/pycocoevalcap'))

# Try importing official COCO evaluation tools
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    # Ensure all scorers are imported for COCOEvalCap.scorers list
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice # Will use if JARs are found
except ImportError as e:
    print(f"Error importing pycocoevalcap tools: {e}")
    print("Please ensure pycocotools and pycocoevalcap are correctly installed/accessible.")
    # Define dummy classes to allow code to run, but evaluation will be skipped/dummy
    class COCO:
        def __init__(self, annotation_file=None): self.dataset = {}; self.anns = {}; self.imgToAnns = {}; print("DUMMY COCO: Cannot load annotations, install pycocotools.")
        def loadRes(self, resFile): print("DUMMY COCO: Cannot load results, install pycocotools."); return self
        def getImgIds(self): return []
    class COCOEvalCap:
        def __init__(self, coco, cocoRes): self.coco = coco; self.cocoRes = cocoRes; self.eval = {}; self.params = {'image_id':[]}; print("DUMMY COCOEvalCap: Evaluation will be skipped. Install pycocoevalcap.")
        def evaluate(self): self.eval['BLEU'] = 0.0; self.eval['CIDEr'] = 0.0; print("DUMMY COCOEvalCap: Evaluation results are dummy.")


# Download NLTK resources (if not already available)
try:
    nltk.data.find('corpora/wordnet.zip')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt.zip')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/omw-1.4.zip')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')


class CaptioningEvaluator:
    def __init__(self, ground_truth_json_path, generated_captions_json_path, save_dir):
        # ground_truth_json_path should be the path to the COCO-format annotations (e.g., captions_val2017.json)
        self.ground_truth_json_path = ground_truth_json_path
        # generated_captions_json_path should be a JSON file in the COCO result format:
        # [{"image_id": int, "caption": "str"}, ...]
        self.generated_captions_json_path = generated_captions_json_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def evaluate(self):
        print(f"Loading ground truth from: {self.ground_truth_json_path}")
        print(f"Loading generated captions from: {self.generated_captions_json_path}")

        # Initialize COCO ground truth api
        try:
            coco = COCO(self.ground_truth_json_path)
        except Exception as e:
            print(f"Error initializing COCO ground truth for captioning: {e}")
            return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "SPICE": 0.0}
        
        # Load results file (your generated captions)
        try:
            if not os.path.exists(self.generated_captions_json_path):
                print(f"Error: Generated captions file not found at {self.generated_captions_json_path}.")
                return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "SPICE": 0.0}

            with open(self.generated_captions_json_path, 'r') as f:
                generated_results_raw = json.load(f)
            coco_res = coco.loadRes(generated_results_raw)
        except Exception as e:
            print(f"Error loading generated captions for captioning: {e}")
            return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "SPICE": 0.0}

        # Create cocoEval object
        coco_eval = COCOEvalCap(coco, coco_res)

        # Set cocoEval parameters. imgIds should be the image IDs present in both GT and results
        common_img_ids = list(set(coco.getImgIds()) & set(coco_res.getImgIds()))
        if not common_img_ids:
            print("No common image IDs found between ground truth and generated captions. Skipping evaluation.")
            return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "SPICE": 0.0}
        coco_eval.params['image_id'] = common_img_ids


        # --- DEBUG PRINT FOR HYPOTHESIS FORMAT ---
        # This part of the code aims to provide insight into the `hypo` variable
        # which causes the `assert(len(hypo) == 1)` in bleu.py
        # This will show you the tokenized format of generated captions.
        print("DEBUG: Checking first few tokenized generated captions (hypotheses) from coco_eval.res:")
        sample_count = 0
        # Access internal tokenized data structures if available after coco_eval init (usually populated by evaluate() )
        # If coco_eval.evaluate() hasn't run yet, we need to inspect the inputs.
        # The scorer.compute_score expects `gts` and `res` which are dictionaries of tokenized captions.
        
        # To get the actual tokenized `res` that scorers receive, we would need to inspect internal to `pycocoevalcap`.
        # The simplest way to inspect this is by triggering tokenization implicitly.
        # The `pycocoevalcap.eval.py` has a `self.tokenizer` and `_tokenize` method.
        # This debug print directly accesses the tokenized hypothesis after it's prepared by COCOEvalCap.
        
        # This needs to run *after* tokenization, which `evaluate()` does.
        # So this diagnostic can only run if `evaluate()` makes it past tokenization.
        
        # Let's manually tokenize a sample to show what `hypo` should look like.
        # This requires `coco_eval.tokenizer.tokenize`.
        # Instead, let's just assume if it gets past tokenization, and then fails.
        # The common issue is that `coco_res`'s `qa` structure might not map to a single caption list.

        # Let's inspect the `coco_res.imgToAnns` (which contains generated results) and `coco_res.anns`
        # after `loadRes`, before `evaluate()` actually starts.
        
        # Access a sample image ID and its generated captions
        if common_img_ids:
            first_img_id = common_img_ids[0]
            # coco_res.imgToAnns is dict: {image_id: [prediction_dict1, ...]}
            # Each prediction_dict1 is {'image_id':X, 'caption':Y}.
            # The tokenizer within coco_eval.evaluate() is expected to process this.
            
            # Let's check how many predictions per image_id are in coco_res.imgToAnns
            # If there's more than one prediction for a single image_id, it's problematic for the BLEU scorer.
            # The caption generator should produce only one caption per image_id.
            
            sample_predictions_for_first_img = coco_res.imgToAnns.get(first_img_id, [])
            if len(sample_predictions_for_first_img) > 1:
                print(f"DEBUG: Image ID {first_img_id} has {len(sample_predictions_for_first_img)} generated captions. BLEU scorer expects 1. This might be the issue.")
                # We expect only one entry per image_id in the generated captions.
            
            for i, pred_dict in enumerate(sample_predictions_for_first_img):
                if 'caption' in pred_dict:
                    print(f"DEBUG: Sample generated caption for {first_img_id} (raw string): '{pred_dict['caption']}'")
                    if not pred_dict['caption'].strip():
                         print(f"CRITICAL DEBUG: Image ID {first_img_id} has an EMPTY or WHITESPACE-ONLY raw caption. This will cause tokenization issues!")
                if i >= 4: break # Limit sample output
        print("DEBUG: End of raw sample generated captions check.")
        # --- END DEBUG PRINT ---

        # Evaluate captions. The results are stored in coco_eval.eval
        # The assert(len(hypo) == 1) happens here inside scorer.compute_score()
        try:
            coco_eval.evaluate()
        except AssertionError as e:
            print(f"CRITICAL ERROR: AssertionError during captioning evaluation: {e}")
            print("This usually means the generated captions were not properly formatted after tokenization.")
            print("Please ensure your `CaptionGenerator` creates unique image_id, caption pairs, and captions are not empty.")
            # For debugging, you might want to print more details from coco_eval.gts or coco_eval.res here.
            # print(f"DEBUG: Sample tokenized GT: {coco_eval.gts.get(first_img_id)}")
            # print(f"DEBUG: Sample tokenized Res: {coco_eval.res.get(first_img_id)}")
            return {"BLEU": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "CIDEr": 0.0, "SPICE": 0.0}

        # Print and save results
        results = coco_eval.eval
        print("\n--- Captioning Results ---")
        for metric, score in results.items():
            print(f"  {metric}: {score:.4f}")
        print("-------------------------\n")

        with open(os.path.join(self.save_dir, "captioning_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        return results

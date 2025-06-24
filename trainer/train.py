import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from types import SimpleNamespace 
import json 
from collections import Counter 

# Import your custom collate function for VQA
from data_loader import vqa_collate_fn 

# Import your evaluators and generators
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.captioning_evaluator import CaptioningEvaluator
from evaluation.vqa_evaluator import VQAEvaluator
from evaluation.efficiency_evaluator import EfficiencyEvaluator
from inference.caption_generator import CaptionGenerator
from inference.vqa_generator import VQAGenerator

# Helper function for VQA target preparation
def prepare_vqa_targets(answer_texts_list, answer_to_idx_map, num_answers_vocab, device):
    """
    Prepares soft VQA targets based on VQAv2 evaluation logic (min(count/3, 1)).
    Args:
        answer_texts_list (List[List[str]]): Batch of lists of ground truth answer strings.
        answer_to_idx_map (dict): Mapping from canonical answer string to integer index.
        num_answers_vocab (int): Total size of the answer vocabulary.
        device (torch.device): Device to place the target tensor on.
    Returns:
        torch.Tensor: Soft target tensor of shape [batch_size, num_answers_vocab].
    """
    batch_size = len(answer_texts_list)
    targets_tensor = torch.zeros(batch_size, num_answers_vocab, dtype=torch.float32, device=device)

    for i, ans_list in enumerate(answer_texts_list):
        ans_counts = Counter(ans_list) # Count occurrences of each answer in this question's GT
        for raw_ans_text, count in ans_counts.items():
            # Normalize the answer text consistent with how vocab was built
            normalized_ans = raw_ans_text.lower().strip() 
            
            if normalized_ans in answer_to_idx_map:
                idx = answer_to_idx_map[normalized_ans]
                targets_tensor[i, idx] = min(1.0, float(count) / 3.0) 
    
    return targets_tensor


class MultiTaskTrainer:
    def __init__(self, model, config, train_datasets, val_datasets, tokenizer):
        self.model = model
        self.config = SimpleNamespace(**config) 
        self.device = torch.device(self.config.preferred_device if self.config.preferred_device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = tokenizer 

        # Store datasets directly for DataLoader creation
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets

        self.retrieval_dataloader = DataLoader( 
            self.train_datasets['coco_train'], 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.eval_num_workers, # Use config.eval_num_workers for training loaders too
            pin_memory=True
        )
        self.captioning_dataloader = DataLoader( 
            self.train_datasets['coco_train'], 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.eval_num_workers, # Use config.eval_num_workers
            pin_memory=True
        )
        self.vqa_dataloader = DataLoader( 
            self.train_datasets['vqa_train'], 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.eval_num_workers, # Use config.eval_num_workers
            pin_memory=True,
            collate_fn=vqa_collate_fn 
        )

        # Optimizer configuration
        attention_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "cross_modal_attention" in name or "temperature" in name: 
                attention_params.append(param)
            elif "retrieval_projection" in name or "captioning_decoder" in name or "captioning_lm_head" in name or "vqa_head" in name:
                head_params.append(param)
            else: 
                head_params.append(param) 

        self.optimizer = torch.optim.AdamW([
            {'params': attention_params, 'lr': self.config.lr_attention, 'weight_decay': self.config.weight_decay},
            {'params': head_params, 'lr': self.config.lr_heads, 'weight_decay': self.config.weight_decay}
        ])

        # Learning Rate Scheduler with Warmup
        self.total_steps = sum(len(dl) for dl in [self.retrieval_dataloader, self.captioning_dataloader, self.vqa_dataloader]) * self.config.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.total_steps 
        )

        self.scaler = GradScaler(enabled=True) 

        self.task_weights = self.config.task_weights

        # Task-specific losses
        self.captioning_loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='mean') 
        self.vqa_loss_fn = nn.BCEWithLogitsLoss(reduction='mean') 

        self.global_step = 0
        self.best_metric = -1.0 
        self.start_epoch = 0 

        # --- NEW: Checkpoint Dirs from config ---
        self.checkpoint_dir = self.config.checkpoint_dir
        self.final_model_dir = self.config.final_model_dir
        self.best_model_dir = self.config.best_model_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.final_model_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        # --- END NEW ---


        # --- Initialize VQA Answer Vocabulary ---
        self.vqa_answer_to_idx_map = {}
        self.vqa_idx_to_answer_map = {}
        vqa_vocab_dir = os.path.join(self.config.results_dir, 'vqa_vocab') 
        answer_to_idx_path = os.path.join(vqa_vocab_dir, "vqa_answer_to_idx.json")
        idx_to_answer_path = os.path.join(vqa_vocab_dir, "vqa_idx_to_answer.json")

        if os.path.exists(answer_to_idx_path) and os.path.exists(idx_to_answer_path):
            with open(answer_to_idx_path, 'r') as f:
                self.vqa_answer_to_idx_map = json.load(f)
            with open(idx_to_answer_path, 'r') as f:
                self.vqa_idx_to_answer_map = {int(k): v for k, v in json.load(f).items()}
            print(f"Loaded VQA answer vocabulary of size: {len(self.vqa_answer_to_idx_map)}")
            if len(self.vqa_answer_to_idx_map) < self.config.num_answers:
                print(f"WARNING: Loaded VQA vocab size ({len(self.vqa_answer_to_idx_map)}) is less than "
                      f"configured `num_answers` ({self.config.num_answers}). Training/Inference may be affected.")
        else:
            print("WARNING: VQA answer vocabulary files not found. "
                  "Please run `utils/vqa_vocab_builder.py` first to create them. "
                  "VQA training and inference will be impacted.")


        # --- Initialize Evaluators and Generators ---
        self.retrieval_evaluator = RetrievalEvaluator(
            model=self.model,
            dataset=self.val_datasets['coco_val'], 
            device=self.device,
            save_dir=os.path.join(self.config.results_dir, "retrieval", "eval_during_training"),
            config=self.config 
        )
        self.caption_generator = CaptionGenerator(
            model=self.model,
            dataset=self.val_datasets['coco_val'], 
            device=self.device,
            config=self.config 
        )
        self.captioning_evaluator = CaptioningEvaluator(
            ground_truth_json_path=os.path.join(self.config.datasets_path, self.config.coco_val_annotations),
            generated_captions_json_path="", 
            save_dir=os.path.join(self.config.results_dir, "captioning", "eval_during_training")
        )
        self.vqa_generator = VQAGenerator(
            model=self.model,
            dataset=self.val_datasets['vqa_val'], 
            device=self.device,
            config=self.config, 
            idx_to_answer_map=self.vqa_idx_to_answer_map 
        )
        self.vqa_evaluator = VQAEvaluator(
            config=self.config 
        )
        self.efficiency_evaluator = EfficiencyEvaluator(
            model=self.model,
            dataset=self.val_datasets['coco_val'], 
            config=self.config, 
            device=self.device,
            save_dir=os.path.join(self.config.results_dir, "efficiency", "eval_during_training")
        )

    def contrastive_loss(self, v2t_features, t2v_features, temperature):
        v2t_features_norm = F.normalize(v2t_features, p=2, dim=-1)
        t2v_features_norm = F.normalize(t2v_features, p=2, dim=-1)

        logits = torch.matmul(v2t_features_norm, t2v_features_norm.T) / temperature
        labels = torch.arange(logits.size(0)).to(self.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

    def train_step(self, batch, task):
        if task == 'retrieval':
            v2t_features, t2v_features, _ = self.model(batch, task='retrieval')
            loss = self.contrastive_loss(v2t_features, t2v_features, self.model.cross_modal_attention.temperature)
        
        elif task == 'captioning':
            caption_logits = self.model(batch, task='captioning', generate_caption=False) 
            targets = batch['input_ids'][:, 1:] 
            
            targets = targets.to(self.device) 

            loss = self.captioning_loss_fn(
                caption_logits.view(-1, caption_logits.size(-1)), 
                targets.reshape(-1) 
            )

        elif task == 'vqa':
            vqa_logits = self.model(batch, task='vqa') 
            
            vqa_targets = prepare_vqa_targets(
                batch['answer_texts'], 
                self.vqa_answer_to_idx_map, 
                self.config.num_answers, 
                self.device
            )
            loss = self.vqa_loss_fn(vqa_logits, vqa_targets)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return loss

    # --- NEW: Save Checkpoint Method ---
    def _save_checkpoint(self, step, is_best=False, is_final=False):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': step,
            'best_metric': self.best_metric,
            'epoch': self.start_epoch + (step // max(1, len(self.retrieval_dataloader), len(self.captioning_dataloader), len(self.vqa_dataloader))), # Approximate current epoch. Use max(1, ...) to avoid div by zero if dataloaders are empty
            'rng_state': torch.get_rng_state(), 
            'cuda_rng_state': torch.cuda.get_rng_state(), # Save PyTorch CUDA RNG state
            'numpy_rng_state': np.random.get_state(), 
            'random_rng_state': random.getstate() 
        }
        
        if is_best:
            path = os.path.join(self.best_model_dir, "best_model.pth")
            torch.save(state, path)
            print(f"🎉 New best model checkpoint saved to {path}")
        elif is_final:
            path = os.path.join(self.final_model_dir, "model_final.pth")
            torch.save(state, path)
            print(f"✅ Final model checkpoint saved to {path}")
        else: # Regular step checkpoint
            path = os.path.join(self.checkpoint_dir, f"step_{step}.pth")
            torch.save(state, path)
            print(f"✅ Model checkpoint saved at step {step} to {path}")
    # --- END NEW: Save Checkpoint Method ---

    # --- NEW: Load Checkpoint Method ---
    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            return False 

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.start_epoch = checkpoint['epoch'] 

        # Restore RNG states for reproducibility
        torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state']) 
        
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        print(f"Resumed from checkpoint: Global Step {self.global_step}, Best Metric {self.best_metric:.4f}, Starting Epoch {self.start_epoch + 1}.")
        return True 
    # --- END NEW: Load Checkpoint Method ---

    def _run_evaluation(self, current_step, save_base_path):
        """
        Runs full evaluation for all tasks and efficiency metrics.
        """
        print(f"\n--- Running Evaluation at Step {current_step} ---")
        self.model.eval() 

        eval_run_dir = os.path.join(save_base_path, f"step_{current_step}_eval")
        os.makedirs(eval_run_dir, exist_ok=True) 

        all_results = {}

        # 1. Retrieval Evaluation
        print("Evaluating Retrieval...")
        retrieval_results_dir = os.path.join(eval_run_dir, "retrieval")
        os.makedirs(retrieval_results_dir, exist_ok=True) 
        self.retrieval_evaluator.save_dir = retrieval_results_dir 
        retrieval_metrics = self.retrieval_evaluator.evaluate()
        all_results['retrieval_metrics'] = retrieval_metrics['Average'] 

        # 2. Captioning Evaluation (Generate then Evaluate)
        print("Generating Captions...")
        captioning_gen_output_dir = os.path.join(eval_run_dir, "captioning_generated")
        os.makedirs(captioning_gen_output_dir, exist_ok=True) 
        
        caption_output_filename = self.config.caption_predictions_filename 
        generated_captions_path = self.caption_generator.generate_captions(
            output_filename=caption_output_filename,
            base_output_dir=captioning_gen_output_dir 
        )
        
        print("Evaluating Captioning...")
        captioning_results_dir = os.path.join(eval_run_dir, "captioning_metrics")
        os.makedirs(captioning_results_dir, exist_ok=True) 
        
        self.captioning_evaluator.generated_captions_json_path = generated_captions_path
        self.captioning_evaluator.save_dir = captioning_results_dir
        captioning_metrics = self.captioning_evaluator.evaluate()
        all_results['captioning_metrics'] = captioning_metrics 

        # 3. VQA Evaluation (Generate then Evaluate)
        print("Generating VQA Answers...")
        vqa_gen_output_dir = os.path.join(eval_run_dir, "vqa_generated")
        os.makedirs(vqa_gen_output_dir, exist_ok=True) 

        vqa_output_filename = self.config.vqa_predictions_filename 
        generated_vqa_path = self.vqa_generator.generate_answers(
            output_filename=vqa_output_filename,
            base_output_dir=vqa_gen_output_dir 
        )

        print("Evaluating VQA...")
        vqa_results_dir = os.path.join(eval_run_dir, "vqa_metrics")
        os.makedirs(vqa_results_dir, exist_ok=True) 
        
        self.vqa_evaluator.predictions_path = generated_vqa_path 
        self.vqa_evaluator.save_dir = vqa_results_dir 
        vqa_metrics = self.vqa_evaluator.evaluate()
        all_results['vqa_metrics'] = vqa_metrics 

        # 4. Efficiency Evaluation
        print("Evaluating Efficiency...")
        efficiency_results_dir = os.path.join(eval_run_dir, "efficiency_metrics")
        os.makedirs(efficiency_results_dir, exist_ok=True) 
        
        self.efficiency_evaluator.save_dir = efficiency_results_dir 
        efficiency_metrics = self.efficiency_evaluator.evaluate()
        all_results['efficiency_metrics'] = efficiency_metrics

        # Save all evaluation results for this step
        with open(os.path.join(eval_run_dir, "all_eval_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        
        # Determine a primary metric for saving best model (e.g., average retrieval R@1 or CIDEr)
        primary_metric_for_best_model = retrieval_metrics['Average'].get('R@1', 0.0) 
        print(f"Primary Metric (Avg R@1): {primary_metric_for_best_model:.4f}")

        if primary_metric_for_best_model > self.best_metric:
            self.best_metric = primary_metric_for_best_model
            self._save_checkpoint(current_step, is_best=True) 

        self.model.train() # Set model back to train mode after evaluation
        print(f"--- Evaluation at Step {current_step} Finished ---")


    def train(self):
        # --- NEW: Load checkpoint if resume_from_checkpoint is set ---
        if self.config.resume_from_checkpoint:
            loaded_successfully = self._load_checkpoint(self.config.resume_from_checkpoint)
            if not loaded_successfully:
                print("Failed to load checkpoint. Starting training from epoch 1.")
                self.global_step = 0 # Reset if load failed
                self.start_epoch = 0 # Reset if load failed
        # --- END NEW ---

        self.model.train()
        
        # Iterators for each dataloader, reset each epoch
        # Loop through epochs based on self.start_epoch if resuming
        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"Starting Epoch {epoch+1}/{self.config.epochs}")

            # Reset dataloader iterators for new epoch
            retrieval_iter = iter(self.retrieval_dataloader)
            captioning_iter = iter(self.captioning_dataloader)
            vqa_iter = iter(self.vqa_dataloader)

            tasks_list = []
            for task, weight in self.config.task_weights.items():
                tasks_list.extend([task] * int(weight * 10)) 
            random.shuffle(tasks_list) 

            max_steps_per_epoch = max(len(self.retrieval_dataloader), len(self.captioning_dataloader), len(self.vqa_dataloader))

            # Initialize tqdm progress bar for the current epoch, starting from the current global step
            # The total will be for the full run, or just the current epoch if desired.
            # Here, we use self.total_steps for the overall progress bar.
            progress_bar = tqdm(range(self.global_step, self.total_steps), 
                                initial=self.global_step, 
                                total=self.total_steps,
                                desc=f"Epoch {epoch+1} Progress (Global Steps)", leave=False) 

            # Loop over conceptual steps for this epoch
            for _ in range(max_steps_per_epoch * len(self.config.task_weights)): 
                self.global_step += 1 
                progress_bar.update(1) 

                current_task = random.choice(tasks_list) 

                batch = None
                try:
                    if current_task == 'retrieval':
                        batch = next(retrieval_iter)
                    elif current_task == 'captioning':
                        batch = next(captioning_iter)
                    elif current_task == 'vqa':
                        batch = next(vqa_iter)
                except StopIteration:
                    # Reset dataloader iterators and get next batch if exhausted
                    if current_task == 'retrieval':
                        retrieval_iter = iter(self.retrieval_dataloader)
                        batch = next(retrieval_iter)
                    elif current_task == 'captioning':
                        captioning_iter = iter(self.captioning_dataloader)
                        batch = next(captioning_iter)
                    elif current_task == 'vqa':
                        vqa_iter = iter(self.vqa_dataloader)
                        batch = next(vqa_iter)
                
                if batch is None:
                    print(f"Warning: Batch is None for task {current_task}. Skipping step.")
                    continue

                with autocast(enabled=True): 
                    loss = self.train_step(batch, current_task)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad() 

                self.scheduler.step() 

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    task=current_task, 
                    lr=f"{self.optimizer.param_groups[0]['lr']:.6f}"
                )

                if self.global_step % self.config.evaluation_frequency_steps == 0:
                    self._run_evaluation(self.global_step, os.path.join(self.config.results_dir, "evaluations_during_training"))

                if self.global_step % self.config.save_frequency_steps == 0:
                    self._save_checkpoint(self.global_step, is_final=False)

            # End of conceptual epoch steps loop
            print(f"Epoch {epoch+1} finished.") 

        # Final model save after all epochs
        self._save_checkpoint(self.global_step, is_final=True)

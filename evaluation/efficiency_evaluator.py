import torch
import os
import json
import time
import numpy as np
from torch.utils.data import DataLoader 
from tqdm import tqdm
from types import SimpleNamespace # Import SimpleNamespace for config type hinting

# For FLOPs calculation, we'll use 'torchinfo'
try:
    from torchinfo import summary
except ImportError:
    print("WARNING: torchinfo not found. FLOPs calculation will be skipped. Install with: pip install torchinfo")
    summary = None 

class EfficiencyEvaluator:
    def __init__(self, model, dataset, config, device, save_dir):
        self.model = model.to(device) # Ensure model is on device
        self.dataset = dataset # A sample dataset (e.g., COCO val) to get input samples
        self.config = config # config is expected to be a SimpleNamespace here
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Create a DataLoader to get a sample batch for profiling
        self.sample_dataloader = DataLoader(
            self.dataset,
            batch_size=1, # Use batch_size 1 for FLOPs/memory to get per-sample stats
            shuffle=False,
            num_workers=0, # No workers needed for a single sample for profiling
            pin_memory=False
        )
        self.sample_batch = next(iter(self.sample_dataloader))


    def measure_flops(self):
        if summary is None:
            return "N/A" # Skip if torchinfo is not installed
        
        # Prepare a dummy input for FLOPs calculation
        # The model's forward expects a dictionary 'batch' with 'image', 'input_ids', 'attention_mask'
        dummy_images = self.sample_batch['image'].unsqueeze(0).to(self.device)
        dummy_input_ids = self.sample_batch['input_ids'].unsqueeze(0).to(self.device)
        dummy_attention_mask = self.sample_batch['attention_mask'].unsqueeze(0).to(self.device)
        
        dummy_batch_for_flops = {
            'image': dummy_images,
            'input_ids': dummy_input_ids,
            'attention_mask': dummy_attention_mask,
        }

        try:
            # Profiling for the 'retrieval' task path
            self.model.eval() 
            model_summary = summary(
                self.model, 
                input_data=(dummy_batch_for_flops, 'retrieval'), # Pass (batch_dict, task_string) as args
                verbose=0, 
                mode="train" # Use "train" for full graph, not just inference
            )
            total_flops = model_summary.total_flops
            return total_flops
        except Exception as e:
            print(f"Error calculating FLOPs with torchinfo: {e}. Skipping FLOPs measurement.")
            return "N/A"

    def measure_memory_and_time(self, num_warmup=10, num_runs=100):
        # Prepare a sample batch for inference
        sample_images = self.sample_batch['image'].to(self.device)
        sample_input_ids = self.sample_batch['input_ids'].to(self.device)
        sample_attention_mask = self.sample_batch['attention_mask'].to(self.device)
        
        sample_batch_for_inference = {
            'image': sample_images,
            'input_ids': sample_input_ids,
            'attention_mask': sample_attention_mask
        }

        self.model.eval() # Ensure model is in eval mode

        # Warm-up runs for consistent timing
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = self.model(sample_batch_for_inference, task='retrieval') # Run retrieval task
            if self.device.type == 'cuda':
                torch.cuda.synchronize() # Wait for GPU operations to complete

        # Measure peak memory usage
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            with torch.no_grad():
                _ = self.model(sample_batch_for_inference, task='retrieval')
            torch.cuda.synchronize()
            peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            peak_memory_mb = "N/A (CPU)" 

        # Measure inference time
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                
                _ = self.model(sample_batch_for_inference, task='retrieval')
                
                if self.device.type == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize() 
                    timings.append(start_event.elapsed_time(end_event)) # milliseconds
                else:
                    start_time = time.perf_counter()
                    _ = self.model(sample_batch_for_inference, task='retrieval')
                    end_time = time.perf_counter()
                    timings.append((end_time - start_time) * 1000) # milliseconds

        avg_inference_time_ms = np.mean(timings)
        std_inference_time_ms = np.std(timings)

        return peak_memory_mb, avg_inference_time_ms, std_inference_time_ms


    def evaluate(self):
        # 1. Number of Parameters and Model Size
        num_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = num_params * 4 / 1e6 # Assuming float32 (4 bytes/param)

        # 2. FLOPs
        flops = self.measure_flops()

        # 3. Peak Memory and Inference Time (for retrieval task, single batch)
        peak_memory_mb, avg_inference_time_ms, std_inference_time_ms = self.measure_memory_and_time()

        results = {
            "Num Parameters": num_params,
            "Model Size (MB)": f"{model_size_mb:.2f}",
            "FLOPs (for retrieval task)": f"{flops:.2e}" if isinstance(flops, (int, float)) else flops,
            "Peak GPU Memory (MB)": f"{peak_memory_mb:.2f}" if isinstance(peak_memory_mb, (int, float)) else peak_memory_mb,
            "Average Inference Time (ms)": f"{avg_inference_time_ms:.3f}",
            "Std Dev Inference Time (ms)": f"{std_inference_time_ms:.3f}",
            "Rank k": self.model.cross_modal_attention.rank 
        }

        with open(os.path.join(self.save_dir, "efficiency_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        print("\n--- Efficiency Results ---")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("--------------------------\n")
        
        return results

import torch

class GPUMemoryManager:
    def __init__(self, preferred_device=None):
        """
        preferred_device: manually specify device ID, e.g. 0 or 1
        """
        self.device = self._select_device(preferred_device)

    def _select_device(self, preferred_device):
        if torch.cuda.is_available():
            if preferred_device is not None:
                print(f"✅ Using manually specified GPU: {preferred_device}")
                return torch.device(f"cuda:{preferred_device}")
            else:
                # Select least occupied GPU automatically
                device_id = self.get_free_gpu()
                print(f"✅ Auto-selected GPU: {device_id}")
                return torch.device(f"cuda:{device_id}")
        else:
            print("⚠️ CUDA not available, using CPU.")
            return torch.device("cpu")

    def get_free_gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            mem_free = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_free.append(info.free)
            best_gpu = mem_free.index(max(mem_free))
            pynvml.nvmlShutdown()
            return best_gpu
        except Exception as e:
            print("⚠️ pynvml not installed or failed:", e)
            return 0

    def get_device(self):
        return self.device

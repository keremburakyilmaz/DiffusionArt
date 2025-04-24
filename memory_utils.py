import torch
import gc

def free_memory():
    """Free up GPU memory by clearing cache and garbage collecting"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def log_memory_usage(logger=None, prefix=""):
    """Log memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        message = f"{prefix}GPU memory: allocated={allocated:.2f}GB, cached={cached:.2f}GB"
        
        if logger:
            logger.info(message)
        else:
            print(message)
import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
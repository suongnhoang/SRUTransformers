import torch
import random, os
import numpy as np

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    

def get_device(idx=None):
    if torch.cuda.is_available():
        if idx is not None:
            try:
                device = torch.device(f"cuda:{idx}")
                torch.cuda.get_device_name(device)
            except:
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        print(f"[INFO] Using {device}")
        print(torch.cuda.get_device_name(device))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(device)/1024**3,1), 'GB')
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using {device}")
    return device

def reset_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

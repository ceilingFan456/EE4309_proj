import json
import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_jsonl(data, path):
    """Append a list of dictionaries to a JSONL file (one item per line)."""
    with open(path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

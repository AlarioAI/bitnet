import random

import numpy as np
import torch

from bitnet.config import ExperimentConfig

def set_seed(seed_value: int | None):
    if seed_value is None:
        seed_value = ExperimentConfig.SEED
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


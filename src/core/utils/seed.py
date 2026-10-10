import random

import numpy
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    numpy.random.seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

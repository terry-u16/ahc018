import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def conv2d():
    N_IN = 4
    C_IN = 2
    C_OUT = 3
    in_tensor = torch.tensor(np.random.rand(C_IN, N_IN, N_IN), dtype=torch.float32)
    print("=== INPUT ===")
    print(in_tensor.flatten())

    model = nn.Conv2d(C_IN, C_OUT, kernel_size=3, padding="same")
    out_tensor = model(in_tensor)
    print("=== OUTPUT ===")
    print(out_tensor.flatten())

    print("=== MODEL ===")
    for param in model.parameters():
        print(param.flatten())

set_seed()
conv2d()

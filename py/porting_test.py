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

def relu():
    N_IN = 4
    C_IN = 2
    in_array = np.random.rand(C_IN, N_IN, N_IN)
    in_array = in_array * 2 - 1
    in_tensor = torch.tensor(in_array, dtype=torch.float32)
    print("=== INPUT ===")
    print(in_tensor.flatten())

    model = nn.ReLU()
    out_tensor = model(in_tensor)
    print("=== OUTPUT ===")
    print(out_tensor.flatten())

    print("=== MODEL ===")
    for param in model.parameters():
        print(param.flatten())


def batch_norm2d():
    N_IN = 2
    C_IN = 4
    in_tensor = torch.tensor(np.random.rand(1, C_IN, N_IN, N_IN), dtype=torch.float32)
    print("=== INPUT ===")
    print(in_tensor.flatten())

    model = nn.BatchNorm2d(C_IN)

    for _ in range(1):
        _ = model(in_tensor)

    model.eval()
    out_tensor = model(in_tensor)
    print("=== OUTPUT ===")
    print(out_tensor.flatten())

    print("=== MODEL ===")
    print(model.running_mean)
    print(model.running_var)
    print(model.eps)
    for param in model.parameters():
        print(param.flatten())


def upsample2d():
    N_IN = 3
    C_IN = 1
    in_tensor = torch.tensor(np.random.rand(1, C_IN, N_IN, N_IN), dtype=torch.float32)
    print("=== INPUT ===")
    print(in_tensor.flatten())

    model = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    out_tensor = model(in_tensor)
    print("=== OUTPUT ===")
    print(out_tensor.flatten())

set_seed()
#conv2d()
#relu()
#batch_norm2d()
upsample2d()
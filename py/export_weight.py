import base64
import struct
from collections import OrderedDict

import numpy as np
import torch


def pack(value: np.float32) -> bytes:
    return struct.pack('<f', value)

def export_weights(weights: OrderedDict, path: str):
    with open(path, mode="w") as f:
        for key, value in weights.items():
            value = value.flatten().to("cpu").numpy()
            stream = b""

            for v in value:
                stream += pack(v)

            s = base64.b64encode(stream)
            f.write(key)
            f.write("\n")
            f.write(s.decode("utf-8"))
            f.write("\n")

dict = torch.load("data/model_weight.pth")
export_weights(dict, "data/weight_base64.txt")
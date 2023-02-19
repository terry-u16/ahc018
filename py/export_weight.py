import base64
import struct
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image


def pack(value: np.float32) -> bytes:
    return struct.pack('<f', value)

def export_weights(weights: OrderedDict, path: str):
    with open(path, mode="w") as f:
        for key, value in weights.items():
            value = value.flatten().to("cpu").numpy()
            stream = b""

            for v in value:
                stream += pack(v)

            s = base64.b64encode(stream).decode("utf-8")
            f.write(f"\"{key}\" => b\"{s}\",\n")

def generate_testcase(data_dir: str):
    torch.set_printoptions(edgeitems=10000, precision=8)
    image = np.array(Image.open(f"{data_dir}/image_x0/8000.bmp").convert("L"))
    print(torch.tensor(image.flatten() / 255))
    image = np.array(Image.open(f"{data_dir}/image_x1/8000.bmp").convert("L"))
    print(torch.tensor(image.flatten() / 255))
    image = np.array(Image.open(f"{data_dir}/pred/0000.bmp").convert("L"))
    print(torch.tensor(image.flatten() / 255))


dict = torch.load("data/model_weight.pth")
export_weights(dict, "data/weight_base64.txt")
generate_testcase("data")
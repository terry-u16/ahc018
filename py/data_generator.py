import random

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_SIZE = 200
STRIDE = 5
SIZE = RAW_SIZE // STRIDE
SCALING = 5000
DATA_COUNT = 10000

# [8, 256]からサンプルする点の数をランダムに選ぶ
MIN_SAMPLING_POW = 3
MAX_SAMPLING_POW = 8

def read_image(path: str) -> np.ndarray:
    array = np.zeros((SIZE, SIZE), dtype=np.float64)
    with open(path) as f:
        _ = f.readline()

        for row in range(RAW_SIZE):
            row //= STRIDE
            line = list(map(int, f.readline().split()))
            for col, v in enumerate(line):
                col //= STRIDE
                array[row, col] += v / SCALING
            
    array /= STRIDE * STRIDE

    return array

def write_cost_img(path: str, img: np.ndarray):
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img.save(path)


def gen_sampled_img(img: np.ndarray) -> np.ndarray:
    seen = set()
    sum_val = 0.0
    rand_pow = random.uniform(MIN_SAMPLING_POW, MAX_SAMPLING_POW)
    trial_count = int(round(pow(2, rand_pow)))

    for _ in range(trial_count):
        row = random.randrange(0, SIZE - 1)
        col = random.randrange(0, SIZE - 1)
        if not (row, col) in seen:
            seen.add((row, col))
            sum_val += img[row, col]

    avg = sum_val / len(seen)
    sampled_img = np.full_like(img, avg)

    for row, col in seen:
        sampled_img[row, col] = img[row, col]

    return sampled_img

for seed in tqdm(range(10000)):
    img = read_image(f"data/learning_in/{seed:0>4}.txt")
    write_cost_img(f"data/image_y/{seed:0>4}.bmp", img)
    sampled_img = gen_sampled_img(img)
    write_cost_img(f"data/image_x/{seed:0>4}.bmp", sampled_img)

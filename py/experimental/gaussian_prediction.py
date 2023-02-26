import math
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.linalg as linalg
from PIL import Image

RAW_SIZE = 200
SIZE = 40
STRIDE = 200 // SIZE
IMG_NO = 16
CNN_PATH = "data/nn_history/20230221_0037_学習データをnumpyに変更"


def read_image(path: str) -> np.ndarray:
    array = np.zeros((RAW_SIZE, RAW_SIZE), dtype=np.float64)
    with open(path) as f:
        _ = f.readline()

        for row in range(RAW_SIZE):
            line = list(map(int, f.readline().split()))
            for col, v in enumerate(line):
                array[row, col] += v

    return array


def read_sampling_points() -> List[Tuple[int, int]]:
    image = np.array(Image.open(f"{CNN_PATH}/image_x1/{IMG_NO + 8000:0>4}.bmp"))
    points = []

    for i in range(SIZE):
        for j in range(SIZE):
            if image[i, j] == 255:
                points.append((i * STRIDE, j * STRIDE))

    return points


def kernel(
    x0: np.matrix,
    x1: np.matrix,
    theta1: float,
    theta2: float,
    theta3: float,
    i: int,
    j: int,
) -> float:
    # ガウスカーネル
    dx = x0 - x1
    norm = np.multiply(dx, dx).sum()
    ret = theta1 * math.exp(-norm / theta2)
    if i == j:
        ret += theta3
    return ret


def kernel_mat(x: np.matrix, theta1: float, theta2: float, theta3: float) -> np.matrix:
    n = len(x)
    k = np.matrix(list(range(n * n)), dtype=np.float64).reshape((n, n))
    for i in range(n):
        for j in range(n):
            k[i, j] = kernel(x[i], x[j], theta1, theta2, theta3, i, j)

    return k


def kernel_prob(k: np.matrix, y: np.matrix) -> float:
    return -math.log(np.linalg.det(k)) - y.T * np.linalg.inv(k) * y


def grid_search_theta(x: np.matrix, y: np.matrix) -> Tuple[float, float, float]:
    best_t1 = -1
    best_t2 = -1
    best_t3 = -1
    best_prob = -1e100
    for t1_pow in range(3, 10):
        t1 = math.pow(2.0, t1_pow)
        for t2 in range(2, 12):
            t2 = t2 * t2 * 5 * 5
            for t3_pow in range(0, 5):
                t3 = math.pow(2.0, t3_pow)
                k = kernel_mat(x, t1, t2, t3)
                prob = kernel_prob(k, y)

                if best_prob < prob:
                    best_prob = prob
                    best_t1 = t1
                    best_t2 = t2
                    best_t3 = t3
                    print(
                        f"t1: {best_t1}, t2: {best_t2}, t3: {best_t3}, prob: {best_prob}"
                    )

    print(f"t1: {best_t1}, t2: {best_t2}, t3: {best_t3}")
    return best_t1, best_t2, best_t3


def gaussian_process_regression(
    x_test: np.matrix, x_train: np.matrix, y_train: np.matrix
) -> np.matrix:
    n = len(y_train)
    m = len(x_test)

    t1, t2, t3 = grid_search_theta(x_train, y_train)
    k = kernel_mat(x_train, t1, t2, t3)

    # k_inv = np.linalg.inv(k)
    lu_pair = linalg.lu_factor(k)

    # yy = k_inv * y_train
    yy = linalg.lu_solve(lu_pair, y_train)
    mu = np.matrix(list(range(m)), dtype=np.float64).reshape((m, 1))
    var = np.matrix(list(range(m)), dtype=np.float64).reshape((m, 1))

    since = time.perf_counter()
    for i in range(m):
        kk = np.matrix(list(range(n)), dtype=np.float64).reshape((n, 1))

        for j in range(n):
            kk[j, 0] = kernel(x_train[j], x_test[i], t1, t2, t3, j, i + n)

        s = kernel(x_test[i], x_test[i], t1, t2, t3, i + n, i + n)

        mu[i] = kk.T * yy
        # var[i] = s - kk.T * k_inv * kk
        var[i] = s - kk.T * linalg.lu_solve(lu_pair, kk)

    until = time.perf_counter()
    print(until - since)

    return mu, var


def plot(
    y_truth: np.matrix,
    y_mu: np.matrix,
    y_lower: np.matrix,
    y_upper: np.matrix,
    points: List[Tuple[int, int]],
):
    fig = plt.figure(figsize=(17, 9))

    x = []
    y = []
    for i, j in points:
        x.append(j)
        y.append(RAW_SIZE - i - 0.5)

    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(x, y, marker="o", s=5)
    ax.set_title("sampling points")
    ax.set_xlim([0, 250])
    ax.set_ylim([0, 200])
    ax.add_patch(patches.Rectangle(xy=(200, 0), width=50, height=200, fc="k"))

    ax = fig.add_subplot(2, 3, 2)
    heatmap = ax.pcolor(np.flipud(y_truth), cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)
    ax.set_title("ground truth")

    path = f"{CNN_PATH}/pred/{IMG_NO + 0:0>4}.bmp"
    nn_pred = Image.open(path)
    nn_pred = np.array(nn_pred, dtype=np.float64) * 5000 / 255
    ax = fig.add_subplot(2, 3, 3)
    heatmap = ax.pcolor(np.flipud(nn_pred), cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)
    ax.set_title("CNN (U-Net)")

    ax = fig.add_subplot(2, 3, 4)
    heatmap = ax.pcolor(np.flipud(y_lower), cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)
    ax.set_title("gaussian process (-1.0 sigma)")

    ax = fig.add_subplot(2, 3, 5)
    heatmap = ax.pcolor(np.flipud(y_mu), cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)
    ax.set_title("gaussian process (mean)")

    ax = fig.add_subplot(2, 3, 6)
    heatmap = ax.pcolor(np.flipud(y_upper), cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)
    ax.set_title("gaussian process (+1.0 sigma)")

    plt.show()


image = read_image(f"data/learning_in/{IMG_NO + 8000:0>4}.txt")


def f(x: float, y: float) -> float:
    x = int(round(x))
    y = int(round(y))
    return image[x, y]


x_train = read_sampling_points()
y_train = []
x_train_list = x_train
print(f"sampling points: {len(x_train)}")

for i, j in x_train:
    y_train.append(f(i, j))

POWER_RATIO = 2.0


x_train = np.matrix(x_train)
y_train = np.power(np.matrix(y_train).T, 1 / POWER_RATIO)

# 平均を引く
y_mean = y_train.mean()
y_train -= y_mean

x_test = np.matrix(range(RAW_SIZE * RAW_SIZE * 2), dtype=np.float64).reshape(
    RAW_SIZE * RAW_SIZE, 2
)

for i in range(RAW_SIZE):
    for j in range(RAW_SIZE):
        x_test[i * RAW_SIZE + j, 0] = i
        x_test[i * RAW_SIZE + j, 1] = j

(y_pred_mu, y_pred_var) = gaussian_process_regression(x_test, x_train, y_train)
y_pred_mu += y_mean
y_pred_std = np.sqrt(np.where(y_pred_var >= 0, y_pred_var, 0))
y_pred_lower = y_pred_mu - y_pred_std
y_pred_upper = y_pred_mu + y_pred_std
y_pred_mu = np.power(np.array(y_pred_mu).reshape((RAW_SIZE, RAW_SIZE)), POWER_RATIO)
y_pred_lower = np.power(
    np.array(y_pred_lower).reshape((RAW_SIZE, RAW_SIZE)), POWER_RATIO
)
y_pred_upper = np.power(
    np.array(y_pred_upper).reshape((RAW_SIZE, RAW_SIZE)), POWER_RATIO
)
y_pred_lower = np.where(y_pred_lower >= 10, y_pred_lower, 10)
y_pred_upper = np.where(y_pred_upper <= 5000, y_pred_upper, 5000)

plot(image, y_pred_mu, y_pred_lower, y_pred_upper, x_train_list)

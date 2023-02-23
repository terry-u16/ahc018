import math
import random

import matplotlib.pyplot as plt
import numpy as np

RAW_SIZE = 200
SIZE = 40
STRIDE = 200 // SIZE
random.seed(42)


def read_image(path: str) -> np.ndarray:
    array = np.zeros((SIZE, SIZE), dtype=np.float64)
    with open(path) as f:
        _ = f.readline()

        for row in range(RAW_SIZE):
            row //= STRIDE
            line = list(map(int, f.readline().split()))
            for col, v in enumerate(line):
                col //= STRIDE
                array[row, col] += v

    array /= STRIDE * STRIDE

    return array


def kernel(x0: np.matrix, x1: np.matrix) -> float:
    # ガウスカーネル
    THETA1 = 100.0
    THETA2 = 20.0
    dx = x0 - x1
    norm = np.multiply(dx, dx).sum()
    ret = THETA1 * math.exp(-norm / THETA2)
    return ret


def gaussian_process_regression(
    x_test: np.matrix, x_train: np.matrix, y_train: np.matrix
) -> np.matrix:
    n = len(y_train)
    m = len(x_test)
    k = np.matrix(list(range(n * n)), dtype=np.float64).reshape((n, n))
    for i in range(n):
        for j in range(n):
            k[i, j] = kernel(x_train[i], x_train[j])

    print(k)
    k_inv = np.linalg.inv(k)

    yy = k_inv * y_train
    mu = np.matrix(list(range(m)), dtype=np.float64).reshape((m, 1))
    var = np.matrix(list(range(m)), dtype=np.float64).reshape((m, 1))

    for i in range(m):
        kk = np.matrix(list(range(n)), dtype=np.float64).reshape((1, n))

        for j in range(n):
            kk[0, j] = kernel(x_train[j], x_test[i])

        s = kernel(x_test[i], x_test[i])

        mu[i] = kk * yy
        var[i] = s - kk * k_inv * kk.T

    return mu, var


def plot(y_truth: np.matrix, y_mu: np.matrix, y_var: np.matrix):
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1, 3, 1)
    heatmap = ax.pcolor(y_truth, cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)

    ax = fig.add_subplot(1, 3, 2)
    heatmap = ax.pcolor(y_mu, cmap="jet")
    heatmap.set_clim([0, 5000])
    fig.colorbar(heatmap)

    ax = fig.add_subplot(1, 3, 3)
    heatmap = ax.pcolor(y_var, cmap="jet")
    heatmap.set_clim([0, 300])
    fig.colorbar(heatmap)

    plt.show()


image = read_image("data/in/0004.txt")


def f(x: float, y: float) -> float:
    x = int(round(x))
    y = int(round(y))
    return image[x, y]


x_train = []
y_train = []
seen = set()

for _ in range(60):
    while True:
        x1 = random.randint(0, SIZE - 1)
        x2 = random.randint(0, SIZE - 1)
        if not (x1, x2) in seen:
            seen.add((x1, x2))
            break
    x_train.append([x1, x2])
    y_train.append(f(x1, x2))

x_train = np.matrix(x_train)
y_train = np.matrix(y_train).T

print(x_train)
print(y_train)

x_test = np.matrix(range(SIZE * SIZE * 2), dtype=np.float64).reshape(SIZE * SIZE, 2)

for i in range(SIZE):
    for j in range(SIZE):
        x_test[i * SIZE + j, 0] = i
        x_test[i * SIZE + j, 1] = j

(y_pred_mu, y_pred_var) = gaussian_process_regression(x_test, x_train, y_train)
print(y_pred_mu)
print(y_pred_var)
y_pred_mu = np.array(y_pred_mu).reshape((SIZE, SIZE))
y_pred_var = np.array(y_pred_var).reshape((SIZE, SIZE))

plot(image, y_pred_mu, y_pred_var)

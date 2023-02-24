import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SIZE = 200
IMG_NO = 0
random.seed(42)

def read_image(path: str) -> np.ndarray:
    array = np.zeros((SIZE,), dtype=np.float64)
    with open(path) as f:
        _ = f.readline()

        line = list(map(int, f.readline().split()))
        for col, v in enumerate(line):
            array[col] += v

    return array


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
    return -math.log(max(np.linalg.det(k), 1e-100)) - y.T * np.linalg.inv(k) * y


def grid_search_theta(x: np.matrix, y: np.matrix) -> Tuple[float, float, float]:
    best_t1 = -1
    best_t2 = -1
    best_t3 = -1
    best_prob = -1e100
    for t1_pow in range(2, 8):
        t1 = math.pow(2.0, t1_pow)
        for t2_pow in range(3, 6):
            t2 = math.pow(2.0, t2_pow)
            t2 = t2 * t2
            for t3_pow in range(-2, 3):
                t3 = math.pow(2.0, t3_pow)
                k = kernel_mat(x, t1, t2, t3)
                prob = kernel_prob(k, y)

                if best_prob < prob:
                    best_prob = prob
                    best_t1 = t1
                    best_t2 = t2
                    best_t3 = t3
                    print(
                        f"t1: {best_t1}, t2: {best_t2}, t3: {best_t3} prob: {best_prob}"
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

    k_inv = np.linalg.inv(k)

    yy = k_inv * y_train
    mu = np.matrix(list(range(m)), dtype=np.float64).reshape((m, 1))
    var = np.matrix(list(range(m)), dtype=np.float64).reshape((m, 1))

    for i in range(m):
        kk = np.matrix(list(range(n)), dtype=np.float64).reshape((n, 1))

        for j in range(n):
            kk[j, 0] = kernel(x_train[j], x_test[i], t1, t2, t3, j, n + i)

        s = kernel(x_test[i], x_test[i], t1, t2, t3, n + i, n + i)

        mu[i] = kk.T * yy
        var[i] = s - kk.T * k_inv * kk

    return mu, var


def plot(
    y_truth: np.matrix,
    y_mu: np.matrix,
    y_lower: np.matrix,
    y_upper: np.matrix,
    x_train: np.matrix,
    y_train: np.matrix,
):
    x = np.arange(SIZE)
    y_truth = np.array(y_truth).reshape((SIZE,))
    y_mu = np.array(y_mu).reshape((SIZE,))
    y_lower = np.array(y_lower).reshape((SIZE,))
    y_upper = np.array(y_upper).reshape((SIZE,))
    x_train = np.array(x_train).flatten()
    y_train = np.array(y_train).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_truth)
    ax.plot(x, y_mu, color="orange")
    ax.fill_between(x, y_lower, y_upper, color="orange", alpha=0.3)
    ax.scatter(x_train, y_train, color="green")
    plt.show()


image = read_image(f"data/in/{IMG_NO:0>4}.txt")


def f(x: float) -> float:
    x = int(round(x))
    return image[x]


x_train_raw = []
y_train_raw = []

for _ in range(20):
    while True:
        x = random.randint(0, SIZE - 1)

        ok = True
        for xi in x_train_raw:
            if abs(xi - x) < 5:
                ok = False
                break

        if ok:
            break

    noise = random.randint(0, 50)
    x_train_raw.append(x)
    y_train_raw.append(f(x) + noise)

POWER_RATIO = 2.0
x_train = np.matrix(x_train_raw).T
y_train = np.power(np.matrix(y_train_raw).T, 1 / POWER_RATIO)

x_test = np.matrix(range(SIZE), dtype=np.float64).reshape(SIZE, 1)

for i in range(SIZE):
    x_test[i] = i

(y_pred_mu, y_pred_var) = gaussian_process_regression(x_test, x_train, y_train)
SIGMA = 1
y_pred_var = np.where(y_pred_var >= 0, y_pred_var, 0)
y_lower = y_pred_mu - np.sqrt(y_pred_var) * SIGMA
y_upper = y_pred_mu + np.sqrt(y_pred_var) * SIGMA
y_lower = np.where(y_lower >= 0, y_lower, 0)
y_upper = np.where(y_upper <= 5000, y_upper, 5000)
y_pred_mu = np.power(np.array(y_pred_mu).reshape((SIZE,)), POWER_RATIO)
y_lower = np.power(np.array(y_lower).reshape((SIZE,)), POWER_RATIO)
y_upper = np.power(np.array(y_upper).reshape((SIZE,)), POWER_RATIO)

plot(image, y_pred_mu, y_lower, y_upper, x_train_raw, y_train_raw)

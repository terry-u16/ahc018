import heapq
import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SIZE = 200
IMG_NO = 20
random.seed(42)


def read_image(path: str) -> np.ndarray:
    array = np.zeros((SIZE, SIZE), dtype=np.float64)
    with open(path) as f:
        _ = f.readline()

        for row in range(SIZE):
            line = list(map(int, f.readline().split()))
            for col, v in enumerate(line):
                array[row, col] = v

    return array


def dijkstra(dist_map: np.ndarray, si: int, sj: int, gi: int, gj: int) -> np.ndarray:
    INF = 1000000000
    dists = [[INF] * SIZE for _ in range(SIZE)]
    trail = [[(-1, -1)] * SIZE for _ in range(SIZE)]
    queue = [(0, si, sj)]
    dists[si][sj] = 0
    diffs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(queue) > 0:
        (d, i, j) = heapq.heappop(queue)

        if dists[i][j] < d:
            continue

        for (di, dj) in diffs:
            ni = i + di
            nj = j + dj

            if 0 <= ni and ni < SIZE and 0 <= nj and nj < SIZE:
                next_dist = dist_map[ni, nj]
                if dists[ni][nj] > next_dist:
                    dists[ni][nj] = next_dist
                    trail[ni][nj] = i, j
                    heapq.heappush(queue, (next_dist, ni, nj))

    i = gi
    j = gj
    dist_stack = [dist_map[i, j]]

    while (i, j) != (si, sj):
        i, j = trail[i][j]
        dist_stack.append(dist_map[i, j])

    dist_stack.reverse()

    return np.array(dist_stack)


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
        for t2_pow in range(1, 6):
            t2 = math.pow(2.0, t2_pow)
            t2 = t2 * t2
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
    x = np.arange(len(y_truth))
    y_truth = np.array(y_truth).flatten()
    y_mu = np.array(y_mu).flatten()
    y_lower = np.array(y_lower).flatten()
    y_upper = np.array(y_upper).flatten()
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

while True:
    si = random.randint(0, SIZE - 1)
    sj = random.randint(0, SIZE - 1)
    gi = random.randint(0, SIZE - 1)
    gj = random.randint(0, SIZE - 1)

    dist = abs(si - gi) + abs(sj - gj)

    if dist >= 100 and image[si, sj] <= 500 and image[gi, gj] <= 500:
        break

path = dijkstra(image, si, sj, gi, gj)
path_len = len(path)


def f(x: float) -> float:
    x = int(round(x))
    return path[x]


x_train_raw = []
y_train_raw = []
x_train_raw.append(0)
y_train_raw.append(f(0))
x_train_raw.append(len(path) - 1)
y_train_raw.append(f(len(path) - 1))

for _ in range(20):
    while True:
        x = random.randint(1, len(path) - 2)

        ok = True
        for xi in x_train_raw:
            if abs(xi - x) < 5:
                ok = False
                break

        if ok:
            break

    x_train_raw.append(x)
    y_train_raw.append(f(x))

POWER_RATIO = 2.0
x_train = np.matrix(x_train_raw).T
y_train = np.power(np.matrix(y_train_raw).T, 1 / POWER_RATIO)

x_test = np.matrix(range(path_len), dtype=np.float64).reshape(path_len, 1)

for i in range(path_len):
    x_test[i] = i

train_mean = y_train.mean()
y_train -= train_mean
(y_pred_mu, y_pred_var) = gaussian_process_regression(x_test, x_train, y_train)
y_pred_mu += train_mean
SIGMA = 1
y_pred_var = np.where(y_pred_var >= 0, y_pred_var, 0)
y_lower = y_pred_mu - np.sqrt(y_pred_var) * SIGMA
y_upper = y_pred_mu + np.sqrt(y_pred_var) * SIGMA
y_lower = np.where(y_lower >= 0, y_lower, 0)
y_upper = np.where(y_upper <= 5000, y_upper, 5000)
y_pred_mu = np.power(np.array(y_pred_mu), POWER_RATIO)
y_lower = np.power(np.array(y_lower), POWER_RATIO)
y_upper = np.power(np.array(y_upper), POWER_RATIO)

plot(path, y_pred_mu, y_lower, y_upper, x_train_raw, y_train_raw)

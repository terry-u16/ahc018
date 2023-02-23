# https://atsblog.org/pythonbayesian-optimizationgpyopt/
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt

x1_max = 3
x1_min = 0
x2_max = 3
x2_min = 0


def lerp(a, b, t):
    return a + (b - a) * t


# https://yuki67.github.io/post/perlin_noise/
class Perlin:
    slopes = 2 * np.random.random((256, 2)) - 1
    rand_index = np.zeros(512, dtype=np.int8)
    for i, rand in enumerate(np.random.permutation(256)):
        rand_index[i] = rand
        rand_index[i + 256] = rand

    @staticmethod
    def hash(i, j):
        # 前提条件: 0 <= i, j <= 256
        return Perlin.rand_index[Perlin.rand_index[i] + j]

    @staticmethod
    def fade(x):
        return 6 * x**5 - 15 * x**4 + 10 * x**3

    @staticmethod
    def weight(ix, iy, dx, dy):
        # 格子点(ix, iy)に対する(ix + dx, iy + dy)の重みを求める
        ix %= 256
        iy %= 256
        ax, ay = Perlin.slopes[Perlin.hash(ix, iy)]
        return ax * dx + ay * dy

    @staticmethod
    def noise(x1, x2):
        ix = math.floor(x1)
        iy = math.floor(x2)
        dx = x1 - math.floor(x1)
        dy = x2 - math.floor(x2)

        # 重みを求める
        w00 = Perlin.weight(ix, iy, dx, dy)
        w10 = Perlin.weight(ix + 1, iy, dx - 1, dy)
        w01 = Perlin.weight(ix, iy + 1, dx, dy - 1)
        w11 = Perlin.weight(ix + 1, iy + 1, dx - 1, dy - 1)

        # 小数部分を変換する
        wx = Perlin.fade(dx)
        wy = Perlin.fade(dy)

        # 線形補間して返す
        y0 = lerp(w00, w10, wx)
        x2 = lerp(w01, w11, wx)
        w = lerp(y0, x2, wy) + 0.5

        w = 1.0 / (1.0 + math.exp(-3 * (w - 0.25)))
        w = math.pow(w, 3)
        return w


def plot_bo(bo):
    # プロット範囲 (決め打ち)
    X1 = [x1 for x1 in np.arange(x1_min, x1_max, 0.1)]
    X2 = [x2 for x2 in np.arange(x1_min, x2_max, 0.1)]
    X1, X2 = np.meshgrid(X1, X2)

    # 真の関数
    y_truth = np.zeros_like(X1)

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            y_truth[i, j] = Perlin.noise(X1[i, j], X2[i, j])

    y_max = y_truth.max()
    y_min = y_truth.min()

    # 予測値
    x_concat = np.stack([X1, X2], axis=2)
    y_pred = np.zeros_like(X1)
    y_sigma = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            mean, sigma = bo._gp.predict(
                np.array(x_concat[i, j]).reshape(-1, 2), return_std=True
            )
            y_pred[i, j] = mean
            y_sigma[i, j] = sigma

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1, 3, 1)
    heatmap = ax.pcolor(y_truth, cmap="jet")
    heatmap.set_clim([y_min, y_max])
    fig.colorbar(heatmap)

    ax = fig.add_subplot(1, 3, 2)
    heatmap = ax.pcolor(y_pred, cmap="jet")
    heatmap.set_clim([y_min, y_max])
    fig.colorbar(heatmap)

    ax = fig.add_subplot(1, 3, 3)
    heatmap = ax.pcolor(y_sigma, cmap="jet")
    fig.colorbar(heatmap)

    plt.show()


def main():
    # 探索するパラメータと範囲を決める
    pbounds = {"x1": (x1_min, x1_max), "x2": (x2_min, x2_max)}

    # 探索対象の関数と、探索するパラメータと範囲を渡す
    bo = BayesianOptimization(f=Perlin.noise, pbounds=pbounds)
    # 最大化する
    bo.maximize(init_points=0, n_iter=50)

    print("OPTIMIZE: END")

    # 結果をグラフに描画する
    plot_bo(bo)

    # bo._gp.pre

    # plt.cla()

    # アニメーション--------------
    # plot_ani(bo)


if __name__ == "__main__":
    main()

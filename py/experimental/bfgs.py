# https://science-log.com/%E6%95%B0%E5%AD%A6/%E3%80%90%E6%9C%80%E9%81%A9%E5%8C%96%E5%95%8F%E9%A1%8C%E3%81%AE%E5%9F%BA%E7%A4%8E%E3%80%91%E6%BA%96%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%88%E3%83%B3%E6%B3%95%E3%81%A8%E3%82%BB%E3%82%AB%E3%83%B3%E3%83%88/
import numpy as np


def f(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def fx(x, y):
    h = 0.0000001
    return (f(x + h, y) - f(x - h, y)) / (2 * h)


def fy(x, y):
    h = 0.0000001
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


X = np.matrix([[3.0], [0.5]])
rho = 0.9
c = 0.4
B = np.matrix(np.eye(2))
grad = np.matrix([[fx(X[0, 0], X[1, 0])], [(fy(X[0, 0], X[1, 0]))]])

for i in range(1000):
    dfdx = fx(X[0, 0], X[1, 0])
    dfdy = fy(X[0, 0], X[1, 0])
    grad = np.matrix([[dfdx], [dfdy]])
    if np.linalg.norm(grad) < 1e-3:
        print(f"iter: {i}")
        break
    else:
        # Armijo条件を満たすalphaを求める
        alpha = 1.0
        f_init = f(X[0, 0], X[1, 0])
        d = np.linalg.solve(B, -grad)
        dot_tmp = c * grad.T * d
        X_tmp = X + alpha + d
        cnt = 0
        while f(X_tmp[0, 0], X_tmp[1, 0]) > f_init + alpha * dot_tmp:
            alpha *= rho
            X_tmp = X + alpha * d
        X += alpha * d

        old_grad = grad
        grad = np.matrix([[fx(X[0, 0], X[1, 0])], [(fy(X[0, 0], X[1, 0]))]])
        print(alpha)
        s = alpha * d
        y = grad - old_grad
        Bs = B * s
        B += (y * y.T) / (s.T * y) - (Bs * Bs.T) / (s.T * Bs)

print(f"({X[0, 0]}, {X[1, 0]})")
print(f"eval_count: {eval_count}")

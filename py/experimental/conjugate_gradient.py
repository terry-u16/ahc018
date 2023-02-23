import numpy as np


def dot_prod(a: np.matrix, b: np.matrix) -> float:
    return np.sum(np.multiply(a, b))


def solve_cg(a: np.matrix, b: np.matrix, iter: int) -> np.matrix:
    alpha = 0
    x = np.matrix(np.zeros(b.shape))
    m = a.T * (a * x - b)
    t = -dot_prod(m, a.T * (a * x - b)) / dot_prod(m, a.T * a * m)
    x = x + t * m

    for _ in range(iter):
        alpha = dot_prod(m, a.T * a * a.T * (a * x - b)) / dot_prod(m, a.T * a * m)
        m = a.T * (a * x - b) - alpha * m
        t = -dot_prod(m, a.T * (a * x - b)) / dot_prod(m, a.T * a * m)
        x = x + t * m

    return x


A = np.matrix([[2, 4, 1], [1, -2, -2], [-2, -5, 3]])
b = np.matrix([[6], [1], [5]])
x = solve_cg(A, b, 2)
print(x)

expected = np.matrix([[143 / 37], [-39 / 37], [92 / 37]])
print(expected)

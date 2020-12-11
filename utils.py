import numpy as np


def construct_gram_matrix(X):
    """
    construct a gram matrix

    Parameters
    ----------
    X: data matrix; (p, n)

    Returns
    -------
    G: gram matrix; (n, n)
       (G)_{ij} = \| X_i - X_j \|^2
    """
    norms = ((X.T) ** 2 @ np.ones(X.shape[0]))[:, np.newaxis]  # squared norms
    allone = np.ones((X.shape[1], 1))
    G = allone @ norms.T + norms @ allone.T - 2 * X.T @ X
    return G


import math

# Reference: https://conx.readthedocs.io/en/latest/Two-Spirals.html
def spiral_point(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i / (10 * 16) * math.pi
    r = 6.5 * ((10 * 104 - i) / (10 * 104)) + 2 * np.random.uniform()
    x0 = (r * math.cos(φ) * spiral_num) / 13 + 0.5
    x1 = (r * math.sin(φ) * spiral_num) / 13 + 0.5
    return (x0, x1), φ


def spiral_2d(spiral_num):
    xs = []
    cs = []
    for i in range(10 * 97):  # 10 * 97
        x, c = spiral_point(i, spiral_num)
        xs.append(x)
        cs.append(c)

    xs = np.vstack(xs).T
    return xs, cs
import numpy as np
from typing import Optional


def mahalanobis_distance(x: np.ndarray, u: np.ndarray, cov: Optional[np.ndarray] = None, inv_cov_mat: Optional[np.ndarray] = None) -> float:
    if x.shape != u.shape:
        raise RuntimeError(f"x and mean are not the same shape. {x.shape} != {u.shape}")

    if cov is None and inv_cov_mat is None:
        raise RuntimeError(f"Need cov or inv_cov_mat. Got None for both.")
    if inv_cov_mat is None:
        inv_cov_mat = np.linalg.inv(cov.copy())

    return np.sqrt(np.subtract(x, u).T @ inv_cov_mat @ np.subtract(x, u))


def main():
    x_t = np.array([1.6, 1.5]).T
    means = np.array([[0.1, 0.1], [2.1, 1.9], [-1.5, 2.0]])
    cov_mat = np.array([[1.2, 0.4], [0.4, 1.8]])
    inv_cov_mat = np.linalg.inv(cov_mat.copy())
    outputs = []
    for idx, mean_ in enumerate(means):
        mean_ = mean_.T
        outputs.append(mahalanobis_distance(x_t, mean_, inv_cov_mat=inv_cov_mat))
        print(f"w_{idx}: {outputs[-1] ** 2}")

    a = 0


if __name__ == '__main__':
    main()
import numpy as np

X_1 = np.asarray([[0.2, 0.7], [0.3, 0.3], [0.4, 0.5], [0.6, 0.5], [0.1, 0.4]])
X_2 = np.asarray([[0.4, 0.6], [0.6, 0.2], [0.7, 0.4], [0.8, 0.6], [0.7, 0.5]])
A = np.concatenate([np.concatenate([X_1, X_2]), np.expand_dims(np.ones((len(X_1) + len(X_2))), axis=1)], axis=1)
y = np.concatenate([[1] * len(X_1), [-1] * len(X_2)])

w = np.linalg.inv(A.T @ A) @ A.T @ y
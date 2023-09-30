import numpy as np


A = np.array([[2, 1], [1, 1], [3, 1], [4, 1]])

print(f"A = {A}\n\nA.T = {A.T}\n\n")

A_ = A.T @ A
print(f"A.T @ A = {A_}\n\n")

A_inv = np.linalg.inv(A_)
print(f"(A.T @ A)^-1 = {A_inv}\n\n")

y = np.array([1, 1, -1, -1]).T

w = np.linalg.inv(A.T @ A) @ (A.T @ y)
print(f"w = {w}\n\n")

x_test = np.array([1.25, 1]).T

print(f"w.T @ x_test = {w @ x_test}")
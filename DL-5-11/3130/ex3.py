import numpy as np


def sigmoid(z):
    # 1 / (1 + e^-z)
    return 1 / (1 + np.exp(-z))


print(f"sigmoid([0, 2]) = {sigmoid(np.array([0, 2]))}")

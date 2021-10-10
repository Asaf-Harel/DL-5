import numpy as np


def initialize_with_zeros(dim):
    W = np.zeros((dim, dim))
    b = 0
    return W, b


W, b = initialize_with_zeros(2)
print("W = " + str(W))
print("b = " + str(b))

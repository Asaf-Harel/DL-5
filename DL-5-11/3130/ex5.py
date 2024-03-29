import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(X, Y, w, b):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    J = (-1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))

    return A, J


w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([1, 0, 1])
A, cost = forward_propagation(X, Y, w, b)
print("cost = " + str(cost))

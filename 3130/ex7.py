import numpy as np


def initialize_with_zeros(dimx, dimy):
    W = np.zeros((dimx, dimy))
    b = 0
    return W, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(X, Y, w, b):
    """
    :param X: matrix of inputs (n, m)
    :param Y: classification vector (m)
    :param w: weights vector (n, 1)
    :param b: number
    :return: Logistic regression results vector, and the value of the cost function
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))

    return A, cost


def backward_propagation(X, Y, A):
    """
    :param X: matrix of inputs (n, m)
    :param Y: classification vector (m)
    :param A: activation results (m)
    :return: weights derivatives and b derivatives
    """

    m = X.shape[1]
    dz = (1 / m) * (A - Y)
    dw = np.dot(X, dz.T)
    db = np.sum(dz)

    return dw, db


def train(X, Y, num_iterations, learning_rate):
    n = X.shape[0]
    W, b = initialize_with_zeros(n, 1)

    for _ in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)

        W -= learning_rate * dw
        b -= learning_rate * db

    return W, b


def train_adaptive(X, Y, alpha, num_iterations, learning_rate):
    n = X.shape[0]
    W, b = initialize_with_zeros(n, 1)
    alpha_w = np.full((n, 1), alpha)
    alpha_b = alpha

    for _ in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)

        alpha_w = np.where(dw * alpha_w > 0, alpha_w * 1.1, alpha_w * -0.5)
        alpha_b *= 1.1 if (db * alpha_b > 0) else -0.5
        W -= learning_rate * dw
        b -= learning_rate * db

        W -= alpha_w
        b -= alpha_b

    return W, b


X, Y = np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([1, 0, 1])
W, b = train(X, Y, num_iterations=100, learning_rate=0.009)


print("W = " + str(W))
print("b = " + str(b))

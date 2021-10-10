import numpy as np


class Perceptron:
    def __init__(self, X, Y, num_iterations=4000, learning_rate=0.01):
        self._X = X
        self._Y = Y
        self._num_iter = num_iterations
        self._learning_rate = learning_rate

    def _initialize_with_zeros(self, dim):
        W = np.zeros(dim)
        b = 0
        return W, b

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _forward_propagation(self, X, Y, w, b):
        """
        :param X: matrix of inputs (n, m)
        :param Y: classification vector (m)
        :param w: weights vector (n, 1)
        :param b: number
        :return: Logistic regression results vector, and the value of the cost function
        """
        m = X.shape[1]
        A = self._sigmoid(np.dot(w.T, X) + b)
        cost = (-1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))

        return A, cost

    def _backward_propagation(self, X, Y, A):
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

    def train(self):
        n = self._X.shape[0]
        W, b = self._initialize_with_zeros(n)

        for i in range(self._num_iter):
            A, cost = self._forward_propagation(self._X, self._Y, W, b)
            dw, db = self._backward_propagation(self._X, self._Y, A)
            W -= self._learning_rate * dw
            b -= self._learning_rate * db

        self._W = W
        self._b = b

    def predict(self, X):
        A = self._sigmoid(np.dot(self._W.T, X) + self._b)
        return np.where(A > 0.5, 1, 0)

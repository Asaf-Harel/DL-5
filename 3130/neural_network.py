import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.W = None
        self.b = None

    def initialize_with_zeros(self, dim):
        W = np.zeros(dim)
        b = 0
        return W, b

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X, Y, w, b):
        """
        :param X: matrix of inputs (n, m)
        :param Y: classification vector (m)
        :param w: weights vector (n, 1)
        :param b: number
        :return: Logistic regression results vector, and the value of the cost function
        """
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T, X) + b)
        cost = (-1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))

        return A, cost

    def backward_propagation(self, X, Y, A):
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

    def train(self, X, Y, num_iterations, learning_rate):
        n = X.shape[0]
        W, b = self.initialize_with_zeros(n)

        for i in range(num_iterations):
            A, cost = self.forward_propagation(X, Y, W, b)
            dw, db = self.backward_propagation(X, Y, A)
            W -= learning_rate * dw
            b -= learning_rate * db

        self.W = W
        self.b = b

    def predict(self, X):
        A = self.sigmoid(np.dot(self.W.T, X) + self.b)
        return np.where(A > 0.5, 1, 0)

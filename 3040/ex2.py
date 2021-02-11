import numpy as np
import time
import matplotlib.pyplot as plt
import unit10.b_utils as u10


def main():
    X, Y = u10.load_dataB1W4_trainN()
    np.random.seed(1)
    J, dW, db = calc_J_np_v1(X, Y, np.random.randn(len(X), 1), 3)
    print(J)
    print(dW.shape)
    print(db)

    J, dW, db = calc_J_np_v2(X, Y, np.random.randn(len(X), 1), 3)
    print(J)
    print(dW.shape)
    print(db)


def calc_J_np_v1(X, Y, W, b):
    m = len(Y)
    n = len(W)
    J = 0
    dW = np.zeros((n, 1))
    db = 0
    for i in range(m):
        x_i = X[:, i].reshape(len(W), 1)
        y_hat_i = np.dot(W.T, x_i) + b
        diff = float(y_hat_i - Y[i])
        J += (diff ** 2) / m
        dW += (2 * diff / m) * x_i
        db += 2 * diff / m
    return J, dW, db


def calc_J_np_v2(X, Y, W, b):
    m = len(Y)
    y_hat = np.dot(W.T, X) + b
    diff = y_hat - Y
    J = np.sum(diff ** 2) / m
    dW = np.sum(2 * X * diff, axis=1, keepdims=True) / m
    db = np.sum(diff * 2) / m
    return J, dW, db


def train(X, Y, alpha, num_iterations, calc_J):
    m, n, J, costs = len(Y), len(X), 0, []
    W, b, alpha_W, alpha_b = np.zeros((n, 1)), 0, np.full((n, 1), alpha), alpha

    for i in range(1, num_iterations + 1):
        cost, dW, db = calc_J(X, Y, W, b)
        alpha_W = np.where(dW * alpha_W > 0, alpha_W * 1.1, alpha_W * -0.5)
        alpha_b *= 1.1 if (db * alpha_b > 0) else -0.5
        W -= alpha_W
        b -= alpha_b
        if i % (num_iterations // 50) == 0:
            print(f'Iterations: {i} | cost: {cost}')
            costs.append(cost)
    return costs, W, b


if __name__ == '__main__':
    main()

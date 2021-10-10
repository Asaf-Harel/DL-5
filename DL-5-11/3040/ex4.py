import time
import numpy as np
import matplotlib.pyplot as plt
import unit10.b_utils as u10
from ex3 import calc_J_np_v2


def main():
    X, Y = u10.load_dataB1W4_trainN()
    np.random.seed(1)

    tic = time.time()
    costs1, W1, b1 = train_non_vector(X, Y, 0.001, 100000, calc_J_non_vector)
    toc = time.time()
    time_non_vector = 1000 * toc - tic

    tic = time.time()
    costs2, W2, b2 = train_adaptive(X, Y, 0.001, 100000, calc_J_np_v2)
    toc = time.time()
    time_vector = 1000 * toc - tic

    print()
    print(f"Non Vectorized version: {time_non_vector}ms")
    print(f"J={str(costs1[-1])}, w1={str(W1[0])}, w2={str(W1[1])}, w3={str(W1[2])}, w4={str(W1[3])}, b={str(b1)}")
    print()
    print(f"J={str(costs2[-1])}, w1={str(W2[0])}, w2={str(W2[1])}, w3={str(W2[2])}, w4={str(W2[3])}, b={str(b2)}")
    print(f"Vectorized version: {time_vector}ms")

    plt.plot(costs1)
    plt.ylabel('cost')
    plt.xlabel('iterations')


# ------------------------------- Vector -------------------------------------------- #
def train_adaptive(X, Y, alpha, num_iterations, calc_J):
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


# ------------------------------- Non Vector ------------------------------- #
def calc_J_non_vector(X, Y, W, b):
    m = len(Y)
    n = len(W)
    J = 0
    dW = []
    for j in range(n):
        dW.append(0)
    db = 0
    for i in range(m):
        y_hat_i = b
        for j in range(n):
            y_hat_i += W[j] * X[j][i]
        diff = y_hat_i - Y[i]
        J += (diff ** 2) / m
        for j in range(n):
            dW[j] += (2 * diff / m) * X[j][i]
        db += 2 * diff / m
    return J, dW, db


def train_non_vector(X, Y, alpha, num_iterations, calc_J):
    m, n = len(Y), len(X)
    costs, W, alpha_W, b = [], [], [], 0
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    alpha_b = alpha
    for i in range(1, num_iterations + 1):
        cost, dW, db = calc_J(X, Y, W, b)
        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j] * alpha_W[j] > 0 else -0.5
        alpha_b *= 1.1 if db * alpha_b > 0 else -0.5
        for j in range(n):
            W[j] -= alpha_W[j]
        b -= alpha_b
        if i % (num_iterations // 50) == 0:
            print(f'Iteration {i} cost {cost}')
            costs.append(cost)
    return costs, W, b


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10

MAX_D = 1000000


def calc_J(X, Y, W, b):
    m = len(Y)
    n = len(X)
    J = 0
    dW = []
    for j in range(n):
        dW.append(0)
    db = 0

    for i in range(m):
        y_hat_i = b
        for j in range(n):
            y_hat_i += W[j] * X[j][i]
        diff = (float)(y_hat_i - Y[i])
        J += (diff ** 2) / m
        for j in range(n):
            dW[j] += (2 * diff / m) * X[j][i]
        db += 2 * (diff);
    return J, dW, db / m


def calc_J_non_adaptive(X, Y, W, b):
    m = len(Y)
    n = len(X)
    J = 0
    dW = []
    for j in range(n):
        dW.append(0)
    db = 0

    for i in range(m):
        y_hat_i = b
        for j in range(n):
            y_hat_i += W[j] * X[j][i]
        diff = (float)(y_hat_i - Y[i])
        J += (diff ** 2) / m
        for j in range(n):
            dW[j] += (2 * diff / m) * X[j][i]
        db += 2 * (diff / m);
    dW, db = boundary(dW, db)
    return J, dW, db


def train_n_adaptive(X, Y, alpha, num_iterations, calc_J):
    m, n = len(Y), len(X)
    costs, W, alpha_W, b = [], [], [], 0
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    alpha_b = alpha

    for i in range(num_iterations):
        cost, dW, db = calc_J(X, Y, W, b)
        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j] * alpha_W[j] > 0 else -0.5
        alpha_b *= 1.1 if db * alpha_b > 0 else -0.5
        for j in range(n):
            W[j] -= alpha_W[j]
        b -= alpha_b
        if (i % 10000) == 0:
            print(f'Iteration: {str(i)}  cost={str(cost)}')
            costs.append(cost)
    return costs[1:], W, b


def train_non_adaptive(X, Y, alpha, num_iterations, calc_J_non_adaptive):
    m, n = len(Y), len(X)
    costs, W, alpha_W, b = [], [], [], 0
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    alpha_b = alpha

    for i in range(num_iterations):
        cost, dW, db = calc_J_non_adaptive(X, Y, W, b)
        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j] * alpha_W[j] > 0 else -0.5
            W[j] -= alpha_W[j]

        alpha_b *= 1.1 if db * alpha_b > 0 else -0.5
        b -= alpha_b

        if (i % 10000) == 0:
            print(f'Iteration: {str(i)}  cost={str(cost)}')
            costs.append(cost)
    return costs[1:], W, b


def boundary1(d):
    return MAX_D if d > MAX_D else -MAX_D if d < -MAX_D else d


def boundary(dW, db):
    for i in range(len(dW)):
        dW[i] = boundary1(dW[i])
    return dW, boundary1(db)


X, Y = u10.load_dataB1W4_trainN()

costs, W, b = train_non_adaptive(X, Y, 0.0001, 150000, calc_J_non_adaptive)
print(f'w1={str(W[0])}  w2={str(W[1])}  w3={str(W[2])}  w4={str(W[3])}  b={str(b)}')
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 10,000)')
plt.show()

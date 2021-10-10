import numpy as np
import unit10.b_utils as u10


def main():
    X, Y = u10.load_dataB1W4_trainN()
    np.random.seed(1)
    J, dW, db = calc_J_np_v1(X, Y, np.random.randn(len(X), 1), 3)
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


if __name__ == '__main__':
    main()

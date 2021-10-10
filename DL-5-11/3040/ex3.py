import numpy as np
import unit10.b_utils as u10


def main():
    X, Y = u10.load_dataB1W4_trainN()
    np.random.seed(1)
    J, dW, db = calc_J_np_v2(X, Y, np.random.randn(len(X), 1), 3)
    print(J)
    print(dW.shape)
    print(db)


def calc_J_np_v2(X, Y, W, b):
    m = len(Y)
    y_hat = np.dot(W.T, X) + b
    diff = y_hat - Y
    J = np.sum(diff ** 2) / m
    dW = np.sum(2 * X * diff, axis=1, keepdims=True) / m
    db = np.sum(diff * 2) / m
    return J, dW, db


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


def Li(a, b, xi, yi):
    return (a * xi + b - yi) ** 2


def calc_J(X, Y, a, b):
    m = len(Y)
    sumJ = 0
    sumDa = 0
    sumDb = 0
    for i in range(m):
        sumJ += Li(a, b, X[i], Y[i])
        sumDa += 2 * X[i] * (a * X[i] + b - Y[i])
        sumDb += 2 * (a * X[i] + b - Y[i])
    return sumJ / m, sumDa / m, sumDb / m


def train_adaptive(X, Y, alpha, epocs, MyFunc):
    a_min = random.randrange(-10, 10)  # gues the minimum
    b_min = random.randrange(-10, 10)  # gues the minimum

    alpha_a = alpha
    alpha_b = alpha
    for i in range(epocs):
        Fx, Fda, Fdb = MyFunc(X, Y, a_min, b_min)

        if (Fda * alpha_a < 0):
            alpha_a *= 1.1  # stay same direction
        else:
            alpha_a *= -0.5  # change direction
        a_min += np.abs(Fda) * alpha_a

        if (Fdb * alpha_b < 0):
            alpha_b *= 1.1  # stay same direction
        else:
            alpha_b *= -0.5  # change direction
        b_min += np.abs(Fdb) * alpha_b

    return Fx, a_min, b_min


X, Y = u10.load_dataB1W3Ex1()
J, a, b = train_adaptive(X, Y, 0.01, 1000, calc_J)

print('J=' + str(J) + ', a=' + str(a) + ", b=" + str(b))
plt.plot(X, Y, 'r.')
plt.plot([0, 100], [a * 0 + b, 100 * a + b])
plt.show()

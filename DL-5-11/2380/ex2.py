from gradient_descent import train_non_adaptive, boundary
import matplotlib.pyplot as plt

X = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
Y = [230.0588, 160.9912, 150.4624, 124.9425, 127.4042, 95.59201, 69.67605, 40.69738, 28.12561, 14.42037, 7, 0.582744,
     -1.27835, -15.755, -24.692, -23.796, 12.21919, 9.337909, 19.05403, 23.83852, 9.313449, 66.47649, 10.60984,
     77.97216, 27.41264, 149.7796, 173.2468]


def calc_J(X, Y, W, b):
    m = len(Y)
    J = 0
    d_w = []
    for j in range(m):
        d_w.append(0)
    d_b = 0

    for i in range(m):
        y_hat_i = b
        for j in range(m):
            y_hat_i += W[j] * X[j]

        diff = float(y_hat_i - Y[i])
        J += (diff ** 2) / m

        for j in range(m):
            d_w[j] += (2 * diff / m) * X[j]
        d_b += 2 * diff

    return J, d_w, d_b / m


costs, w, b = train_non_adaptive(X, Y, 0.0001, 150000, calc_J)
print(f'w1={str(w[0])}  w2={str(w[1])}  w3={str(w[2])}  w4={str(w[3])}  b={str(b)}')
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 10,000)')
plt.show()

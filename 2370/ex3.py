def Li(a, b, xi, yi):
    return (a * xi + b - yi) ** 2


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

import random

random.seed(5)


def find_f(x):
    """
    :return function and derivative
    """
    f_x = (x ** 3) - (107 * (x ** 2)) - 9 * x + 3
    f_dx = 3 * (x ** 2) - (214 * x) - 9

    return f_x, f_dx


def train_min(alpha, epochs, func):
    """ Find the x of the min point
    :param alpha: change rate
    :param epochs: number of iterations
    :param func: function
    :return: x of the min point, function and derivative
    """

    x_min = 0

    for i in range(epochs):
        f_x, fd_x = func(x_min)

        if fd_x > 0:
            x_min -= fd_x * alpha
        else:
            x_min += fd_x * alpha

    f_x, fd_x = func(x_min)

    return x_min, f_x, fd_x


def train_max(alpha, epochs, func):
    """ Find the x of the max point
    :param alpha: change rate
    :type alpha: float
    :param epochs: number of iterations
    :type epochs: float
    :param func: function
    :type func: <class 'function'>
    :return x of the max point, function and derivative
    """
    x_max = 0
    for i in range(epochs):
        f_x, fd_x = func(x_max)

        if fd_x > 0:
            x_max += fd_x * alpha
        else:
            x_max -= fd_x * alpha

    f_x, fd_x = func(x_max)

    return x_max, f_x, fd_x


print("MIN INFO: ", train_min(0.001, 100, find_f))
print("MAX INFO: ", train_max(0.001, 100, find_f))

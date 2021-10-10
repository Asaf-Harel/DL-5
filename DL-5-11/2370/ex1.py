import numpy as np


# Functions that are in the presentation:
def Func1(x, y):
    f_xy = 4 * x ** 2 + 12 * x * y + 9 * y ** 2
    d_fx = 8 * x + 12 * y
    d_fy = 12 * x + 18 * y
    return f_xy, d_fx, d_fy


def Fun2(a, b):
    g_ab = 2 * a ** 3 + 6 * a * b ** 2 - 3 * b ** 3 - 150 * abs
    g_da = 6 * a ** 2 + 6 * b ** 2 - 150
    g_db = 12 * a * b - 9 * b ** 2
    return g_ab, g_da, g_db


def Func3(x, y):
    Fxy = (2 * x + 3 * y) ** 2
    Fdx = 4 * (2 * x + 3 * y)
    Fdy = 6 * (2 * x + 3 * y)
    return Fxy, Fdx, Fdy


def Func4(a, b, X, y, i):
    F = (1 / 3) * np.sum((a * X + b + y) ** 2)
    DfXi = 2 * a(a * X[i] + b + y) / 3

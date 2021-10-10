import numpy as np
import unit10.c1w2_utils as u10
from PIL import Image
import matplotlib.pyplot as plt


def initialize_with_zeros(dim):
    W = np.zeros(dim)
    b = 0
    return W, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(X, Y, w, b):
    """
    :param X: matrix of inputs (n, m)
    :param Y: classification vector (m)
    :param w: weights vector (n, 1)
    :param b: number
    :return: Logistic regression results vector, and the value of the cost function
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))

    return A, cost


def backward_propagation(X, Y, A):
    """
    :param X: matrix of inputs (n, m)
    :param Y: classification vector (m)
    :param A: activation results (m)
    :return: weights derivatives and b derivatives
    """

    m = X.shape[1]
    dz = (1 / m) * (A - Y)
    dw = np.dot(X, dz.T)
    db = np.sum(dz)

    return dw, db


def train(X, Y, num_iterations, learning_rate):
    n = X.shape[0]
    W, b = initialize_with_zeros(n)

    for i in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)
        W -= learning_rate * dw
        b -= learning_rate * db

    return W, b


def predict(X, W, b):
    A = sigmoid(np.dot(W.T, X) + b)
    return np.where(A > 0.5, 1, 0)


def check_cat(path):
    img = Image.open(path)
    img = img.resize((64, 64), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    my_image = np.array(img).reshape(1, -1).T
    my_predicted_image = predict(my_image, W, b)
    print(
        f'y = {str(np.squeeze(my_predicted_image))}, your algorithm predicts a "{classes[int(np.squeeze(my_predicted_image))].decode("utf-8")}" picture.'
    )


train_X, train_set_y, test_X, test_set_y, classes = u10.load_datasetC1W2()

train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

train_X_flatten = train_X.reshape(train_X.shape[0], -1).T
test_X_flatten = test_X.reshape(test_set_y.shape[0], -1).T

train_set_x = train_X_flatten / 255.0
test_set_x = test_X_flatten / 255.0

W, b = train(train_set_x, train_set_y, num_iterations=4000, learning_rate=0.005)

Y_prediction_test = predict(test_set_x, W, b)
Y_prediction_train = predict(train_set_x, W, b)

check_cat("./cat.jpg")
check_cat("./cat2.jpg")
check_cat("./cat3.jpg")

from DL3 import *
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def start(n):
    print(f"-------------------- Exercise {n} --------------------")


def end():
    print("-------------------------------------------------------\n")


# -------------------- Exercise 2.1 --------------------
# start(2.1)
# mnist = fetch_openml('mnist_784')
# X, Y = mnist["data"], mnist["target"]
# X = X / 255.0 - 0.5
# i = 12
#
# img = X[i:i + 1].to_numpy().reshape(28, 28)
# plt.imshow(img, cmap=matplotlib.cm.binary)
# plt.axis("off")
# plt.show()
# print("Label is: '" + Y[i] + "'")
# end()
#
# # -------------------- Exercise 2.2 --------------------
# start(2.2)
# digits = 10
# examples = Y.shape[0]
# Y = np.array(Y).reshape(1, examples)
# Y_new = np.eye(digits)[Y.astype('int32')]
# Y_new = Y_new.T.reshape(digits, examples)
# print(Y_new[:, 12])
# end()
#
# # -------------------- Exercise 2.3 --------------------
# start(2.3)
# m = 60000
# X = np.array(X)
#
# m_test = X.shape[0] - m
# X_train, X_test = X[:m].T, X[m:].T
# Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]
#
np.random.seed(111)
# shuffle_index = np.random.permutation(m)
# X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
# i = 12
#
# plt.imshow(X_train[:, i].reshape(28, 28), cmap=matplotlib.cm.binary)
# plt.axis("off")
# plt.show()
# print(Y_train[:, i])
# end()
#
# # -------------------- Exercise 2.4 --------------------
np.random.seed(1)
hidden_layer = DLLayer("Softmax 1", 64, (784,), "sigmoid", "./weights/Layer1.h5", 1)
softmax_layer = DLLayer("Softmax 1", 10, (64,), "softmax", "./weights/Layer2.h5", 1)
model = DLModel()
model.add(hidden_layer)
model.add(softmax_layer)
model.compile("categorical_cross_entropy")
# costs = model.train(X_train, Y_train, 2000)
#
# plt.plot(np.squeeze(costs))
# plt.ylabel('cost')
# plt.xlabel('iterations')
# plt.title("Learning rate =" + str(1))
# plt.show()
#
# # -------------------- Exercise 2.5 --------------------
# start(2.5)
#
#
# def predict_softmax(X, Y, model):
#     AL = model.predict(X)
#     predictions = np.argmax(AL, axis=0)
#     labels = np.argmax(Y, axis=0)
#     return confusion_matrix(predictions, labels)

#
# print('Deep train accuracy')
# pred_train = predict_softmax(X_train, Y_train, model)
# print(pred_train)
# print('Deep test accuracy')
#
# pred_test = predict_softmax(X_test, Y_test, model)
# print(pred_test)
#
# i = 4
# print('train', str(i), str(pred_train[i][i] / np.sum(pred_train[:, i])))
# print('test', str(i), str(pred_test[i][i] / np.sum(pred_test[:, i])))
# end()

# -------------------- Exercise 2.6 --------------------
start(2.6)


def predict(X, Y, model):
    AL = model.predict(X)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)

    return predictions, labels


num_px = 28
img_path = 'images/3.jpeg'  # full path of the rgb image
my_label_y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # change the 1’s position to fit image

image = Image.open(img_path)
image28 = image.resize((num_px, num_px), Image.ANTIALIAS)  # resize to 28X28
plt.imshow(image)

# Before scale
plt.show()
plt.imshow(image28)  # After scale
plt.show()

gray_image = ImageOps.grayscale(image28)

# grayscale – to fit to training data
my_image = np.reshape(gray_image, (num_px * num_px, 1))
my_label_y = np.reshape(my_label_y, (10, 1))
my_image = my_image / 255.0 - 0.5

# normalize
p, l = predict(my_image, my_label_y, model)
print('real:', l, 'prediction:', p)
end()

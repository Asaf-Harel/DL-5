from DL3 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from PIL import Image, ImageOps


def predict_softmax(X, Y, model):
    AL = model.predict(X)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    return predictions, labels


# mnist = fetch_openml('mnist_784')
# X, Y = mnist["data"], mnist["target"]
# X = X / 255.0 - 0.5
#
# digits = 10
# examples = Y.shape[0]
# Y = np.array(Y).reshape(1, examples)
# Y_new = np.eye(digits)[Y.astype('int32')]
# Y_new = Y_new.T.reshape(digits, examples)
# print(Y_new[:, 12])
#
# m = 60000
# X = np.array(X)
# m_test = X.shape[0] - m
# X_train, X_test = X[:m].T, X[m:].T
# Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]

np.random.seed(111)
# shuffle_index = np.random.permutation(m)
# X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

np.random.seed(1)
hidden_layer = DLLayer("Softmax 1", 64, (784,), "sigmoid", "./weights/Layer1.h5", 1)
softmax_layer = DLLayer("Softmax 1", 10, (64,), "softmax", "./weights/Layer2.h5", 1)

model = DLModel()
model.add(hidden_layer)
model.add(softmax_layer)
model.compile("categorical_cross_entropy")

# p, l = predict_softmax(X_test, Y_test, model)
#
# for i in range(l.shape[0]):
#     print('real:', l[i], '| pred:', p[i])

num_px = 28
img_path = 'images/5.jpg'  # full path of the rgb image
my_label_y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # change the 1’s position to fit image

image = Image.open(img_path)
image28 = image.resize((num_px, num_px), Image.ANTIALIAS)  # resize to 28X28
plt.imshow(image)
plt.show()

# Before scale
plt.imshow(image28)  # After scale
plt.show()

gray_image = image28.convert('L')
plt.imshow(gray_image)  # After scale
plt.show()

# grayscale – to fit to training data
my_image = np.reshape(gray_image, (num_px * num_px, 1))
my_label_y = np.reshape(my_label_y, (10, 1))
my_image = my_image / 255.0 - 0.5

# normalize
p, l = predict_softmax(my_image, my_label_y, model)
print()
print(l, p)

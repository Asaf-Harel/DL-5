import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from DL import DLLayer, DLModel
from sklearn.metrics import confusion_matrix

data = pd.read_csv('fer2013.csv')

emotions_names = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
data['emotion_name'] = data['emotion'].map(emotions_names)

pixels_values = data.pixels.str.split(" ").tolist()
pixels_values = pd.DataFrame(pixels_values, dtype=np.int8)

images = pixels_values.values
images = images.astype(float)

# -------------------------- Standardizing images --------------------------
# each_pixel_mean = images.mean(axis=0)
# each_pixel_std = np.std(images, axis=0)
# images = np.divide(np.subtract(images, each_pixel_mean), each_pixel_std)

image_pixels = images.shape[1]
labels_flat = data["emotion"].values.ravel()
labels_count = np.unique(labels_flat).shape[0]


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
    return labels_one_hot


labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

m = 32298

X_train, Y_train = images[:m].T, labels[:m].T
X_test, Y_test = images[m:].T, labels[m:].T
print(X_train.shape, Y_train.shape)

# --------------------------------------- Training ---------------------------------------------------------
model = DLModel()
model.add(DLLayer("Hidden 1", 180, (2304,), "sigmoid", "random", learning_rate=0.01))
model.add(DLLayer("Output", 7, (180,), "softmax", "He"))

model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 1500)

model.save_weights('weights2')

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()


def predict_softmax(X, Y, model):
    AL = model.predict(X)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    return confusion_matrix(predictions, labels)


print('Deep train accuracy')
pred_train = predict_softmax(X_train, Y_train, model)
print(pred_train)

print('Deep test accuracy')
pred_test = predict_softmax(X_test, Y_test, model)
print(pred_test)

i = 4
print('train', str(i), str(pred_train[i][i] / np.sum(pred_train[:, i])))
print('test', str(i), str(pred_test[i][i] / np.sum(pred_test[:, i])))

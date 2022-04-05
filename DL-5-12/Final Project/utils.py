import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from PIL import Image
import random

import tensorflow as tf
from tensorflow.keras import layers, models, losses


def extract_rgb(image: np.ndarray):
    rgb = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):
                rgb.append(image[i, j, k])
    return np.array(rgb)

    # def rgb_hist(array: np.ndarray):
    pass


def show_hist(array: np.ndarray, title='histogram', bins=256):
    plt.hist(array, bins=bins)
    plt.title(title)
    plt.show()


def get_data():
    data_path = path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    CLASS_NAMES = os.listdir(data_path)
    classes = {}

    for i in range(len(CLASS_NAMES)):
        classes[CLASS_NAMES[i]] = i

    X = []
    Y = []
    for class_name in CLASS_NAMES:
        for file in os.listdir(path.join(data_path, class_name)):
            if '.png' in file:
                image_path = path.join(data_path, class_name, file)
                X.append(np.array(Image.open(image_path).resize((220, 220))))
                Y.append(classes[class_name])

    temp = list(zip(X, Y))
    random.shuffle(temp)
    X, Y = zip(*temp)
    X = np.array(X)
    Y = np.array(Y)

    n = round(X.shape[0] * 0.7)

    X_train, Y_train = X[:n], Y[:n]
    X_test, Y_test = X[n:], Y[n:]

    return (X_train, Y_train), (X_test, Y_test)


def create_model(weights_path: str):
    model = models.Sequential()
    model.add(
        layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.load_weights(weights_path)

    return model


def get_image(X: np.ndarray):
    return tf.pad(X, [[0, 0], [2, 2], [2, 2], [0, 0]]) / 255


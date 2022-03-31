import numpy as np
import matplotlib.pyplot as plt


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
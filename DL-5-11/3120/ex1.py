import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

red1, green1 = 0, 0
raccoon = Image.open(r'../unit10/Raccoon.png')

plt.imshow(raccoon)
plt.show()

array = np.array(raccoon)
tic = time.time()

for r in range(len(array)):
    for c in range(len(array[0])):
        if array[r][c][0] > array[r][c][1]:
            red1 += 1
        elif array[r][c][1] > array[r][c][0]:
            green1 += 1

# non vectorized code
toc = time.time()
print("Non Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(
    1000 * (toc - tic)) + "ms")
tic = time.time()

# vectorized code
toc = time.time()
red_arr = array[:, :, 0]
green_arr = array[:, :, 1]
red1 = np.sum(red_arr > green_arr)
green1 = np.sum(green_arr > red_arr)
print("Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(
    1000 * (toc - tic)) + "ms")

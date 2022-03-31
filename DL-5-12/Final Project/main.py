from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import utils

image = Image.open('data/trojans/YouAreAnIdiot.png')

width, height = image.size

image_arr = np.array(image)

section = height // 3

top = image_arr[:section]
bottom = image_arr[section * 2:]

top_center = image_arr[section:(section + (section // 2))]
bottom_center = image_arr[(section + (section // 2)):section * 2]

print('top', top.shape)
print('top center', top_center.shape)
print('bottom center', bottom_center.shape)
print('bottom', bottom.shape)
print('total height', top.shape[0] + top_center.shape[0] + bottom_center.shape[0] + bottom.shape[0])

top_rgb = utils.extract_rgb(top)
top_rgb_hist = np.histogram(top_rgb, bins=256)

print(top_rgb_hist[0].shape)
utils.show_hist(top_rgb, 'top')
# [print(x, end=' ') for x in top_rgb_hist[0]]
# [print(x, end=' ') for x in top_rgb_hist[1]]


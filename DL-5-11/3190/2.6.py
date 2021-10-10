from DL3 import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

np.random.seed(1)
hidden_layer = DLLayer("Softmax 1", 64, (784,), "sigmoid", "./weights/Layer1.h5", 1)
softmax_layer = DLLayer("Softmax 1", 10, (64,), "softmax", "./weights/Layer2.h5", 1)
model = DLModel()
model.add(hidden_layer)
model.add(softmax_layer)
model.compile("categorical_cross_entropy")

num_px = 28
i = 3
img_path = f'images/{i}.jpg'  # full path of the rgb image
my_label_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # change the 1’s position to fit image
my_label_y[i] = 1

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
AL = model.predict(my_image)
prediction = np.argmax(AL, axis=0)
label = np.argmax(my_label_y, axis=0)

result = ':)' if label == prediction else ':('

print('real:', label, 'prediction:', prediction, result)

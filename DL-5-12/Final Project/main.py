import numpy as np
import utils
from binvis import converter
import matplotlib.pyplot as plt

import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix

classes = ['malware', 'normal']

model = utils.create_model('weights/weights.h5')

file_path = '/Users/asafharel/Desktop/MalwareDatabase/Endermanch@000.exe'

image = converter.convert_to_image(256, file_path, save_file=False)

plt.imshow(image)
plt.show()

image = np.array(image.resize((220, 220)))

x_test = utils.get_image(np.array([image]))

y_pred = model.predict(x_test).argmax(axis=1)[0]

print(classes[y_pred])


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from DL import DLLayer, DLModel

model = DLModel()
model.add(DLLayer("Hidden 1", 180, (2304,), "sigmoid", "./weights/Layer1.h5", learning_rate=0.001))
model.add(DLLayer("Output", 7, (180,), "softmax", "./weights/Layer2.h5"))


# --------------------------------------- Testing ---------------------------------------------------------

def predict_emotion(image_path):
    emotions_names = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    image = Image.open(image_path)
    image48 = image.resize((48, 48), Image.ANTIALIAS)

    gray_image = ImageOps.grayscale(image48)

    plt.imshow(gray_image)
    plt.show()

    my_image = np.reshape(gray_image, (48 * 48, 1))
    each_pixel_mean = my_image.mean(axis=0)
    each_pixel_std = np.std(my_image, axis=0)
    my_image = np.divide(np.subtract(my_image, each_pixel_mean), each_pixel_std)
    my_image = my_image.astype('float32')

    AL = model.predict(my_image)
    prediction = np.argmax(AL, axis=0)

    print(emotions_names[prediction[0]])


predict_emotion('./images/sad.jpg')
